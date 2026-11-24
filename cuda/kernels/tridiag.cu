// -----------------------------------------------------------------------------
// Tridiagonal factorization and Thomas solves for the ADI time step.
//
// Thread mapping: one thread per grid line, 32-thread blocks so the few lines
// spread over all SMs. solve_s: one thread per column, sweeping down i; a warp
// reads 32 neighbouring columns, so every access is coalesced. solve_v: one
// thread per row, sweeping along j; the right-hand side is read with a stride of
// `cols`, while A2 and its factors are the same for every row >= 1, so all
// lanes of a warp read the same factor (a broadcast).
//
// The systems are (I - c*A) x = r with sub-diagonal a = (-c)*lower, diagonal
// b = 1 - c*diag, super-diagonal u = (-c)*upper, exactly as tridiag.py builds
// them. Factors (den, c* = u/den) depend only on the matrix and are computed
// once per value of c; each solve then runs the forward sweep
// d*_k = (r_k - a_k d*_{k-1}) / den_k and the back sweep x_k = d*_k - c*_k x_{k+1}.
//
// Project: ADI finite-difference methods for American options under Heston
//          Math/CS 714, UW–Madison, Fall 2026
// Author:  Ganesh Arivoli <arivoli@wisc.edu>
// -----------------------------------------------------------------------------
#include "kernels/kernels.cuh"

#include <utility>

namespace adi {

namespace {

constexpr int kLineBlock = 32;  // threads per block for thread-per-line kernels (decision 2)

int line_blocks(int lines) { return (lines + kLineBlock - 1) / kLineBlock; }

#ifndef ADI_RECOMPUTE_FACTORS  // the precomputed factors (default)

// What the factorization stores for a pivot, and how a solve applies it.
#ifdef ADI_EXACT
__device__ __forceinline__ real stored_pivot(real den) { return den; }
__device__ __forceinline__ real apply_pivot(real x, real pivot) { return x / pivot; }  // as Python
#else
__device__ __forceinline__ real stored_pivot(real den) { return real(1) / den; }
__device__ __forceinline__ real apply_pivot(real x, real pivot) { return x * pivot; }  // no division
#endif

// One column of (I - c*A1): den and c* down i.
__global__ void factorize_s_kernel(OperatorView op, real c, real* cstar, real* pivot) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= op.cols) return;
    real cstar_prev = 0;
    for (int i = 0; i < op.rows; ++i) {
        const int k = i * op.cols + j;
        const real a = -c * op.a1_lower[k];
        const real b = real(1) - c * op.a1_diag[k];
        const real u = -c * op.a1_upper[k];
        const real den = (i == 0) ? b : b - a * cstar_prev;
        cstar_prev = u / den;
        cstar[k] = cstar_prev;
        pivot[k] = stored_pivot(den);
    }
}

// The two sequences of (I - c*A2) along j: thread 0 for the Dirichlet row 0 (zero coefficients, an
// identity solve done with the same arithmetic as Python), thread 1 for every row i >= 1.
__global__ void factorize_v_kernel(OperatorView op, real c, real* cstar, real* pivot) {
    const int t = threadIdx.x;
    if (t > 1) return;
    const bool dirichlet = (t == 0);
    real cstar_prev = 0;
    for (int j = 0; j < op.cols; ++j) {
        const real a = -c * (dirichlet ? real(0) : op.a2_lower[j]);
        const real b = real(1) - c * (dirichlet ? real(0) : op.a2_diag[j]);
        const real u = -c * (dirichlet ? real(0) : op.a2_upper[j]);
        const real den = (j == 0) ? b : b - a * cstar_prev;
        cstar_prev = u / den;
        cstar[t * op.cols + j] = cstar_prev;
        pivot[t * op.cols + j] = stored_pivot(den);
    }
}

#endif  // ADI_RECOMPUTE_FACTORS

// (I - c*A1) X = R, one thread per column. The forward sweep stores d* in X; the back sweep overwrites it.
__global__ void solve_s_kernel(OperatorView op, FactorView f, const real* R, real* X) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= op.cols) return;
    const int cols = op.cols;

    real d_prev = 0;
#ifdef ADI_RECOMPUTE_FACTORS
    real cstar_prev = 0;
#endif
    for (int i = 0; i < op.rows; ++i) {
        const int k = i * cols + j;
        const real a = -f.c * ld(op.a1_lower + k);
        const real r = ld(R + k);
#ifdef ADI_RECOMPUTE_FACTORS
        const real b = real(1) - f.c * ld(op.a1_diag + k);
        const real u = -f.c * ld(op.a1_upper + k);
        const real den = (i == 0) ? b : b - a * cstar_prev;
        cstar_prev = u / den;
        f.scratch[k] = cstar_prev;
        const real d = (i == 0) ? r / den : (r - a * d_prev) / den;
#else
        const real p = ld(f.s_pivot + k);
        const real d = (i == 0) ? apply_pivot(r, p) : apply_pivot(r - a * d_prev, p);
#endif
        X[k] = d;
        d_prev = d;
    }

    real x_next = d_prev;  // x at the last row is d* there, already in X
    for (int i = op.rows - 2; i >= 0; --i) {
        const int k = i * cols + j;
#ifdef ADI_RECOMPUTE_FACTORS
        const real cs = f.scratch[k];
#else
        const real cs = ld(f.s_cstar + k);
#endif
        x_next = X[k] - cs * x_next;
        X[k] = x_next;
    }
}

// (I - c*A2) X = Y - c*A2X, one thread per row; the right-hand side is formed on the fly.
__global__ void solve_v_kernel(OperatorView op, FactorView f, const real* Y, const real* A2X, real* X) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= op.rows) return;
    const int cols = op.cols;
    const int row = i * cols;
    const bool dirichlet = (i == 0);
#ifndef ADI_RECOMPUTE_FACTORS
    const real* cstar = f.v_cstar + (dirichlet ? 0 : cols);
    const real* pivot = f.v_pivot + (dirichlet ? 0 : cols);
#endif

    real d_prev = 0;
#ifdef ADI_RECOMPUTE_FACTORS
    real cstar_prev = 0;
#endif
    for (int j = 0; j < cols; ++j) {
        const real a = -f.c * (dirichlet ? real(0) : ld(op.a2_lower + j));
        const real r = A2X ? ld(Y + row + j) - f.c * ld(A2X + row + j) : ld(Y + row + j);
#ifdef ADI_RECOMPUTE_FACTORS
        const real b = real(1) - f.c * (dirichlet ? real(0) : ld(op.a2_diag + j));
        const real u = -f.c * (dirichlet ? real(0) : ld(op.a2_upper + j));
        const real den = (j == 0) ? b : b - a * cstar_prev;
        cstar_prev = u / den;
        f.scratch[row + j] = cstar_prev;
        const real d = (j == 0) ? r / den : (r - a * d_prev) / den;
#else
        const real p = ld(pivot + j);
        const real d = (j == 0) ? apply_pivot(r, p) : apply_pivot(r - a * d_prev, p);
#endif
        X[row + j] = d;
        d_prev = d;
    }

    real x_next = d_prev;
    for (int j = cols - 2; j >= 0; --j) {
#ifdef ADI_RECOMPUTE_FACTORS
        const real cs = f.scratch[row + j];
#else
        const real cs = ld(cstar + j);
#endif
        x_next = X[row + j] - cs * x_next;
        X[row + j] = x_next;
    }
}

}  // namespace

Tridiag::Tridiag(const OperatorView& op) : op_(op) {
#ifdef ADI_RECOMPUTE_FACTORS
    scratch_ = DeviceBuffer<real>(static_cast<std::size_t>(op.rows) * op.cols);
#endif
}

int Tridiag::prepare(real c, cudaStream_t stream) {
    for (std::size_t s = 0; s < sets_.size(); ++s) {
        if (sets_[s].c == c) return static_cast<int>(s);
    }
    Set set;
    set.c = c;
#ifndef ADI_RECOMPUTE_FACTORS
    const std::size_t n = static_cast<std::size_t>(op_.rows) * op_.cols;
    set.s_cstar = DeviceBuffer<real>(n);
    set.s_pivot = DeviceBuffer<real>(n);
    set.v_cstar = DeviceBuffer<real>(2 * static_cast<std::size_t>(op_.cols));
    set.v_pivot = DeviceBuffer<real>(2 * static_cast<std::size_t>(op_.cols));
    factorize_s_kernel<<<line_blocks(op_.cols), kLineBlock, 0, stream>>>(op_, c, set.s_cstar.data(),
                                                                          set.s_pivot.data());
    CUDA_CHECK_LAST();
    factorize_v_kernel<<<1, 2, 0, stream>>>(op_, c, set.v_cstar.data(), set.v_pivot.data());
    CUDA_CHECK_LAST();
#else
    (void)stream;
#endif
    sets_.push_back(std::move(set));
    return static_cast<int>(sets_.size() - 1);
}

FactorView Tridiag::view(int set) {
    Set& s = sets_.at(static_cast<std::size_t>(set));
    return {s.s_cstar.data(), s.s_pivot.data(), s.v_cstar.data(), s.v_pivot.data(), scratch_.data(), s.c};
}

void Tridiag::solve_s(int set, const real* R, real* X, cudaStream_t stream) {
    solve_s_kernel<<<line_blocks(op_.cols), kLineBlock, 0, stream>>>(op_, view(set), R, X);
    CUDA_CHECK_LAST();
}

void Tridiag::solve_v(int set, const real* Y, const real* A2X, real* X, cudaStream_t stream) {
    solve_v_kernel<<<line_blocks(op_.rows), kLineBlock, 0, stream>>>(op_, view(set), Y, A2X, X);
    CUDA_CHECK_LAST();
}

}  // namespace adi
