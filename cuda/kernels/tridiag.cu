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
// With so few threads, each sweep would wait on global memory at every element.
// Its loads do not depend on the chain, so they are issued ADI_PREFETCH_DEPTH
// elements ahead into two register buffers (pipelined_sweep); the arithmetic,
// and so every result, is unchanged.
//
// Project: ADI finite-difference methods for American options under Heston
//          Math/CS 714, UW–Madison, Fall 2026
// Author:  Ganesh Arivoli <arivoli@wisc.edu>
// -----------------------------------------------------------------------------
#include "kernels/kernels.cuh"

#include <stdexcept>
#include <utility>

namespace adi {

namespace {

constexpr int kLineBlock = 32;  // threads per block for thread-per-line kernels (decision 2)

#ifndef ADI_PREFETCH_DEPTH
#define ADI_PREFETCH_DEPTH 8  // set by CMake; 8 is the fastest depth without spills in every build
#endif
constexpr int kPrefetch = ADI_PREFETCH_DEPTH;  // elements per chunk of loads issued ahead (pipelined_sweep)
static_assert(kPrefetch >= 1 && kPrefetch <= 32, "ADI_PREFETCH_DEPTH must be 1 to 32");

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

// Runs step(t, load(t)) for t = 0, 1, ..., n-1 in order, with the loads issued up to 2D elements ahead.
// A thread-per-line sweep is a chain of dependent steps, each waiting on its global loads; with one warp
// per SM nothing else hides that wait. The loads do not depend on the chain, so they are issued in chunks
// of D into two register buffers: chunk k+1 is loaded into one buffer while chunk k is computed from the
// other. Both buffers are fully unrolled and never rotate, so no register is moved or overwritten while
// its load is still in flight. load() must only load (its results are used D or more elements later);
// step() does all the arithmetic, in the same order as without the buffers. The last < 2D elements, if
// any, run one at a time.
template <int D, class Load, class Step>
__device__ __forceinline__ void pipelined_sweep(int n, Load load, Step step) {
    using In = decltype(load(0));
    In a[D], b[D];
    int t = 0;
    bool a_ready = (n >= D);  // a holds elements t .. t+D-1
    if (a_ready) {
#pragma unroll
        for (int u = 0; u < D; ++u) a[u] = load(u);
    }
    while (a_ready && t + 2 * D <= n) {
#pragma unroll
        for (int u = 0; u < D; ++u) b[u] = load(t + D + u);
#pragma unroll
        for (int u = 0; u < D; ++u) step(t + u, a[u]);
        a_ready = (t + 3 * D <= n);
        if (a_ready) {
#pragma unroll
            for (int u = 0; u < D; ++u) a[u] = load(t + 2 * D + u);
        }
#pragma unroll
        for (int u = 0; u < D; ++u) step(t + D + u, b[u]);
        t += 2 * D;
    }
    if (a_ready) {
#pragma unroll
        for (int u = 0; u < D; ++u) step(t + u, a[u]);
        t += D;
    }
    for (; t < n; ++t) step(t, load(t));
}

// What each sweep loads per element. r0 is R (along s) or Y (along v); r1 is A2X (along v only).
#ifdef ADI_RECOMPUTE_FACTORS
struct ForwardIn {
    real lower, diag, upper, r0, r1;
};
#else
struct ForwardIn {
    real lower, r0, r1, pivot;
};
#endif
struct BackIn {
    real d, cstar;  // d* from the forward sweep, and c*
};

// (I - c*A1) X = R, one thread per column. The forward sweep stores d* in X; the back sweep overwrites it.
__global__ void solve_s_kernel(OperatorView op, FactorView f, const real* __restrict__ R, real* __restrict__ X) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= op.cols) return;
    const int cols = op.cols;
    const real c = f.c;

    real d_prev = 0;
#ifdef ADI_RECOMPUTE_FACTORS
    real* __restrict__ scratch = f.scratch;
    real cstar_prev = 0;
    pipelined_sweep<kPrefetch>(
        op.rows,
        [&](int i) {
            const int k = i * cols + j;
            return ForwardIn{ld(op.a1_lower + k), ld(op.a1_diag + k), ld(op.a1_upper + k), ld(R + k), real(0)};
        },
        [&](int i, const ForwardIn& in) {
            const real a = -c * in.lower;
            const real b = real(1) - c * in.diag;
            const real u = -c * in.upper;
            const real den = (i == 0) ? b : b - a * cstar_prev;
            cstar_prev = u / den;
            scratch[i * cols + j] = cstar_prev;
            const real d = (i == 0) ? in.r0 / den : (in.r0 - a * d_prev) / den;
            X[i * cols + j] = d;
            d_prev = d;
        });
#else
    pipelined_sweep<kPrefetch>(
        op.rows,
        [&](int i) {
            const int k = i * cols + j;
            return ForwardIn{ld(op.a1_lower + k), ld(R + k), real(0), ld(f.s_pivot + k)};
        },
        [&](int i, const ForwardIn& in) {
            const real a = -c * in.lower;
            const real d = (i == 0) ? apply_pivot(in.r0, in.pivot) : apply_pivot(in.r0 - a * d_prev, in.pivot);
            X[i * cols + j] = d;
            d_prev = d;
        });
#endif

    // Back sweep over i = rows-2, ..., 0 (element t is row rows-2-t). x at the last row is d* there.
    // X is read with ordinary loads: this kernel wrote it, and each element is read before it is overwritten.
    real x_next = d_prev;
    const int last = op.rows - 2;
    pipelined_sweep<kPrefetch>(
        op.rows - 1,
        [&](int t) {
            const int k = (last - t) * cols + j;
#ifdef ADI_RECOMPUTE_FACTORS
            return BackIn{X[k], scratch[k]};
#else
            return BackIn{X[k], ld(f.s_cstar + k)};
#endif
        },
        [&](int t, const BackIn& in) {
            x_next = in.d - in.cstar * x_next;
            X[(last - t) * cols + j] = x_next;
        });
}

// (I - c*A2) X = Y - c*A2X, one thread per row; the right-hand side is formed on the fly.
__global__ void solve_v_kernel(OperatorView op, FactorView f, const real* __restrict__ Y,
                               const real* __restrict__ A2X, real* __restrict__ X) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= op.rows) return;
    const int cols = op.cols;
    const int row = i * cols;
    const bool dirichlet = (i == 0);
    const bool has_a2x = (A2X != nullptr);
    const real c = f.c;

    real d_prev = 0;
#ifdef ADI_RECOMPUTE_FACTORS
    real* __restrict__ scratch = f.scratch + row;
    real cstar_prev = 0;
    pipelined_sweep<kPrefetch>(
        cols,
        [&](int j) {
            return ForwardIn{dirichlet ? real(0) : ld(op.a2_lower + j), dirichlet ? real(0) : ld(op.a2_diag + j),
                             dirichlet ? real(0) : ld(op.a2_upper + j), ld(Y + row + j),
                             has_a2x ? ld(A2X + row + j) : real(0)};
        },
        [&](int j, const ForwardIn& in) {
            const real a = -c * in.lower;
            const real r = has_a2x ? in.r0 - c * in.r1 : in.r0;
            const real b = real(1) - c * in.diag;
            const real u = -c * in.upper;
            const real den = (j == 0) ? b : b - a * cstar_prev;
            cstar_prev = u / den;
            scratch[j] = cstar_prev;
            const real d = (j == 0) ? r / den : (r - a * d_prev) / den;
            X[row + j] = d;
            d_prev = d;
        });
#else
    const real* cstar = f.v_cstar + (dirichlet ? 0 : cols);
    const real* pivot = f.v_pivot + (dirichlet ? 0 : cols);
    pipelined_sweep<kPrefetch>(
        cols,
        [&](int j) {
            return ForwardIn{dirichlet ? real(0) : ld(op.a2_lower + j), ld(Y + row + j),
                             has_a2x ? ld(A2X + row + j) : real(0), ld(pivot + j)};
        },
        [&](int j, const ForwardIn& in) {
            const real a = -c * in.lower;
            const real r = has_a2x ? in.r0 - c * in.r1 : in.r0;
            const real d = (j == 0) ? apply_pivot(r, in.pivot) : apply_pivot(r - a * d_prev, in.pivot);
            X[row + j] = d;
            d_prev = d;
        });
#endif

    // Back sweep over j = cols-2, ..., 0 (element t is column cols-2-t).
    real x_next = d_prev;
    const int last = cols - 2;
    pipelined_sweep<kPrefetch>(
        cols - 1,
        [&](int t) {
            const int j = last - t;
#ifdef ADI_RECOMPUTE_FACTORS
            return BackIn{X[row + j], scratch[j]};
#else
            return BackIn{X[row + j], ld(cstar + j)};
#endif
        },
        [&](int t, const BackIn& in) {
            x_next = in.d - in.cstar * x_next;
            X[row + last - t] = x_next;
        });
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

// The solve kernels take their arrays as __restrict__, so the host checks they are distinct.
void Tridiag::solve_s(int set, const real* R, real* X, cudaStream_t stream) {
    if (R == X) throw std::runtime_error("Tridiag::solve_s: R and X must be different arrays");
    solve_s_kernel<<<line_blocks(op_.cols), kLineBlock, 0, stream>>>(op_, view(set), R, X);
    CUDA_CHECK_LAST();
}

void Tridiag::solve_v(int set, const real* Y, const real* A2X, real* X, cudaStream_t stream) {
    if (Y == X || A2X == X || (A2X && A2X == Y)) {
        throw std::runtime_error("Tridiag::solve_v: Y, A2X and X must be different arrays");
    }
    solve_v_kernel<<<line_blocks(op_.rows), kLineBlock, 0, stream>>>(op_, view(set), Y, A2X, X);
    CUDA_CHECK_LAST();
}

}  // namespace adi
