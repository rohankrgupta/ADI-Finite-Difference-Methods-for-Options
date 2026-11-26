// -----------------------------------------------------------------------------
// The operator split A = A0 + A1 + A2 on the GPU: device copies of the
// coefficients, per-node stencils, and the host functions that launch the
// building-block kernels.
//
// Layout: row-major, node (i, j) at i*cols + j, i along s, j along v.
// A1 is stored as three 2D planes. A2 does not depend on s, so it is stored as
// three rows of `cols` values; row 0 (s = 0, Dirichlet) has zero coefficients.
// The stencils follow python/adi_heston/operators.py and fd.py operation by
// operation, so with ADI_EXACT (no FMA) they reproduce Python bit for bit,
// signed zeros included.
//
// Project: ADI finite-difference methods for American options under Heston
//          Math/CS 714, UW–Madison, Fall 2026
// Author:  Ganesh Arivoli <arivoli@wisc.edu>
// -----------------------------------------------------------------------------
#pragma once

#include <cuda_runtime.h>

#include <climits>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

#include "host/io.hpp"
#include "kernels/common.cuh"

namespace adi {

// Read-only views of the operator coefficients, passed to kernels by value.
struct OperatorView {
    const real* a1_lower;  // rows*cols each: A1 couples node (i, j) to (i-1, j) and (i+1, j)
    const real* a1_diag;
    const real* a1_upper;
    const real* a2_lower;  // cols each: A2 couples (i, j) to (i, j-1) and (i, j+1), the same for every i >= 1
    const real* a2_diag;
    const real* a2_upper;
    const real* a0_coef;  // rows*cols: rho*xi*s*v, zero on all edges
    const real* w_s;      // rows*3: first-derivative weights in s on (i-1, i, i+1)
    const real* w_v;      // cols*3: first-derivative weights in v on (j-1, j, j+1)
    int rows;             // m + 1
    int cols;             // n + 1
};

// Owns the device copies of the operator coefficients of one problem.
class DeviceOperators {
public:
    explicit DeviceOperators(const HostProblem& p)
        : a1_(p.a1), a2_(p.a2), a0_coef_(p.a0_coef), w_s_(p.w_s), w_v_(p.w_v),
          rows_(static_cast<int>(p.rows)), cols_(static_cast<int>(p.cols)) {
        if (p.rows * p.cols >= static_cast<std::size_t>(INT_MAX)) {
            throw std::runtime_error("grid too large for 32-bit node indices");
        }
    }

    OperatorView view() const {
        const std::size_t n = static_cast<std::size_t>(rows_) * cols_;
        const std::size_t c = cols_;
        return {a1_.data(), a1_.data() + n, a1_.data() + 2 * n,
                a2_.data(), a2_.data() + c, a2_.data() + 2 * c,
                a0_coef_.data(), w_s_.data(), w_v_.data(), rows_, cols_};
    }
    int rows() const { return rows_; }
    int cols() const { return cols_; }

private:
    DeviceBuffer<real> a1_, a2_, a0_coef_, w_s_, w_v_;
    int rows_, cols_;
};

// Launch shape of every one-thread-per-node kernel: 32x8 blocks with x along j (contiguous in memory),
// so a warp reads 32 neighbouring values of each array.
constexpr int kNodeBlockX = 32;
constexpr int kNodeBlockY = 8;
inline dim3 node_block() { return dim3(kNodeBlockX, kNodeBlockY); }
inline dim3 node_grid(const OperatorView& op) {
    return dim3((op.cols + kNodeBlockX - 1) / kNodeBlockX, (op.rows + kNodeBlockY - 1) / kNodeBlockY);
}

// ---- per-node stencils ------------------------------------------------------------------------

// Load through the read-only data cache. Only for arrays the running kernel does not write.
__device__ __forceinline__ real ld(const real* p) { return __ldg(p); }

// (A1 U)(i, j) = (diag*U(i,j) + lower*U(i-1,j)) + upper*U(i+1,j), the order of operators.apply_A1.
__device__ __forceinline__ real apply_A1_at(const OperatorView& op, const real* U, int i, int j) {
    const int k = i * op.cols + j;
    real value = ld(op.a1_diag + k) * ld(U + k);
    if (i > 0) value += ld(op.a1_lower + k) * ld(U + k - op.cols);
    if (i + 1 < op.rows) value += ld(op.a1_upper + k) * ld(U + k + op.cols);
    return value;
}

// (A2 U)(i, j), the order of operators.apply_A2. Row 0 still multiplies its zero coefficients,
// as the reference does, so a negative U gives -0 there exactly as in Python.
__device__ __forceinline__ real apply_A2_at(const OperatorView& op, const real* U, int i, int j) {
    const int k = i * op.cols + j;
    const bool dirichlet = (i == 0);
    const real lower = dirichlet ? real(0) : ld(op.a2_lower + j);
    const real diag = dirichlet ? real(0) : ld(op.a2_diag + j);
    const real upper = dirichlet ? real(0) : ld(op.a2_upper + j);
    real value = diag * ld(U + k);
    if (j > 0) value += lower * ld(U + k - 1);
    if (j + 1 < op.cols) value += upper * ld(U + k + 1);
    return value;
}

// dU/dv at (i, j): zero in the first and last column, else the three-point formula of fd.apply_three_point.
__device__ __forceinline__ real dv_at(const OperatorView& op, const real* U, int i, int j) {
    if (j == 0 || j + 1 == op.cols) return real(0);
    const int k = i * op.cols + j;
    const real* w = op.w_v + 3 * j;
    return (ld(w) * ld(U + k - 1) + ld(w + 1) * ld(U + k)) + ld(w + 2) * ld(U + k + 1);
}

// (A0 U)(i, j) = a0_coef * (s-derivative of the v-derivative), as fd.mixed_derivative; zero on rows 0 and m.
__device__ __forceinline__ real apply_A0_at(const OperatorView& op, const real* U, int i, int j) {
    if (i == 0 || i + 1 == op.rows) return real(0);
    const real* w = op.w_s + 3 * i;
    const real mixed = (ld(w) * dv_at(op, U, i - 1, j) + ld(w + 1) * dv_at(op, U, i, j)) +
                       ld(w + 2) * dv_at(op, U, i + 1, j);
    return ld(op.a0_coef + i * op.cols + j) * mixed;
}

// ---- kernels (launched through these host functions) ------------------------------------------

// Arguments of K1, the explicit stage of every scheme. Optional outputs are written only when non-null.
struct ExplicitStageArgs {
    const real* U;    // solution at the old time
    const real* lam;  // Ikonen-Toivanen multiplier, or nullptr for a European option
    real* R1;         // Y0 - c*A1U, row 0 = boundary: right-hand side of the s-solve
    real* A2U;        // needed by the v-solve
    real* A1U;        // optional: Craig-Sneyd and Modified Craig-Sneyd
    real* Y0;         // optional: the corrected schemes
    real* A0U;        // optional: Craig-Sneyd and Modified Craig-Sneyd
    real* AU;         // optional: Modified Craig-Sneyd and Hundsdorfer-Verwer
    real dtau;
    real c;           // theta_adi * dtau
    real boundary;    // Dirichlet value on the s = 0 row at the new time
};

// K1: A0U, A1U, A2U; Y0 = U + dtau*AU (+ dtau*lam); R1 = Y0 - c*A1U. One thread per node.
void launch_explicit_stage(const OperatorView& op, const ExplicitStageArgs& args, cudaStream_t stream);

// The Ikonen-Toivanen update after a row's last v-solve (Python: solver.ikonen_toivanen_update), in place:
//     U = max(U_bar - dtau*lam, g),   lam = max(0, lam + (g - U_bar)/dtau)
// where g = payoff[i] (rows values). Both use the unprojected U_bar and the old lam. One thread per node.
void launch_ikonen_toivanen(const OperatorView& op, real* U, real* lam, const real* payoff, real dtau,
                            cudaStream_t stream);

// A0 X, A1 X and A2 X on the whole grid, for the single-kernel fixture checks.
void launch_apply_probe(const OperatorView& op, const real* X, real* A0X, real* A1X, real* A2X,
                        cudaStream_t stream);

// ---- tridiagonal solves (tridiag.cu) ------------------------------------------------------------

// The factors of (I - c*A1) and (I - c*A2) for one value of c, as the solve kernels see them.
// pivot holds 1/den (default; the solve multiplies) or den (ADI_EXACT; the solve divides, like Python).
// With ADI_RECOMPUTE_FACTORS the factor pointers are null and the solves keep c* in `scratch`.
struct FactorView {
    const real* s_cstar;  // rows*cols
    const real* s_pivot;  // rows*cols
    const real* v_cstar;  // 2*cols: the sequence of the Dirichlet row 0, then the one shared by rows >= 1
    const real* v_pivot;  // 2*cols
    real* scratch;        // rows*cols, ADI_RECOMPUTE_FACTORS only
    real c;
};

// The default build solves along s and v with the partitioned method (partition.cu); the exact and recompute
// builds, and ADI_THOMAS_SOLVES, keep the Thomas solves.
#if !defined(ADI_EXACT) && !defined(ADI_RECOMPUTE_FACTORS) && !defined(ADI_THOMAS_SOLVES)
#define ADI_PARTITIONED 1
#endif

#ifdef ADI_PARTITIONED
// Precomputed for the partitioned solve along v (partition.cu), per factor set. A row is cut into C = 32*parts
// chunks of L values; seq 0 is the Dirichlet row, seq 1 every row >= 1. Element t of chunk q is stored at
// (seq*L + t)*C + q, so the threads' loads are coalesced.
struct VChunks {
    const real* a;            // -c * A2 lower diagonal (zero at j = 0)
    const real* pivot;
    const real* fwd_gain;     // d_j per unit of d before the chunk
    const real* cstar;        // zero at j = cols-1
    const real* back_gain;    // x_j per unit of x after the chunk
    const real* fwd_window;   // [(seq*5 + s)*C + q]: product of chunk slopes over chunks q-2^s+1 .. q
    const real* back_window;  // ... over chunks q .. q+2^s-1
    const real* fwd_prefix;   // [seq*C + q]: product of chunk slopes from the warp's first chunk to q
    const real* back_suffix;  // ... from q to the warp's last chunk
    int L;
    int parts;  // warps per row
};
struct VChunkBuffers {
    DeviceBuffer<real> a, pivot, fwd_gain, cstar, back_gain, fwd_window, back_window, fwd_prefix, back_suffix;
    int L = 0;  // 0: a row does not fit in shared memory; solve along v with Thomas
    int parts = 0;
    VChunks view() const {
        return {a.data(),           pivot.data(),       fwd_gain.data(),   cstar.data(),
                back_gain.data(),   fwd_window.data(),  back_window.data(), fwd_prefix.data(),
                back_suffix.data(), L,                  parts};
    }
};
VChunkBuffers build_v_chunks(const OperatorView& op, real c, const real* v_cstar, const real* v_pivot,
                             cudaStream_t stream);
void launch_solve_v_partition(const OperatorView& op, const VChunks& ch, real c, const real* Y, const real* A2X,
                              real* X, cudaStream_t stream);

// Precomputed for the partitioned solve along s, per factor set: columns are cut into chunks of L rows; a block
// holds `group` neighbouring columns and all their chunks in `warps` warps. 2D arrays are rows*cols, row-major.
struct SChunks {
    const real* a;          // -c * A1 lower diagonal
    const real* pivot;      // the Thomas factors
    const real* cstar;
    const real* fwd_gain;   // d_i per unit of d before the chunk
    const real* back_gain;  // x_i per unit of x after the chunk
    int L;
    int group;
    int warps;
};
struct SChunkBuffers {
    DeviceBuffer<real> a, fwd_gain, back_gain;
    int L = 0;  // 0: no layout fits (columns too tall); solve along s with Thomas
    int group = 0;
    int warps = 0;
    SChunks view(const real* pivot, const real* cstar) const {
        return {a.data(), pivot, cstar, fwd_gain.data(), back_gain.data(), L, group, warps};
    }
};
SChunkBuffers build_s_chunks(const OperatorView& op, real c, const real* s_cstar, const real* s_pivot,
                             cudaStream_t stream);
void launch_solve_s_partition(const OperatorView& op, const SChunks& ch, const real* R, real* X,
                              cudaStream_t stream);
#endif

// Thomas solves of (I - c*A1) and (I - c*A2), one thread per grid line, with the factors computed once
// per distinct value of c. Keeps a copy of the operator view: the DeviceOperators must outlive it.
class Tridiag {
public:
    explicit Tridiag(const OperatorView& op);

    // Index of the factor set for c (compared exactly), factorizing it on first use.
    int prepare(real c, cudaStream_t stream);
    std::size_t factor_sets() const { return sets_.size(); }

    // (I - c*A1) X = R along s, one system per column. R and X must be different arrays; row 0 of R
    // (the Dirichlet row) passes through unchanged.
    void solve_s(int set, const real* R, real* X, cudaStream_t stream);

    // (I - c*A2) X = Y - c*A2X along v, one system per row; with A2X null the right-hand side is Y.
    // Y, A2X and X must be different arrays.
    void solve_v(int set, const real* Y, const real* A2X, real* X, cudaStream_t stream);

    // How the solves run for a factor set, for the log: "Thomas" or the partition layout.
    std::string s_method(int set) const;
    std::string v_method(int set) const;

private:
    struct Set {
        real c = 0;
        DeviceBuffer<real> s_cstar, s_pivot, v_cstar, v_pivot;
#ifdef ADI_PARTITIONED
        SChunkBuffers s_chunks;
        VChunkBuffers v_chunks;
#endif
    };
    FactorView view(int set);

    OperatorView op_;
    std::vector<Set> sets_;
    DeviceBuffer<real> scratch_;  // ADI_RECOMPUTE_FACTORS only
};

}  // namespace adi
