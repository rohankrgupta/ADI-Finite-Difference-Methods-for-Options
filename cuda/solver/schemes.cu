// -----------------------------------------------------------------------------
// The ADI schemes: which kernels each one launches for a schedule row, and in
// what order (Haentjens & in 't Hout 2015, eqs. 4.1-4.4; Python: solver.py).
//
// Every scheme starts with the Douglas pass:
//   K1 explicit stage  U -> R1 = Y0 - c*A1U, A2U    (Y0 includes dtau*lam if American)
//   solve_s            (I - c*A1) Y1 = R1
//   solve_v            (I - c*A2) Y2 = Y1 - c*A2U
// Douglas writes Y2 straight into U (solve_v never reads U). The corrected
// schemes keep Y2 and add a second pass:
//   K4 correction      Y2 -> R1' = Y0~ - c*A1X, row 0 = boundary (one fused kernel)
//   solve_s            (I - c*A1) Y1 = R1'
//   solve_v            (I - c*A2) U  = Y1 - c*A2X
// with, per scheme,
//   Craig-Sneyd:  Y0~ = Y0 + dtau/2 (A0 Y2 - A0U),  A1X = A1U, A2X = A2U.
// Every arrow is a global dependency, so the kernels run one after another on
// one stream; the parallelism is inside each kernel (one thread per node in K1
// and K4, one per grid line in the solves). The American update that follows
// every row is launched by AdiSolver.
//
// Thread mapping of K4: one thread per node, 32x8 blocks, x along j. It reads
// Y2 through the read-only cache, so Y2 must not be written while it runs.
//
// Project: ADI finite-difference methods for American options under Heston
//          Math/CS 714, UW–Madison, Fall 2026
// Author:  Ganesh Arivoli <arivoli@wisc.edu>
// -----------------------------------------------------------------------------
#include "solver/solver.cuh"

namespace adi {

namespace {

// Arguments of K4, the correction kernel of the corrected schemes.
struct CorrectionArgs {
    const real* Y2;   // result of the first pass (the stencils read its neighbours)
    const real* Y0;   // the predictor
    const real* A0U;  // Craig-Sneyd
    const real* A1U;  // Craig-Sneyd: the second pass subtracts c*A1U
    real* R1;         // out: Y0~ - c*A1X, row 0 = boundary
    real c;           // theta_adi * dtau
    real half_dtau;   // 0.5 * dtau, computed on the host as Python computes it
    real boundary;
};

// K4: the correction of the predictor and the right-hand side of the second s-solve, in one pass.
// Each branch follows the operation order of the scheme's Python step function.
template <Scheme S>
__global__ void correction_kernel(OperatorView op, CorrectionArgs a) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    const int i = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= op.rows || j >= op.cols) return;
    const int k = i * op.cols + j;

    const real a0y2 = apply_A0_at(op, a.Y2, i, j);
    real y0_tilde = 0;
    real a1x = 0;
    if constexpr (S == Scheme::craig_sneyd) {
        // craig_sneyd_step: Y0 + 0.5*dtau*(A0 Y2 - A0U); second pass with A1U
        y0_tilde = ld(a.Y0 + k) + a.half_dtau * (a0y2 - ld(a.A0U + k));
        a1x = ld(a.A1U + k);
    }
    a.R1[k] = (i == 0) ? a.boundary : y0_tilde - a.c * a1x;
}

template <Scheme S>
void launch_correction(const OperatorView& op, const CorrectionArgs& args, cudaStream_t stream) {
    correction_kernel<S><<<node_grid(op), node_block(), 0, stream>>>(op, args);
    CUDA_CHECK_LAST();
}

// The Douglas pass every scheme starts with. K1 also stores the arrays the scheme's second pass reads
// (null pointers: not stored). The v-solve writes into `out`: U for Douglas, Y2 for the corrected schemes.
void douglas_pass(const StepContext& ctx, const StepPlan& p, real* A1U, real* Y0, real* A0U, real* out) {
    Workspace& w = *ctx.ws;
    ExplicitStageArgs args{};
    args.U = w.U.data();
    args.lam = w.lam.size() > 0 ? w.lam.data() : nullptr;  // allocated only for an American option
    args.R1 = w.R.data();
    args.A2U = w.A2U.data();
    args.A1U = A1U;
    args.Y0 = Y0;
    args.A0U = A0U;
    args.dtau = p.dtau;
    args.c = p.c;
    args.boundary = p.boundary;
    launch_explicit_stage(ctx.op, args, ctx.stream);
    ctx.tridiag->solve_s(p.factor_set, w.R.data(), w.Y1.data(), ctx.stream);
    ctx.tridiag->solve_v(p.factor_set, w.Y1.data(), w.A2U.data(), out, ctx.stream);
}

}  // namespace

void enqueue_douglas_step(const StepContext& ctx, const StepPlan& p) {
    douglas_pass(ctx, p, nullptr, nullptr, nullptr, ctx.ws->U.data());
}

void enqueue_craig_sneyd_step(const StepContext& ctx, const StepPlan& p) {
    Workspace& w = *ctx.ws;
    douglas_pass(ctx, p, w.A1U.data(), w.Y0.data(), w.A0U.data(), w.Y2.data());

    CorrectionArgs args{};
    args.Y2 = w.Y2.data();
    args.Y0 = w.Y0.data();
    args.A0U = w.A0U.data();
    args.A1U = w.A1U.data();
    args.R1 = w.R.data();
    args.c = p.c;
    args.half_dtau = 0.5 * p.dtau;
    args.boundary = p.boundary;
    launch_correction<Scheme::craig_sneyd>(ctx.op, args, ctx.stream);

    ctx.tridiag->solve_s(p.factor_set, w.R.data(), w.Y1.data(), ctx.stream);
    ctx.tridiag->solve_v(p.factor_set, w.Y1.data(), w.A2U.data(), w.U.data(), ctx.stream);
}

}  // namespace adi
