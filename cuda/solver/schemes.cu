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
//   Modified CS:  Y0~ = Y0 + c (A0 Y2 - A0U) + (1/2 - theta) dtau (A Y2 - A U),
//                 A1X = A1U, A2X = A2U.
//   Hundsdorfer-Verwer:  Y0~ = Y0 + dtau/2 (A Y2 - A U),  A1X = A1 Y2, A2X = A2 Y2
//                 (K4 stores A2 Y2 for the second v-solve).
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
    const real* A0U;  // Craig-Sneyd, Modified Craig-Sneyd
    const real* A1U;  // Craig-Sneyd, Modified Craig-Sneyd: the second pass subtracts c*A1U
    const real* AU;   // Modified Craig-Sneyd, Hundsdorfer-Verwer: A U = (A0U + A1U) + A2U
    real* R1;         // out: Y0~ - c*A1X, row 0 = boundary
    real* A2Y2;       // out, Hundsdorfer-Verwer: A2 Y2 for the second v-solve
    real c;           // theta_adi * dtau
    real half_dtau;   // 0.5 * dtau, computed on the host as Python computes it
    real mcs_weight;  // (0.5 - theta_adi) * dtau, likewise
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
    } else {
        const real a1y2 = apply_A1_at(op, a.Y2, i, j);
        const real a2y2 = apply_A2_at(op, a.Y2, i, j);
        const real change = ((a0y2 + a1y2) + a2y2) - ld(a.AU + k);  // A Y2 - A U
        if constexpr (S == Scheme::modified_craig_sneyd) {
            // modified_craig_sneyd_step: Y0 + c*(A0 Y2 - A0U) + (0.5 - theta)*dtau*(A Y2 - A U); second pass with A1U
            y0_tilde = (ld(a.Y0 + k) + a.c * (a0y2 - ld(a.A0U + k))) + a.mcs_weight * change;
            a1x = ld(a.A1U + k);
        } else {
            // hundsdorfer_verwer_step: Y0 + 0.5*dtau*(A Y2 - A U); second pass with A1 Y2 and A2 Y2
            y0_tilde = ld(a.Y0 + k) + a.half_dtau * change;
            a1x = a1y2;
            a.A2Y2[k] = a2y2;
        }
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
void douglas_pass(const StepContext& ctx, const StepPlan& p, real* A1U, real* Y0, real* A0U, real* AU,
                  real* out) {
    Workspace& w = *ctx.ws;
    ExplicitStageArgs args{};
    args.U = w.U.data();
    args.lam = w.lam.size() > 0 ? w.lam.data() : nullptr;  // allocated only for an American option
    args.R1 = w.R.data();
    args.A2U = w.A2U.data();
    args.A1U = A1U;
    args.Y0 = Y0;
    args.A0U = A0U;
    args.AU = AU;
    args.dtau = p.dtau;
    args.c = p.c;
    args.boundary = p.boundary;
    launch_explicit_stage(ctx.op, args, ctx.stream);
    ctx.tridiag->solve_s(p.factor_set, w.R.data(), w.Y1.data(), ctx.stream);
    ctx.tridiag->solve_v(p.factor_set, w.Y1.data(), w.A2U.data(), out, ctx.stream);
}

}  // namespace

void enqueue_douglas_step(const StepContext& ctx, const StepPlan& p) {
    douglas_pass(ctx, p, nullptr, nullptr, nullptr, nullptr, ctx.ws->U.data());
}

void enqueue_craig_sneyd_step(const StepContext& ctx, const StepPlan& p) {
    Workspace& w = *ctx.ws;
    douglas_pass(ctx, p, w.A1U.data(), w.Y0.data(), w.A0U.data(), nullptr, w.Y2.data());

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

void enqueue_modified_craig_sneyd_step(const StepContext& ctx, const StepPlan& p) {
    Workspace& w = *ctx.ws;
    douglas_pass(ctx, p, w.A1U.data(), w.Y0.data(), w.A0U.data(), w.AU.data(), w.Y2.data());

    CorrectionArgs args{};
    args.Y2 = w.Y2.data();
    args.Y0 = w.Y0.data();
    args.A0U = w.A0U.data();
    args.A1U = w.A1U.data();
    args.AU = w.AU.data();
    args.R1 = w.R.data();
    args.c = p.c;
    args.mcs_weight = (0.5 - p.theta) * p.dtau;
    args.boundary = p.boundary;
    launch_correction<Scheme::modified_craig_sneyd>(ctx.op, args, ctx.stream);

    ctx.tridiag->solve_s(p.factor_set, w.R.data(), w.Y1.data(), ctx.stream);
    ctx.tridiag->solve_v(p.factor_set, w.Y1.data(), w.A2U.data(), w.U.data(), ctx.stream);
}

void enqueue_hundsdorfer_verwer_step(const StepContext& ctx, const StepPlan& p) {
    Workspace& w = *ctx.ws;
    douglas_pass(ctx, p, nullptr, w.Y0.data(), nullptr, w.AU.data(), w.Y2.data());

    CorrectionArgs args{};
    args.Y2 = w.Y2.data();
    args.Y0 = w.Y0.data();
    args.AU = w.AU.data();
    args.R1 = w.R.data();
    args.A2Y2 = w.A2Y2.data();
    args.c = p.c;
    args.half_dtau = 0.5 * p.dtau;
    args.boundary = p.boundary;
    launch_correction<Scheme::hundsdorfer_verwer>(ctx.op, args, ctx.stream);

    ctx.tridiag->solve_s(p.factor_set, w.R.data(), w.Y1.data(), ctx.stream);
    ctx.tridiag->solve_v(p.factor_set, w.Y1.data(), w.A2Y2.data(), w.U.data(), ctx.stream);
}

}  // namespace adi
