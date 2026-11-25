// -----------------------------------------------------------------------------
// The ADI schemes: which kernels each one launches for a schedule row, and in
// what order (Haentjens & in 't Hout 2015, eqs. 4.1-4.4; Python: solver.py).
//
// Douglas:  K1 explicit stage  U -> R1 = Y0 - c*A1U, A2U
//           solve_s            (I - c*A1) Y1 = R1
//           solve_v            (I - c*A2) U  = Y1 - c*A2U   (U overwritten in place;
//                                                           solve_v never reads U)
//
// Project: ADI finite-difference methods for American options under Heston
//          Math/CS 714, UW–Madison, Fall 2026
// Author:  Ganesh Arivoli <arivoli@wisc.edu>
// -----------------------------------------------------------------------------
#include "solver/solver.cuh"

namespace adi {

void enqueue_douglas_step(const StepContext& ctx, const StepPlan& p) {
    Workspace& w = *ctx.ws;
    ExplicitStageArgs args{};
    args.U = w.U.data();
    args.lam = nullptr;
    args.R1 = w.R.data();
    args.A2U = w.A2U.data();
    args.dtau = p.dtau;
    args.c = p.c;
    args.boundary = p.boundary;
    launch_explicit_stage(ctx.op, args, ctx.stream);
    ctx.tridiag->solve_s(p.factor_set, w.R.data(), w.Y1.data(), ctx.stream);
    ctx.tridiag->solve_v(p.factor_set, w.Y1.data(), w.A2U.data(), w.U.data(), ctx.stream);
}

}  // namespace adi
