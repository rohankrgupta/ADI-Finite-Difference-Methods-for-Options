// -----------------------------------------------------------------------------
// The ADI driver: AdiSolver owns everything a run needs on the GPU (operators,
// factor sets, work arrays), plans every schedule row once, and enqueues the
// kernels of each row on one stream without waiting in between.
//
// Project: ADI finite-difference methods for American options under Heston
//          Math/CS 714, UW–Madison, Fall 2026
// Author:  Ganesh Arivoli <arivoli@wisc.edu>
// -----------------------------------------------------------------------------
#pragma once

#include <cuda_runtime.h>

#include <chrono>
#include <cstddef>
#include <string>
#include <vector>

#include "host/io.hpp"
#include "kernels/common.cuh"
#include "kernels/kernels.cuh"

namespace adi {

// The four ADI schemes, numbered as in python/adi_heston/fixtures.py (SCHEME_IDS).
enum class Scheme { douglas = 0, craig_sneyd = 1, modified_craig_sneyd = 2, hundsdorfer_verwer = 3 };

const char* scheme_name(Scheme scheme);

// The device arrays of one run, allocated once and reused by every row. Only the arrays the
// scheme needs are allocated (worklog decisions, Section 6.3).
struct Workspace {
    DeviceBuffer<real> U;    // the solution, updated in place by every row
    DeviceBuffer<real> lam;  // Ikonen-Toivanen multiplier (American only)
    DeviceBuffer<real> R;    // right-hand side of the s-solve (both passes of a corrected scheme)
    DeviceBuffer<real> Y1;   // result of the s-solve (both passes)
    DeviceBuffer<real> A2U;  // A2 applied to U, needed by the v-solve
    // The corrected schemes only:
    DeviceBuffer<real> Y0;   // the predictor U + dtau*A U (+ dtau*lam)
    DeviceBuffer<real> A0U;  // Craig-Sneyd, Modified Craig-Sneyd
    DeviceBuffer<real> A1U;  // Craig-Sneyd, Modified Craig-Sneyd: the second pass subtracts c*A1U
    DeviceBuffer<real> AU;   // Modified Craig-Sneyd, Hundsdorfer-Verwer: A U, for the correction A Y2 - A U
    DeviceBuffer<real> Y2;   // result of the first pass
    DeviceBuffer<real> A2Y2; // Hundsdorfer-Verwer: A2 Y2, for the second v-solve
};

// One schedule row, ready to enqueue.
struct StepPlan {
    real dtau;
    real theta;     // theta_adi
    real c;         // theta_adi * dtau
    real boundary;  // Dirichlet value on the s = 0 row at the end of the row
    int factor_set; // Tridiag factor set for c
    Scheme scheme;  // the start-up rows always use Douglas
};

// What a scheme's step function needs to enqueue its kernels.
struct StepContext {
    OperatorView op;
    Tridiag* tridiag;
    Workspace* ws;
    cudaStream_t stream;
};

// schemes.cu: one row of each scheme (the kernel order is described there).
void enqueue_douglas_step(const StepContext& ctx, const StepPlan& plan);
void enqueue_craig_sneyd_step(const StepContext& ctx, const StepPlan& plan);
void enqueue_modified_craig_sneyd_step(const StepContext& ctx, const StepPlan& plan);
void enqueue_hundsdorfer_verwer_step(const StepContext& ctx, const StepPlan& plan);

class AdiSolver {
public:
    // Whether the time loop of this scheme is implemented.
    static bool supports(Scheme scheme);

    // Uploads the operators, allocates the work arrays and factorizes once per distinct c of the schedule.
    AdiSolver(const HostProblem& problem, cudaStream_t stream);

    // Start from a host array, or from the problem's U0 kept on the device. For an American option the
    // multiplier starts from *lam, or from zero when lam is null (and always in reset_to_initial).
    void reset(const std::vector<double>& U, const std::vector<double>* lam);
    void reset_to_initial();

    // Enqueue every row of the schedule, or one extra row (factorizing for its c if needed).
    void enqueue_schedule();
    void enqueue_step(const Step& step, Scheme scheme);

    // Wait for the stream, then copy the solution (and multiplier) back.
    void download(std::vector<double>& U, std::vector<double>* lam);

    double setup_ms() const { return setup_ms_; }
    std::size_t factor_sets() const { return tridiag_.factor_sets(); }
    std::string s_method() const { return tridiag_.s_method(plans_.back().factor_set); }  // for the log
    std::string v_method() const { return tridiag_.v_method(plans_.back().factor_set); }

private:
    StepPlan plan(const Step& step, Scheme scheme);
    void enqueue(const StepPlan& plan);

    std::chrono::steady_clock::time_point created_;  // first member: setup time includes the uploads
    cudaStream_t stream_;
    bool american_;
    DeviceOperators ops_;
    Tridiag tridiag_;
    Workspace ws_;
    DeviceBuffer<real> U0_;      // the problem's initial solution, for reset_to_initial
    DeviceBuffer<real> payoff_;  // payoff per s-node (rows values), American only
    std::vector<StepPlan> plans_;
    double setup_ms_ = 0;
};

}  // namespace adi
