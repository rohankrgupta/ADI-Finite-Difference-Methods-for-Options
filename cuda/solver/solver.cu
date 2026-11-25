// -----------------------------------------------------------------------------
// AdiSolver: one-time setup and the time loop.
//
// Setup uploads the operators, allocates every work array once and factorizes
// the tridiagonal systems once per distinct c = theta*dtau of the schedule. The
// time loop then only enqueues kernels: no allocation, no copies, no waiting.
// An NVTX range "time_loop" marks it for Nsight Systems.
//
// Project: ADI finite-difference methods for American options under Heston
//          Math/CS 714, UW–Madison, Fall 2026
// Author:  Ganesh Arivoli <arivoli@wisc.edu>
// -----------------------------------------------------------------------------
#include "solver/solver.cuh"

#include <nvtx3/nvToolsExt.h>

#include <chrono>
#include <stdexcept>
#include <string>

namespace adi {

const char* scheme_name(Scheme scheme) {
    switch (scheme) {
        case Scheme::douglas: return "douglas";
        case Scheme::craig_sneyd: return "craig_sneyd";
        case Scheme::modified_craig_sneyd: return "modified_craig_sneyd";
        case Scheme::hundsdorfer_verwer: return "hundsdorfer_verwer";
    }
    return "unknown";
}

AdiSolver::AdiSolver(const HostProblem& problem, cudaStream_t stream)
    : created_(std::chrono::steady_clock::now()),
      stream_(stream),
      american_(problem.american),
      ops_(problem),
      tridiag_(ops_.view()) {
    const Scheme scheme = static_cast<Scheme>(problem.scheme);
    if (scheme != Scheme::douglas) {
        throw std::runtime_error(std::string("scheme ") + scheme_name(scheme) + " is not implemented yet");
    }
    if (american_) throw std::runtime_error("American options are not implemented yet");

    const std::size_t n = problem.size();
    ws_.U = DeviceBuffer<real>(n);
    ws_.R = DeviceBuffer<real>(n);
    ws_.Y1 = DeviceBuffer<real>(n);
    ws_.A2U = DeviceBuffer<real>(n);
    U0_ = DeviceBuffer<real>(problem.U0);

    for (std::size_t row = 0; row < problem.schedule.size(); ++row) {
        plans_.push_back(plan(problem.schedule[row], row < problem.startup_half_steps ? Scheme::douglas : scheme));
    }
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    setup_ms_ = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - created_).count();
}

StepPlan AdiSolver::plan(const Step& step, Scheme scheme) {
    const real c = step.theta * step.dtau;  // as Python's step functions compute it
    return {step.dtau, c, step.boundary, tridiag_.prepare(c, stream_), scheme};
}

void AdiSolver::reset(const std::vector<double>& U, const std::vector<double>* lam) {
    ws_.U.upload(U);
    if (lam) ws_.lam.upload(*lam);
}

void AdiSolver::reset_to_initial() {
    CUDA_CHECK(cudaMemcpyAsync(ws_.U.data(), U0_.data(), U0_.size() * sizeof(real), cudaMemcpyDeviceToDevice,
                               stream_));
    if (american_) CUDA_CHECK(cudaMemsetAsync(ws_.lam.data(), 0, ws_.lam.size() * sizeof(real), stream_));
}

void AdiSolver::enqueue(const StepPlan& p) {
    const StepContext ctx{ops_.view(), &tridiag_, &ws_, stream_};
    switch (p.scheme) {
        case Scheme::douglas:
            enqueue_douglas_step(ctx, p);
            break;
        default:
            throw std::runtime_error(std::string("scheme ") + scheme_name(p.scheme) + " is not implemented yet");
    }
}

void AdiSolver::enqueue_schedule() {
    nvtxRangePushA("time_loop");
    for (const StepPlan& p : plans_) enqueue(p);
    nvtxRangePop();
}

void AdiSolver::enqueue_step(const Step& step, Scheme scheme) { enqueue(plan(step, scheme)); }

void AdiSolver::download(std::vector<double>& U, std::vector<double>* lam) {
    CUDA_CHECK(cudaStreamSynchronize(stream_));
    ws_.U.download(U);
    if (lam) ws_.lam.download(*lam);
}

}  // namespace adi
