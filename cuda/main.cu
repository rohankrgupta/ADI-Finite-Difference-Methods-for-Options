// -----------------------------------------------------------------------------
// adi_cuda: the CUDA port of the ADI time loop.
//
//   adi_cuda CASE_DIR OUT_DIR
//       Check mode: reads a fixture case and writes its outputs for
//       `python -m adi_heston.fixtures check CASE_DIR OUT_DIR` and
//       scripts/check_cuda.py: apply_A0/A1/A2 and solve_A1/A2 on kernel_input,
//       k1_R1 / k1_A2U (the explicit stage with the one-step arguments),
//       tridiag_sets (factor sets the schedule needs), and the time loop:
//       douglas_step and U_final for the base cases; step, U_final (plus
//       step_lam and lam_final if American) for the variant cases.
//   adi_cuda INPUTS_DIR --bench [--repeats R]
//       Benchmark mode: one warm-up and R timed runs (default 5) of the whole
//       schedule, timed with CUDA events on the work stream. Prints
//       gpu,rows,cols,steps,median_ms_per_step,min_ms_per_step,setup_ms,scheme,american
//
// Project: ADI finite-difference methods for American options under Heston
//          Math/CS 714, UW–Madison, Fall 2026
// Author:  Ganesh Arivoli <arivoli@wisc.edu>
// -----------------------------------------------------------------------------
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "host/io.hpp"
#include "kernels/common.cuh"
#include "kernels/kernels.cuh"
#include "solver/solver.cuh"

namespace fs = std::filesystem;

namespace {

// Name the GPU on stderr, so every log records the hardware.
void print_device() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));
    int runtime = 0;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    std::fprintf(stderr, "adi_cuda: %s, %d SMs, compute %d.%d, CUDA runtime %d.%d\n", prop.name,
                 prop.multiProcessorCount, prop.major, prop.minor, runtime / 1000, (runtime % 1000) / 10);
}

void write_device(const fs::path& path, const adi::DeviceBuffer<adi::real>& buffer) {
    std::vector<double> host;
    buffer.download(host);
    adi::write_bin(path, host);
}

void check_mode(const fs::path& case_dir, const fs::path& out_dir) {
    const adi::HostProblem problem = adi::load_case(case_dir);
    if (problem.kernel_input.empty() || problem.scalars.size() != 4) {
        throw std::runtime_error(case_dir.string() + " is not a fixture case (no kernel_input.bin or scalars.bin)");
    }
    fs::create_directories(out_dir);
    const std::size_t n = problem.size();
    const adi::DeviceOperators ops(problem);
    const adi::OperatorView op = ops.view();
    const adi::DeviceBuffer<adi::real> X(problem.kernel_input);

    // Single kernels on kernel_input.
    adi::DeviceBuffer<adi::real> A0X(n), A1X(n), A2X(n);
    adi::launch_apply_probe(op, X.data(), A0X.data(), A1X.data(), A2X.data(), nullptr);
    write_device(out_dir / "apply_A0.bin", A0X);
    write_device(out_dir / "apply_A1.bin", A1X);
    write_device(out_dir / "apply_A2.bin", A2X);

    // K1 on kernel_input with the one-step arguments [solve_c, dtau, theta_adi, boundary], plus
    // lam_input in an American case.
    const double dtau = problem.scalars[1], theta = problem.scalars[2], boundary = problem.scalars[3];
    adi::DeviceBuffer<adi::real> lam;
    if (!problem.lam_input.empty()) lam = adi::DeviceBuffer<adi::real>(problem.lam_input);
    adi::DeviceBuffer<adi::real> R1(n), A2U(n);
    adi::ExplicitStageArgs args{};
    args.U = X.data();
    args.lam = lam.size() > 0 ? lam.data() : nullptr;
    args.R1 = R1.data();
    args.A2U = A2U.data();
    args.dtau = dtau;
    args.c = theta * dtau;
    args.boundary = boundary;
    adi::launch_explicit_stage(op, args, nullptr);
    write_device(out_dir / "k1_R1.bin", R1);
    write_device(out_dir / "k1_A2U.bin", A2U);

    // Both tridiagonal solves on kernel_input with solve_c.
    adi::Tridiag probe(op);
    const int probe_set = probe.prepare(problem.scalars[0], nullptr);
    adi::DeviceBuffer<adi::real> S1(n), S2(n);
    probe.solve_s(probe_set, X.data(), S1.data(), nullptr);
    probe.solve_v(probe_set, X.data(), nullptr, S2.data(), nullptr);
    write_device(out_dir / "solve_A1.bin", S1);
    write_device(out_dir / "solve_A2.bin", S2);

    // The factor sets the schedule needs: one per distinct c = theta*dtau.
    adi::Tridiag schedule_sets(op);
    for (const adi::Step& step : problem.schedule) schedule_sets.prepare(step.theta * step.dtau, nullptr);
    adi::write_bin(out_dir / "tridiag_sets.bin", {static_cast<double>(schedule_sets.factor_sets())});

    // The time loop, for the schemes implemented so far (European and American).
    const adi::Scheme scheme = static_cast<adi::Scheme>(problem.scheme);
    if (!adi::AdiSolver::supports(scheme)) return;
    if (problem.american && problem.lam_input.empty()) {
        throw std::runtime_error(case_dir.string() + ": an American case needs lam_input.bin");
    }
    const adi::CudaStream stream;
    adi::AdiSolver solver(problem, stream.get());
    std::vector<double> U, lam_out;
    std::vector<double>* lam_back = problem.american ? &lam_out : nullptr;

    // One row from kernel_input (and lam_input), no start-up; an American row ends with the update.
    solver.reset(problem.kernel_input, problem.american ? &problem.lam_input : nullptr);
    solver.enqueue_step({dtau, theta, boundary}, scheme);
    solver.download(U, lam_back);
    if (problem.has_settings) {  // variant case
        adi::write_bin(out_dir / "step.bin", U);
        if (problem.american) adi::write_bin(out_dir / "step_lam.bin", lam_out);
    } else {  // base case
        adi::write_bin(out_dir / "douglas_step.bin", U);
    }

    // The full run from the payoff, with lam = 0 for an American option.
    solver.reset(problem.U0, nullptr);
    solver.enqueue_schedule();
    solver.download(U, lam_back);
    adi::write_bin(out_dir / "U_final.bin", U);
    if (problem.american) adi::write_bin(out_dir / "lam_final.bin", lam_out);
}

void bench_mode(const fs::path& dir, int repeats) {
    const adi::HostProblem problem = adi::load_case(dir);
    const adi::CudaStream stream;
    adi::AdiSolver solver(problem, stream.get());
    std::fprintf(stderr, "adi_cuda: solve along v: %s\n", solver.v_method().c_str());
    adi::EventTimer timer;

    std::vector<double> ms_per_row;
    for (int run = 0; run <= repeats; ++run) {  // run 0 is the warm-up
        solver.reset_to_initial();
        timer.start(stream.get());
        solver.enqueue_schedule();
        timer.stop(stream.get());
        const double ms = timer.elapsed_ms();
        if (run > 0) ms_per_row.push_back(ms / static_cast<double>(problem.schedule.size()));
    }
    std::sort(ms_per_row.begin(), ms_per_row.end());
    std::printf("gpu,%zu,%zu,%zu,%.6f,%.6f,%.3f,%s,%d\n", problem.rows, problem.cols, problem.schedule.size(),
                ms_per_row[ms_per_row.size() / 2], ms_per_row.front(), solver.setup_ms(),
                adi::scheme_name(static_cast<adi::Scheme>(problem.scheme)), problem.american ? 1 : 0);
}

[[noreturn]] void usage() {
    std::fprintf(stderr,
                 "usage: adi_cuda CASE_DIR OUT_DIR\n"
                 "       adi_cuda INPUTS_DIR --bench [--repeats R]\n");
    std::exit(2);
}

int parse_count(const char* text) {
    char* end = nullptr;
    const long value = std::strtol(text, &end, 10);
    if (*end != '\0' || value < 1 || value > 100000) usage();
    return static_cast<int>(value);
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> paths;
    bool bench = false;
    int repeats = 5;
    for (int a = 1; a < argc; ++a) {
        const std::string arg = argv[a];
        if (arg == "--bench") {
            bench = true;
        } else if (arg == "--repeats" && a + 1 < argc) {
            repeats = parse_count(argv[++a]);
        } else if (!arg.empty() && arg[0] != '-') {
            paths.push_back(arg);
        } else {
            usage();
        }
    }
    if (paths.size() != (bench ? 1u : 2u)) usage();

    try {
        print_device();
        if (bench) {
            bench_mode(paths[0], repeats);
        } else {
            check_mode(paths[0], paths[1]);
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "adi_cuda: %s\n", e.what());
        return 1;
    }
    return 0;
}
