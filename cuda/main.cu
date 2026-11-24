// -----------------------------------------------------------------------------
// adi_cuda: the CUDA port of the ADI time loop.
//
//   adi_cuda CASE_DIR OUT_DIR
//       Check mode: reads a fixture case and writes its outputs for
//       `python -m adi_heston.fixtures check CASE_DIR OUT_DIR` and
//       scripts/check_cuda.py. So far: apply_A0/A1/A2 on kernel_input, and
//       k1_R1 / k1_A2U (the explicit stage K1 with the one-step arguments).
//   adi_cuda INPUTS_DIR --bench
//       Benchmark mode (not implemented yet).
//
// Project: ADI finite-difference methods for American options under Heston
//          Math/CS 714, UW–Madison, Fall 2026
// Author:  Ganesh Arivoli <arivoli@wisc.edu>
// -----------------------------------------------------------------------------
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
}

[[noreturn]] void usage() {
    std::fprintf(stderr,
                 "usage: adi_cuda CASE_DIR OUT_DIR\n"
                 "       adi_cuda INPUTS_DIR --bench   (not implemented yet)\n");
    std::exit(2);
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> paths;
    bool bench = false;
    for (int a = 1; a < argc; ++a) {
        const std::string arg = argv[a];
        if (arg == "--bench") {
            bench = true;
        } else if (!arg.empty() && arg[0] != '-') {
            paths.push_back(arg);
        } else {
            usage();
        }
    }
    if (bench || paths.size() != 2) usage();

    try {
        print_device();
        check_mode(paths[0], paths[1]);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "adi_cuda: %s\n", e.what());
        return 1;
    }
    return 0;
}
