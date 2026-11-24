// -----------------------------------------------------------------------------
// adi_cuda: the CUDA port of the ADI time loop.
//
//   adi_cuda CASE_DIR OUT_DIR
//       Check mode: reads a fixture case and writes its outputs for
//       `python -m adi_heston.fixtures check CASE_DIR OUT_DIR`.
//       So far (stage 2) it only round-trips U0 and A2 through device memory.
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
#include <string>
#include <vector>

#include "host/io.hpp"
#include "kernels/common.cuh"

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

// Stage 2: send U0 and the compressed A2 through device memory and back, to test loading and buffers.
void check_mode(const fs::path& case_dir, const fs::path& out_dir) {
    const adi::HostProblem problem = adi::load_case(case_dir);
    fs::create_directories(out_dir);
    std::vector<double> back;

    const adi::DeviceBuffer<adi::real> U(problem.U0);
    U.download(back);
    adi::write_bin(out_dir / "roundtrip_U0.bin", back);

    const adi::DeviceBuffer<adi::real> a2(problem.a2);
    a2.download(back);
    adi::write_bin(out_dir / "roundtrip_a2.bin", back);
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
