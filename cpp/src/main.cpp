// adi_port: the C++ multithreaded port of the Douglas ADI time loop.
//
//   adi_port CASE_DIR OUT_DIR [--threads T]
//       Check mode. Reads a fixture case, runs the six kernels on kernel_input and the
//       full time loop from U0, and writes the seven outputs for
//       `python -m adi_heston.fixtures check CASE_DIR OUT_DIR`.
//
//   adi_port INPUTS_DIR --bench [--threads T] [--repeats R]
//       Benchmark mode. Runs the time loop once to warm up, then R timed runs, and prints
//       one CSV line: threads,rows,cols,steps,median_ms_per_step,min_ms_per_step.
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#include "io.hpp"
#include "kernels.hpp"
#include "solver.hpp"

namespace fs = std::filesystem;
using adi::Operators;

namespace {

Operators load_operators(const fs::path& dir) {
    Operators op;
    op.w_s = adi::read_bin(dir / "w_s.bin");
    op.w_v = adi::read_bin(dir / "w_v.bin");
    op.rows = op.w_s.size() / 3;
    op.cols = op.w_v.size() / 3;
    op.a1 = adi::read_bin(dir / "a1.bin");
    op.a2 = adi::read_bin(dir / "a2.bin");
    op.a0_coef = adi::read_bin(dir / "a0_coef.bin");
    if (op.rows < 3 || op.cols < 3 || op.a1.size() != 3 * op.size() || op.a2.size() != 3 * op.size() ||
        op.a0_coef.size() != op.size()) {
        throw std::runtime_error("operator files in " + dir.string() + " have inconsistent sizes");
    }
    return op;
}

std::vector<adi::Step> load_schedule(const fs::path& dir) {
    const std::vector<double> raw = adi::read_bin(dir / "schedule.bin");
    if (raw.size() % 3 != 0) throw std::runtime_error("schedule.bin must hold rows of 3 values");
    std::vector<adi::Step> steps;
    for (std::size_t k = 0; k < raw.size(); k += 3) steps.push_back({raw[k], raw[k + 1], raw[k + 2]});
    return steps;
}

std::vector<double> load_grid_array(const fs::path& path, const Operators& op) {
    std::vector<double> values = adi::read_bin(path);
    if (values.size() != op.size()) throw std::runtime_error(path.string() + " does not match the grid size");
    return values;
}

void check_mode(const fs::path& case_dir, const fs::path& out_dir, std::size_t threads) {
    const Operators op = load_operators(case_dir);
    const std::vector<double> X = load_grid_array(case_dir / "kernel_input.bin", op);
    const std::vector<double> scalars = adi::read_bin(case_dir / "scalars.bin");
    if (scalars.size() != 4) throw std::runtime_error("scalars.bin must hold 4 values");
    const double solve_c = scalars[0];
    const adi::Step one_step{scalars[1], scalars[2], scalars[3]};
    fs::create_directories(out_dir);

    // Single kernels on kernel_input, through the same row and column blocks as the time loop.
    std::vector<double> A0X(op.size()), A1X(op.size()), A2X(op.size()), solved_A1(op.size()), solved_A2(op.size());
    adi::run_team(threads, op.rows, op.cols, [&](std::size_t, const adi::Block& block, std::barrier<>&) {
        adi::Scratch scratch(op, block);
        adi::kernels::apply_A0(op, X.data(), A0X.data(), block.row_begin, block.row_end, scratch.v_derivative.data());
        adi::kernels::apply_A1(op, X.data(), A1X.data(), block.row_begin, block.row_end);
        adi::kernels::apply_A2(op, X.data(), A2X.data(), block.row_begin, block.row_end);
        adi::kernels::solve_A1(op, solve_c, X.data(), solved_A1.data(), block.col_begin, block.col_end,
                               scratch.c_star_s.data(), scratch.d_star_s.data());
        adi::kernels::solve_A2(op, solve_c, X.data(), solved_A2.data(), block.row_begin, block.row_end,
                               scratch.c_star_v.data(), scratch.d_star_v.data());
    });
    adi::write_bin(out_dir / "apply_A0.bin", A0X);
    adi::write_bin(out_dir / "apply_A1.bin", A1X);
    adi::write_bin(out_dir / "apply_A2.bin", A2X);
    adi::write_bin(out_dir / "solve_A1.bin", solved_A1);
    adi::write_bin(out_dir / "solve_A2.bin", solved_A2);

    adi::DouglasSolver solver(op, threads);
    std::vector<double> after_one_step = X;
    solver.run(after_one_step, {one_step});
    adi::write_bin(out_dir / "douglas_step.bin", after_one_step);

    std::vector<double> U = load_grid_array(case_dir / "U0.bin", op);
    solver.run(U, load_schedule(case_dir));
    adi::write_bin(out_dir / "U_final.bin", U);
}

void bench_mode(const fs::path& dir, std::size_t threads, std::size_t repeats) {
    const Operators op = load_operators(dir);
    const std::vector<double> U0 = load_grid_array(dir / "U0.bin", op);
    const std::vector<adi::Step> schedule = load_schedule(dir);
    adi::DouglasSolver solver(op, threads);

    std::vector<double> ms_per_step;
    for (std::size_t run = 0; run <= repeats; ++run) {  // run 0 is the warm-up
        std::vector<double> U = U0;
        const double seconds = solver.run(U, schedule);
        if (run > 0) ms_per_step.push_back(1e3 * seconds / static_cast<double>(schedule.size()));
    }
    std::sort(ms_per_step.begin(), ms_per_step.end());
    std::printf("%zu,%zu,%zu,%zu,%.6f,%.6f\n", threads, op.rows, op.cols, schedule.size(),
                ms_per_step[ms_per_step.size() / 2], ms_per_step.front());
}

[[noreturn]] void usage() {
    std::fprintf(stderr,
                 "usage: adi_port CASE_DIR OUT_DIR [--threads T]\n"
                 "       adi_port INPUTS_DIR --bench [--threads T] [--repeats R]\n");
    std::exit(2);
}

std::size_t parse_count(const char* text) {
    const long value = std::strtol(text, nullptr, 10);
    if (value < 1) usage();
    return static_cast<std::size_t>(value);
}

}  // namespace

int main(int argc, char** argv) {
    std::vector<std::string> paths;
    std::size_t threads = 1, repeats = 5;
    bool bench = false;
    for (int a = 1; a < argc; ++a) {
        const std::string arg = argv[a];
        if (arg == "--bench") {
            bench = true;
        } else if (arg == "--threads" && a + 1 < argc) {
            threads = parse_count(argv[++a]);
        } else if (arg == "--repeats" && a + 1 < argc) {
            repeats = parse_count(argv[++a]);
        } else if (!arg.empty() && arg[0] != '-') {
            paths.push_back(arg);
        } else {
            usage();
        }
    }
    try {
        if (bench && paths.size() == 1) {
            bench_mode(paths[0], threads, repeats);
        } else if (!bench && paths.size() == 2) {
            check_mode(paths[0], paths[1], threads);
        } else {
            usage();
        }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "adi_port: %s\n", e.what());
        return 1;
    }
    return 0;
}
