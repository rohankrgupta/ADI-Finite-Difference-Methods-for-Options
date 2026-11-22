// Threads and the Douglas time loop.
//
// A team of std::jthread workers is created once per run. Each worker owns a fixed block
// of rows and a fixed block of columns. A std::barrier separates the phases whose data
// crosses blocks, and is the only synchronization (docs/phase1_summary.md, Section 4).
#pragma once

#include <barrier>
#include <cstddef>
#include <thread>
#include <vector>

#include "kernels.hpp"

namespace adi {

// One row of the step schedule written by Python's time_schedule().
struct Step {
    double dtau;
    double theta_adi;
    double boundary;  // Dirichlet value on the s = 0 row at the end of the step
};

// The grid lines a worker owns: rows [row_begin, row_end) and columns [col_begin, col_end).
struct Block {
    std::size_t row_begin, row_end, col_begin, col_end;
};

// Split rows and columns into near-equal contiguous blocks, one per worker. Workers beyond
// the number of lines get empty blocks.
std::vector<Block> make_blocks(std::size_t threads, std::size_t rows, std::size_t cols);

// The work space one worker needs for its block, allocated once before the time loop.
struct Scratch {
    Scratch(const Operators& op, const Block& block);
    std::vector<double> v_derivative;        // apply_A0
    std::vector<double> c_star_s, d_star_s;  // solve_A1: Thomas factors for the block's columns
    std::vector<double> c_star_v, d_star_v;  // solve_A2: Thomas factors for one row
};

// Run body(worker, block, barrier) on `threads` jthreads and wait for all of them.
template <class Body>
void run_team(std::size_t threads, std::size_t rows, std::size_t cols, Body body) {
    const std::vector<Block> blocks = make_blocks(threads, rows, cols);
    std::barrier sync(static_cast<std::ptrdiff_t>(threads));
    std::vector<std::jthread> team;
    team.reserve(threads);
    for (std::size_t worker = 0; worker < threads; ++worker) {
        team.emplace_back([&, worker] { body(worker, blocks[worker], sync); });
    }
}  // the jthreads join here

// Runs the Douglas time loop; the work arrays are allocated once and reused by every step.
class DouglasSolver {
public:
    DouglasSolver(const Operators& op, std::size_t threads);

    // Advance `solution` (in place) through every step of the schedule. Returns the seconds
    // spent in the loop, measured by worker 0 between barriers, so thread start-up is excluded.
    double run(std::vector<double>& solution, const std::vector<Step>& schedule);

private:
    const Operators& op_;
    std::size_t threads_;
    std::vector<double> A0U_, A1U_, A2U_, R_, Y1_;
};

}  // namespace adi
