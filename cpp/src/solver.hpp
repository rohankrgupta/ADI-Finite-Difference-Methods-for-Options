// Threads and the ADI time loop.
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

// The four ADI schemes, numbered as in python/adi_heston/fixtures.py (SCHEME_IDS).
enum class Scheme { douglas = 0, craig_sneyd = 1, modified_craig_sneyd = 2, hundsdorfer_verwer = 3 };

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

// Runs the ADI time loop of one scheme; the work arrays are allocated once and reused by every step.
//
// Douglas takes 3 phases per step. Craig-Sneyd, Modified Craig-Sneyd and Hundsdorfer-Verwer
// keep the Douglas result as Y2 and add a correction plus a second pair of solves: 6 phases.
// The first `startup_half_steps` rows of every schedule (the Rannacher start-up) always use Douglas.
class AdiSolver {
public:
    AdiSolver(const Operators& op, std::size_t threads, Scheme scheme = Scheme::douglas,
              std::size_t startup_half_steps = 0);

    // Advance `solution` in place through every step of the schedule. For an American option,
    // pass the payoff and the multiplier `lam` (updated in place); for a European option, pass
    // nullptr for both. Returns the seconds spent in the loop, measured by worker 0 between
    // barriers, so thread start-up is excluded.
    double run(std::vector<double>& solution, const std::vector<Step>& schedule, const double* payoff = nullptr,
               double* lam = nullptr);

private:
    const Operators& op_;
    std::size_t threads_;
    Scheme scheme_;
    std::size_t startup_half_steps_;
    std::vector<double> A0U_, A1U_, A2U_, R_, Y1_;  // every scheme
    std::vector<double> Y0_, Y2_, A0Y2_, A1Y2_, A2Y2_;  // corrected schemes only
};

}  // namespace adi
