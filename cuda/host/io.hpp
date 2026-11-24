// -----------------------------------------------------------------------------
// Reading the fixture and benchmark-input folders written by
// python/adi_heston/fixtures.py, and writing outputs in the same format
// (raw float64, little-endian, row-major, no header).
//
// Project: ADI finite-difference methods for American options under Heston
//          Math/CS 714, UW–Madison, Fall 2026
// Author:  Ganesh Arivoli <arivoli@wisc.edu>
// -----------------------------------------------------------------------------
#pragma once

#include <cstddef>
#include <filesystem>
#include <vector>

namespace adi {

// One row of the time-step schedule written by Python's time_schedule().
struct Step {
    double dtau;
    double theta;     // theta_adi
    double boundary;  // Dirichlet value on the s = 0 row at the end of the step
};

// Everything a run needs from one folder. 2D arrays are row-major: node (i, j) is at i*cols + j,
// with i along s and j along v.
struct HostProblem {
    std::size_t rows = 0;  // m + 1, along s
    std::size_t cols = 0;  // n + 1, along v

    std::vector<double> a1;       // A1: 3 planes (lower, diag, upper) of rows*cols, coupling along s
    std::vector<double> a2;       // A2: 3 rows (lower, diag, upper) of cols values; A2 does not depend on s
    std::vector<double> a0_coef;  // rho*xi*s*v, rows*cols, zero on all edges
    std::vector<double> w_s;      // first-derivative weights in s, rows*3
    std::vector<double> w_v;      // first-derivative weights in v, cols*3
    std::vector<double> U0;       // the payoff on the grid, rows*cols
    std::vector<double> payoff;   // the payoff per s-node, rows values (every row of U0 is constant)
    std::vector<Step> schedule;

    // From settings.bin. Folders without it (the base cases) are Douglas for a European put; their
    // start-up rows are Douglas steps anyway, so startup_half_steps stays 0.
    int scheme = 0;  // 0 Douglas, 1 Craig-Sneyd, 2 Modified Craig-Sneyd, 3 Hundsdorfer-Verwer
    bool american = false;
    std::size_t startup_half_steps = 0;

    // Fixture cases only (empty in benchmark-input folders).
    std::vector<double> kernel_input;  // rows*cols, input of the one-kernel and one-step checks
    std::vector<double> lam_input;     // rows*cols, multiplier for the one-step check of American cases
    std::vector<double> scalars;       // [solve_c, dtau, theta_adi, boundary]

    std::size_t size() const { return rows * cols; }
};

// Read a whole .bin file.
std::vector<double> read_bin(const std::filesystem::path& path);

// Write values in the same format.
void write_bin(const std::filesystem::path& path, const std::vector<double>& values);

// Read a fixture case or benchmark-input folder and run the load-time checks: every size matches
// the grid, A2's rows below row 0 are byte-identical and row 0 is zero, every row of U0 is constant,
// and settings.bin is valid. Throws std::runtime_error naming the file if anything is wrong.
HostProblem load_case(const std::filesystem::path& dir);

}  // namespace adi
