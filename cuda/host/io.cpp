// -----------------------------------------------------------------------------
// Fixture loading with the load-time checks, and raw float64 file I/O.
//
// Project: ADI finite-difference methods for American options under Heston
//          Math/CS 714, UW–Madison, Fall 2026
// Author:  Ganesh Arivoli <arivoli@wisc.edu>
// -----------------------------------------------------------------------------
#include "host/io.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <string>

static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "the .bin files are little-endian");

namespace fs = std::filesystem;

namespace adi {

std::vector<double> read_bin(const fs::path& path) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in) throw std::runtime_error("cannot open " + path.string());
    const auto bytes = static_cast<std::size_t>(in.tellg());
    if (bytes % sizeof(double) != 0) {
        throw std::runtime_error(path.string() + " does not hold a whole number of float64 values");
    }
    std::vector<double> values(bytes / sizeof(double));
    in.seekg(0);
    in.read(reinterpret_cast<char*>(values.data()), static_cast<std::streamsize>(bytes));
    if (!in) throw std::runtime_error("cannot read " + path.string());
    return values;
}

void write_bin(const fs::path& path, const std::vector<double>& values) {
    std::ofstream out(path, std::ios::binary);
    out.write(reinterpret_cast<const char*>(values.data()),
              static_cast<std::streamsize>(values.size() * sizeof(double)));
    if (!out) throw std::runtime_error("cannot write " + path.string());
}

namespace {

std::vector<double> read_sized(const fs::path& path, std::size_t expected) {
    std::vector<double> values = read_bin(path);
    if (values.size() != expected) {
        throw std::runtime_error(path.string() + ": expected " + std::to_string(expected) + " values, found " +
                                 std::to_string(values.size()));
    }
    return values;
}

std::vector<double> read_optional(const fs::path& path, std::size_t expected) {
    return fs::exists(path) ? read_sized(path, expected) : std::vector<double>{};
}

// A2 = 1/2 xi^2 v u_vv + kappa(theta - v) u_v - r/2 u does not depend on s, so every plane must repeat
// one row for i >= 1, with row 0 (the s = 0 Dirichlet row) all zero. Returns that row of each plane.
std::vector<double> compress_a2(const std::vector<double>& a2, std::size_t rows, std::size_t cols,
                                const fs::path& path) {
    std::vector<double> compressed(3 * cols);
    for (std::size_t k = 0; k < 3; ++k) {
        const double* plane = a2.data() + k * rows * cols;
        if (std::any_of(plane, plane + cols, [](double x) { return x != 0.0; })) {
            throw std::runtime_error(path.string() + ": plane " + std::to_string(k) + " row 0 is not zero");
        }
        for (std::size_t i = 2; i < rows; ++i) {
            if (std::memcmp(plane + i * cols, plane + cols, cols * sizeof(double)) != 0) {
                throw std::runtime_error(path.string() + ": plane " + std::to_string(k) + " row " +
                                         std::to_string(i) +
                                         " differs from row 1; the CUDA port assumes A2 does not depend on s");
            }
        }
        std::copy(plane + cols, plane + 2 * cols, compressed.begin() + static_cast<std::ptrdiff_t>(k * cols));
    }
    return compressed;
}

// The payoff max(K - s, 0) depends only on s, so every row of U0 must be constant. Returns one value per row.
std::vector<double> payoff_per_row(const std::vector<double>& U0, std::size_t rows, std::size_t cols,
                                   const fs::path& path) {
    std::vector<double> payoff(rows);
    for (std::size_t i = 0; i < rows; ++i) {
        const double* row = U0.data() + i * cols;
        for (std::size_t j = 1; j < cols; ++j) {
            if (std::memcmp(row + j, row, sizeof(double)) != 0) {
                throw std::runtime_error(path.string() + ": row " + std::to_string(i) + " is not constant");
            }
        }
        payoff[i] = row[0];
    }
    return payoff;
}

bool is_whole(double x, double lo, double hi) { return x >= lo && x <= hi && x == std::floor(x); }

}  // namespace

HostProblem load_case(const fs::path& dir) {
    HostProblem p;
    p.w_s = read_bin(dir / "w_s.bin");
    p.w_v = read_bin(dir / "w_v.bin");
    if (p.w_s.size() % 3 != 0 || p.w_v.size() % 3 != 0) {
        throw std::runtime_error(dir.string() + ": w_s.bin and w_v.bin must hold rows of 3 weights");
    }
    p.rows = p.w_s.size() / 3;
    p.cols = p.w_v.size() / 3;
    if (p.rows < 3 || p.cols < 3) throw std::runtime_error(dir.string() + ": the grid needs at least 3 x 3 nodes");
    const std::size_t n = p.size();

    p.a1 = read_sized(dir / "a1.bin", 3 * n);
    p.a2 = compress_a2(read_sized(dir / "a2.bin", 3 * n), p.rows, p.cols, dir / "a2.bin");
    p.a0_coef = read_sized(dir / "a0_coef.bin", n);
    p.U0 = read_sized(dir / "U0.bin", n);
    p.payoff = payoff_per_row(p.U0, p.rows, p.cols, dir / "U0.bin");

    const std::vector<double> raw = read_bin(dir / "schedule.bin");
    if (raw.empty() || raw.size() % 3 != 0) {
        throw std::runtime_error((dir / "schedule.bin").string() + " must hold one or more rows of 3 values");
    }
    for (std::size_t k = 0; k < raw.size(); k += 3) p.schedule.push_back({raw[k], raw[k + 1], raw[k + 2]});

    const fs::path settings_path = dir / "settings.bin";
    if (fs::exists(settings_path)) {
        const std::vector<double> settings = read_sized(settings_path, 3);
        if (!is_whole(settings[0], 0, 3)) throw std::runtime_error(settings_path.string() + ": scheme id must be 0 to 3");
        if (!is_whole(settings[1], 0, 1)) throw std::runtime_error(settings_path.string() + ": american must be 0 or 1");
        if (!is_whole(settings[2], 0, static_cast<double>(p.schedule.size())) ||
            static_cast<std::size_t>(settings[2]) % 2 != 0) {
            throw std::runtime_error(settings_path.string() + ": start-up half-steps must be even and fit the schedule");
        }
        p.scheme = static_cast<int>(settings[0]);
        p.american = settings[1] != 0.0;
        p.startup_half_steps = static_cast<std::size_t>(settings[2]);
    }

    p.kernel_input = read_optional(dir / "kernel_input.bin", n);
    p.lam_input = read_optional(dir / "lam_input.bin", n);
    p.scalars = read_optional(dir / "scalars.bin", 4);
    return p;
}

}  // namespace adi
