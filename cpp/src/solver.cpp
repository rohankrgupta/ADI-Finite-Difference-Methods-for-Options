#include "solver.hpp"

#include <algorithm>
#include <chrono>
#include <utility>

namespace adi {

namespace {

// The half-open range [begin, end) of share t when `count` items are split into `parts` near-equal shares.
std::pair<std::size_t, std::size_t> share(std::size_t count, std::size_t parts, std::size_t t) {
    const std::size_t base = count / parts, extra = count % parts;
    const std::size_t begin = t * base + std::min(t, extra);
    return {begin, begin + base + (t < extra ? 1 : 0)};
}

}  // namespace

std::vector<Block> make_blocks(std::size_t threads, std::size_t rows, std::size_t cols) {
    std::vector<Block> blocks;
    for (std::size_t t = 0; t < threads; ++t) {
        const auto [row_begin, row_end] = share(rows, threads, t);
        const auto [col_begin, col_end] = share(cols, threads, t);
        blocks.push_back({row_begin, row_end, col_begin, col_end});
    }
    return blocks;
}

Scratch::Scratch(const Operators& op, const Block& block)
    : v_derivative((block.row_end - block.row_begin + 2) * op.cols),
      c_star_s(op.rows * (block.col_end - block.col_begin)),
      d_star_s(c_star_s.size()),
      c_star_v(op.cols),
      d_star_v(op.cols) {}

DouglasSolver::DouglasSolver(const Operators& op, std::size_t threads)
    : op_(op), threads_(threads), A0U_(op.size()), A1U_(op.size()), A2U_(op.size()), R_(op.size()), Y1_(op.size()) {}

double DouglasSolver::run(std::vector<double>& solution, const std::vector<Step>& schedule) {
    const std::size_t cols = op_.cols;
    double* U = solution.data();
    double *A0U = A0U_.data(), *A1U = A1U_.data(), *A2U = A2U_.data(), *R = R_.data(), *Y1 = Y1_.data();
    double seconds = 0.0;

    run_team(threads_, op_.rows, cols, [&](std::size_t worker, const Block& block, std::barrier<>& sync) {
        Scratch scratch(op_, block);
        const std::size_t first = block.row_begin * cols, end = block.row_end * cols;  // this block's rows, flattened

        sync.arrive_and_wait();
        const auto start = std::chrono::steady_clock::now();

        for (const Step& step : schedule) {
            const double dtau = step.dtau, c = step.theta_adi * step.dtau;

            // Phase 1, by rows: A0U, A1U, A2U, then R1 = U + dtau*A U - c*A1U.
            kernels::apply_A0(op_, U, A0U, block.row_begin, block.row_end, scratch.v_derivative.data());
            kernels::apply_A1(op_, U, A1U, block.row_begin, block.row_end);
            kernels::apply_A2(op_, U, A2U, block.row_begin, block.row_end);
            for (std::size_t k = first; k < end; ++k) R[k] = (U[k] + dtau * ((A0U[k] + A1U[k]) + A2U[k])) - c * A1U[k];
            if (block.row_begin == 0 && block.row_end > 0) std::fill(R, R + cols, step.boundary);  // Dirichlet row
            sync.arrive_and_wait();

            // Phase 2, by columns: (I - c*A1) Y1 = R1 along s.
            kernels::solve_A1(op_, c, R, Y1, block.col_begin, block.col_end, scratch.c_star_s.data(),
                              scratch.d_star_s.data());
            sync.arrive_and_wait();

            // Phase 3, by rows: R2 = Y1 - c*A2U, then (I - c*A2) U = R2 along v, in place.
            for (std::size_t k = first; k < end; ++k) R[k] = Y1[k] - c * A2U[k];
            kernels::solve_A2(op_, c, R, U, block.row_begin, block.row_end, scratch.c_star_v.data(),
                              scratch.d_star_v.data());
            sync.arrive_and_wait();
        }

        if (worker == 0) seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    });
    return seconds;
}

}  // namespace adi
