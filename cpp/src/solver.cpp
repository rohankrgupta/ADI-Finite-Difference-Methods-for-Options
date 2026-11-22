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

AdiSolver::AdiSolver(const Operators& op, std::size_t threads, Scheme scheme, std::size_t startup_half_steps)
    : op_(op),
      threads_(threads),
      scheme_(scheme),
      startup_half_steps_(startup_half_steps),
      A0U_(op.size()),
      A1U_(op.size()),
      A2U_(op.size()),
      R_(op.size()),
      Y1_(op.size()) {
    if (scheme != Scheme::douglas) {
        for (auto* array : {&Y0_, &Y2_, &A0Y2_, &A1Y2_, &A2Y2_}) array->resize(op.size());
    }
}

double AdiSolver::run(std::vector<double>& solution, const std::vector<Step>& schedule, const double* payoff,
                      double* lam) {
    const std::size_t cols = op_.cols;
    double* U = solution.data();
    double *A0U = A0U_.data(), *A1U = A1U_.data(), *A2U = A2U_.data(), *R = R_.data(), *Y1 = Y1_.data();
    double *Y0 = Y0_.data(), *Y2 = Y2_.data(), *A0Y2 = A0Y2_.data(), *A1Y2 = A1Y2_.data(), *A2Y2 = A2Y2_.data();
    double seconds = 0.0;

    run_team(threads_, op_.rows, cols, [&](std::size_t worker, const Block& block, std::barrier<>& sync) {
        Scratch scratch(op_, block);
        const std::size_t row_begin = block.row_begin, row_end = block.row_end;
        const std::size_t first = row_begin * cols, end = row_end * cols;  // this block's rows, flattened
        const bool owns_row_0 = row_begin == 0 && row_end > 0;

        sync.arrive_and_wait();
        const auto start = std::chrono::steady_clock::now();

        for (std::size_t n = 0; n < schedule.size(); ++n) {
            const Step& step = schedule[n];
            const Scheme scheme = n < startup_half_steps_ ? Scheme::douglas : scheme_;
            const bool corrected = scheme != Scheme::douglas;
            const bool second_pass_uses_Y2 = scheme == Scheme::hundsdorfer_verwer;  // the others use A U
            const double dtau = step.dtau, theta = step.theta_adi, c = theta * dtau;

            // Phase 1, by rows: A0U, A1U, A2U, the predictor Y0 = U + dtau*A U (+ dtau*lam), R1 = Y0 - c*A1U.
            kernels::apply_A0(op_, U, A0U, row_begin, row_end, scratch.v_derivative.data());
            kernels::apply_A1(op_, U, A1U, row_begin, row_end);
            kernels::apply_A2(op_, U, A2U, row_begin, row_end);
            for (std::size_t k = first; k < end; ++k) {
                double y0 = U[k] + dtau * ((A0U[k] + A1U[k]) + A2U[k]);
                if (lam) y0 += dtau * lam[k];
                if (corrected) Y0[k] = y0;
                R[k] = y0 - c * A1U[k];
            }
            if (owns_row_0) std::fill(R, R + cols, step.boundary);  // Dirichlet row
            sync.arrive_and_wait();

            // Phase 2, by columns: (I - c*A1) Y1 = R1 along s.
            kernels::solve_A1(op_, c, R, Y1, block.col_begin, block.col_end, scratch.c_star_s.data(),
                              scratch.d_star_s.data());
            sync.arrive_and_wait();

            // Phase 3, by rows: (I - c*A2) Y2 = Y1 - c*A2U along v. Douglas writes Y2 straight into U.
            for (std::size_t k = first; k < end; ++k) R[k] = Y1[k] - c * A2U[k];
            kernels::solve_A2(op_, c, R, corrected ? Y2 : U, row_begin, row_end, scratch.c_star_v.data(),
                              scratch.d_star_v.data());
            if (!corrected && lam) kernels::ikonen_toivanen_update(op_, U, lam, payoff, dtau, row_begin, row_end);
            sync.arrive_and_wait();
            if (!corrected) continue;

            // Phase 4, by rows: A applied to Y2, the corrected predictor, and R1 for the second pass
            // (Haentjens & in 't Hout 2015, eqs. 4.2-4.4; Python: solver.py).
            kernels::apply_A0(op_, Y2, A0Y2, row_begin, row_end, scratch.v_derivative.data());
            if (scheme != Scheme::craig_sneyd) {
                kernels::apply_A1(op_, Y2, A1Y2, row_begin, row_end);
                kernels::apply_A2(op_, Y2, A2Y2, row_begin, row_end);
            }
            const auto change_in_A = [&](std::size_t k) {  // (A Y2 - A U) at node k, summed as in Python
                return ((A0Y2[k] + A1Y2[k]) + A2Y2[k]) - ((A0U[k] + A1U[k]) + A2U[k]);
            };
            for (std::size_t k = first; k < end; ++k) {
                double y0_tilde = 0.0;
                switch (scheme) {
                    case Scheme::craig_sneyd:  // corrects the mixed term only
                        y0_tilde = Y0[k] + 0.5 * dtau * (A0Y2[k] - A0U[k]);
                        break;
                    case Scheme::modified_craig_sneyd:
                        y0_tilde = (Y0[k] + c * (A0Y2[k] - A0U[k])) + (0.5 - theta) * dtau * change_in_A(k);
                        break;
                    case Scheme::hundsdorfer_verwer:
                        y0_tilde = Y0[k] + 0.5 * dtau * change_in_A(k);
                        break;
                    case Scheme::douglas:  // not reached: Douglas has no second pass
                        break;
                }
                R[k] = y0_tilde - c * (second_pass_uses_Y2 ? A1Y2[k] : A1U[k]);
            }
            if (owns_row_0) std::fill(R, R + cols, step.boundary);
            sync.arrive_and_wait();

            // Phase 5, by columns: the second s-solve.
            kernels::solve_A1(op_, c, R, Y1, block.col_begin, block.col_end, scratch.c_star_s.data(),
                              scratch.d_star_s.data());
            sync.arrive_and_wait();

            // Phase 6, by rows: the second v-solve, into U, then the American update.
            for (std::size_t k = first; k < end; ++k) R[k] = Y1[k] - c * (second_pass_uses_Y2 ? A2Y2[k] : A2U[k]);
            kernels::solve_A2(op_, c, R, U, row_begin, row_end, scratch.c_star_v.data(), scratch.d_star_v.data());
            if (lam) kernels::ikonen_toivanen_update(op_, U, lam, payoff, dtau, row_begin, row_end);
            sync.arrive_and_wait();
        }

        if (worker == 0) seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    });
    return seconds;
}

}  // namespace adi
