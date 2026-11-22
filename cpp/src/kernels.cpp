// The arithmetic in every kernel follows the Python reference operation by operation
// (operators.py, fd.py, tridiag.py), so the results agree bit for bit.
#include "kernels.hpp"

#include <algorithm>

namespace adi::kernels {

void apply_A1(const Operators& op, const double* U, double* out, std::size_t row_begin, std::size_t row_end) {
    const std::size_t cols = op.cols;
    const double* lower = op.a1_plane(0);
    const double* diag = op.a1_plane(1);
    const double* upper = op.a1_plane(2);
    for (std::size_t i = row_begin; i < row_end; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            const std::size_t k = i * cols + j;
            double value = diag[k] * U[k];
            if (i > 0) value += lower[k] * U[k - cols];
            if (i + 1 < op.rows) value += upper[k] * U[k + cols];
            out[k] = value;
        }
    }
}

void apply_A2(const Operators& op, const double* U, double* out, std::size_t row_begin, std::size_t row_end) {
    const std::size_t cols = op.cols;
    const double* lower = op.a2_plane(0);
    const double* diag = op.a2_plane(1);
    const double* upper = op.a2_plane(2);
    for (std::size_t i = row_begin; i < row_end; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            const std::size_t k = i * cols + j;
            double value = diag[k] * U[k];
            if (j > 0) value += lower[k] * U[k - 1];
            if (j + 1 < cols) value += upper[k] * U[k + 1];
            out[k] = value;
        }
    }
}

void apply_A0(const Operators& op, const double* U, double* out, std::size_t row_begin, std::size_t row_end,
              double* v_derivative) {
    const std::size_t rows = op.rows, cols = op.cols;
    if (row_begin >= row_end) return;

    // v-derivative, zero in the first and last column, on grid rows [first, end): the block
    // plus one neighbour on each side. Grid row i is row (i - first) of v_derivative.
    const std::size_t first = row_begin > 0 ? row_begin - 1 : 0;
    const std::size_t end = row_end < rows ? row_end + 1 : rows;
    for (std::size_t i = first; i < end; ++i) {
        const double* u_row = U + i * cols;
        double* dv_row = v_derivative + (i - first) * cols;
        dv_row[0] = 0.0;
        dv_row[cols - 1] = 0.0;
        for (std::size_t j = 1; j + 1 < cols; ++j) {
            const double* w = op.w_v.data() + 3 * j;  // weights on (j-1, j, j+1)
            dv_row[j] = w[0] * u_row[j - 1] + w[1] * u_row[j] + w[2] * u_row[j + 1];
        }
    }

    // s-derivative of that on interior rows, times rho*xi*s*v; zero on the first and last row.
    for (std::size_t i = row_begin; i < row_end; ++i) {
        double* out_row = out + i * cols;
        if (i == 0 || i + 1 == rows) {
            std::fill(out_row, out_row + cols, 0.0);
            continue;
        }
        const double* w = op.w_s.data() + 3 * i;  // weights on (i-1, i, i+1)
        const double* below = v_derivative + (i - 1 - first) * cols;
        const double* here = below + cols;
        const double* above = here + cols;
        const double* coef = op.a0_coef.data() + i * cols;
        for (std::size_t j = 0; j < cols; ++j) {
            out_row[j] = coef[j] * (w[0] * below[j] + w[1] * here[j] + w[2] * above[j]);
        }
    }
}

void thomas_line(const double* lower, const double* diag, const double* upper, double c, const double* rhs,
                 double* x, std::size_t n, std::size_t stride, double* c_star, double* d_star) {
    // System coefficients: sub = -c*lower, main = 1 - c*diag, super = -c*upper.
    const double pivot = 1.0 - c * diag[0];
    c_star[0] = (-c * upper[0]) / pivot;
    d_star[0] = rhs[0] / pivot;
    for (std::size_t k = 1; k < n; ++k) {
        const std::size_t offset = k * stride;
        const double sub = -c * lower[offset];
        const double denom = (1.0 - c * diag[offset]) - sub * c_star[k - 1];
        c_star[k] = (-c * upper[offset]) / denom;
        d_star[k] = (rhs[offset] - sub * d_star[k - 1]) / denom;
    }
    x[(n - 1) * stride] = d_star[n - 1];
    for (std::size_t k = n - 1; k-- > 0;) {
        x[k * stride] = d_star[k] - c_star[k] * x[(k + 1) * stride];
    }
}

void solve_A1(const Operators& op, double c, const double* R, double* X, std::size_t col_begin, std::size_t col_end,
              double* c_star, double* d_star) {
    const std::size_t rows = op.rows, cols = op.cols, width = col_end - col_begin;  // width: columns in the block
    if (width == 0) return;
    const double* lower = op.a1_plane(0);
    const double* diag = op.a1_plane(1);
    const double* upper = op.a1_plane(2);

    // Forward sweep. c_star and d_star hold `width` values for each grid row.
    for (std::size_t j = 0; j < width; ++j) {
        const std::size_t k = col_begin + j;
        const double pivot = 1.0 - c * diag[k];
        c_star[j] = (-c * upper[k]) / pivot;
        d_star[j] = R[k] / pivot;
    }
    for (std::size_t i = 1; i < rows; ++i) {
        const double* c_prev = c_star + (i - 1) * width;
        const double* d_prev = d_star + (i - 1) * width;
        double* c_row = c_star + i * width;
        double* d_row = d_star + i * width;
        for (std::size_t j = 0; j < width; ++j) {
            const std::size_t k = i * cols + col_begin + j;
            const double sub = -c * lower[k];
            const double denom = (1.0 - c * diag[k]) - sub * c_prev[j];
            c_row[j] = (-c * upper[k]) / denom;
            d_row[j] = (R[k] - sub * d_prev[j]) / denom;
        }
    }

    // Back substitution, row by row.
    for (std::size_t j = 0; j < width; ++j) X[(rows - 1) * cols + col_begin + j] = d_star[(rows - 1) * width + j];
    for (std::size_t i = rows - 1; i-- > 0;) {
        const double* c_row = c_star + i * width;
        const double* d_row = d_star + i * width;
        for (std::size_t j = 0; j < width; ++j) {
            const std::size_t k = i * cols + col_begin + j;
            X[k] = d_row[j] - c_row[j] * X[k + cols];
        }
    }
}

void solve_A2(const Operators& op, double c, const double* R, double* X, std::size_t row_begin, std::size_t row_end,
              double* c_star, double* d_star) {
    const std::size_t cols = op.cols;
    for (std::size_t i = row_begin; i < row_end; ++i) {
        const std::size_t row = i * cols;
        thomas_line(op.a2_plane(0) + row, op.a2_plane(1) + row, op.a2_plane(2) + row, c, R + row, X + row, cols, 1,
                    c_star, d_star);
    }
}

void ikonen_toivanen_update(const Operators& op, double* U, double* lam, const double* payoff, double dtau,
                            std::size_t row_begin, std::size_t row_end) {
    for (std::size_t k = row_begin * op.cols; k < row_end * op.cols; ++k) {
        const double u_bar = U[k];
        U[k] = std::max(u_bar - dtau * lam[k], payoff[k]);
        lam[k] = std::max(0.0, lam[k] + (payoff[k] - u_bar) / dtau);
    }
}

}  // namespace adi::kernels
