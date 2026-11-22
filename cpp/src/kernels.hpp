// The six kernels of the ADI time step (docs/phase1_summary.md, Section 4.3).
//
// Every array is row-major: node (i, j) is at index i*cols + j, with i along s
// (rows = m+1) and j along v (cols = n+1). Each kernel works on a half-open range of
// grid lines chosen by the caller: a thread passes its own share, and one thread passing
// the full range is the serial version. The kernels do no threading themselves.
#pragma once

#include <cstddef>
#include <vector>

namespace adi {

// The Heston operator split A = A0 + A1 + A2, exactly as Python's build_heston_operators
// writes it to the fixture files.
struct Operators {
    std::size_t rows = 0;         // m + 1, along s
    std::size_t cols = 0;         // n + 1, along v
    std::vector<double> a1;       // 3 planes (lower, diag, upper) along s, each rows*cols
    std::vector<double> a2;       // 3 planes (lower, diag, upper) along v, each rows*cols
    std::vector<double> a0_coef;  // rho*xi*s*v, rows*cols, zero on all edges
    std::vector<double> w_s;      // first-derivative weights in s, rows*3
    std::vector<double> w_v;      // first-derivative weights in v, cols*3

    std::size_t size() const { return rows * cols; }
    const double* a1_plane(int k) const { return a1.data() + static_cast<std::size_t>(k) * size(); }
    const double* a2_plane(int k) const { return a2.data() + static_cast<std::size_t>(k) * size(); }
};

namespace kernels {

// Stencil sweeps: out = A1*U or A2*U on rows [row_begin, row_end). U and out hold the whole grid.
void apply_A1(const Operators& op, const double* U, double* out, std::size_t row_begin, std::size_t row_end);
void apply_A2(const Operators& op, const double* U, double* out, std::size_t row_begin, std::size_t row_end);

// out = A0*U on rows [row_begin, row_end). It first takes the v-derivative of those rows and
// one neighbour on each side into `v_derivative`, which must hold
// (row_end - row_begin + 2) * cols doubles, then the s-derivative of that.
void apply_A0(const Operators& op, const double* U, double* out, std::size_t row_begin, std::size_t row_end,
              double* v_derivative);

// Thomas solve of (I - c*A) x = rhs on one grid line of n nodes spaced `stride` apart.
// lower, diag and upper point at the line's first coefficient of the operator A.
// c_star and d_star are scratch of length n; x may be the same array as rhs.
// This per-line routine is the body a GPU thread would run.
void thomas_line(const double* lower, const double* diag, const double* upper, double c, const double* rhs,
                 double* x, std::size_t n, std::size_t stride, double* c_star, double* d_star);

// (I - c*A1) X = R along s for columns [col_begin, col_end). It runs the recurrence of
// thomas_line for all columns of the block at once, row by row, which reads memory in order.
// c_star and d_star are scratch of rows*(col_end - col_begin) doubles each.
void solve_A1(const Operators& op, double c, const double* R, double* X, std::size_t col_begin, std::size_t col_end,
              double* c_star, double* d_star);

// (I - c*A2) X = R along v for rows [row_begin, row_end): one contiguous line per row.
// c_star and d_star are scratch of cols doubles each.
void solve_A2(const Operators& op, double c, const double* R, double* X, std::size_t row_begin, std::size_t row_end,
              double* c_star, double* d_star);

}  // namespace kernels
}  // namespace adi
