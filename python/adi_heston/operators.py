"""The Heston operator split by direction, A = A0 + A1 + A2, stored as stencil coefficients.

    A1 (s-direction):  1/2 s^2 v u_ss + (r - q) s u_s     - r/2 u
    A2 (v-direction):  1/2 xi^2 v u_vv + kappa(theta - v) u_v - r/2 u
    A0 (mixed):        rho xi s v u_sv

A1 and A2 are stored as arrays of shape (3, m+1, n+1) holding the (lower, diag, upper)
coefficients along their own axis: axis 0 (s) for A1, axis 1 (v) for A2. A0 is a
coefficient array times the 9-point mixed stencil.

Convection uses central differences where the cell Peclet number is at most 2, which
is exactly where the central off-diagonal weights stay non-negative. Elsewhere it uses
first-order upwind differences, taken from the side that information flows from.
Either way, every row has off-diagonals >= 0 and sums to -r/2 (or to 0 on the
Dirichlet row). So (I - c*A_j) is strictly diagonally dominant for every c > 0, and
the Thomas algorithm is stable without pivoting.

Boundaries (each operator only handles its own direction's edges):
    s = 0      Dirichlet. All coefficients in row 0 are zero; row 0 holds the boundary value.
    s = S_max  Neumann u_s = 0, using a mirror ghost node. A1 drops its convection there.
    v = 0      No condition: the PDE itself, with a 2-point forward difference for u_v.
    v = V_max  Neumann u_v = 0, using a mirror ghost node. A2 drops its convection there.
    A0 is zero on all four edges.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .fd import first_derivative_weights, mixed_derivative, second_derivative_weights
from .grid import Grid
from .params import HestonParams, OptionSpec
from .tridiag import solve_tridiagonal


@dataclass(frozen=True)
class HestonOperators:
    """The directional pieces of the Heston operator on a fixed grid."""

    grid: Grid
    params: HestonParams
    a1: np.ndarray  # (3, m+1, n+1): lower, diag, upper along axis 0 (s)
    a2: np.ndarray  # (3, m+1, n+1): lower, diag, upper along axis 1 (v)
    a0_coef: np.ndarray  # (m+1, n+1): rho*xi*s*v, zero on all edges
    w_s: np.ndarray  # (m+1, 3) first-derivative weights in s, used by A0
    w_v: np.ndarray  # (n+1, 3) first-derivative weights in v, used by A0
    upwind_s: np.ndarray  # (m+1, n+1) bool: True where A1 uses upwind convection
    upwind_v: np.ndarray  # (m+1, n+1) bool: True where A2 uses upwind convection

    # ---- apply ---------------------------------------------------------------

    def apply_A0(self, U: np.ndarray) -> np.ndarray:
        """Return A0 @ U (the mixed term)."""
        return self.a0_coef * mixed_derivative(self.w_s, self.w_v, U)

    def apply_A1(self, U: np.ndarray) -> np.ndarray:
        """Return A1 @ U (the s-direction stencil sweep)."""
        lo, di, up = self.a1
        out = di * U
        out[1:] += lo[1:] * U[:-1]
        out[:-1] += up[:-1] * U[1:]
        return out

    def apply_A2(self, U: np.ndarray) -> np.ndarray:
        """Return A2 @ U (the v-direction stencil sweep)."""
        lo, di, up = self.a2
        out = di * U
        out[:, 1:] += lo[:, 1:] * U[:, :-1]
        out[:, :-1] += up[:, :-1] * U[:, 1:]
        return out

    def apply(self, U: np.ndarray) -> np.ndarray:
        """Return (A0 + A1 + A2) @ U."""
        return self.apply_A0(U) + self.apply_A1(U) + self.apply_A2(U)

    # ---- solve ---------------------------------------------------------------

    def solve_A1(self, c: float, rhs: np.ndarray) -> np.ndarray:
        """Solve (I - c*A1) Y = rhs: one tridiagonal system per v-line.

        Row 0 has zero coefficients, so it reads Y[0] = rhs[0]. Put the new Dirichlet
        value into rhs[0] before calling.
        """
        lo, di, up = self.a1
        return solve_tridiagonal(-c * lo, 1.0 - c * di, -c * up, rhs, axis=0)

    def solve_A2(self, c: float, rhs: np.ndarray) -> np.ndarray:
        """Solve (I - c*A2) Y = rhs: one tridiagonal system per s-line."""
        lo, di, up = self.a2
        return solve_tridiagonal(-c * lo, 1.0 - c * di, -c * up, rhs, axis=1)


def dirichlet_value(params: HestonParams, option: OptionSpec, tau: float) -> float:
    """Return the value on the s = 0 edge: K*exp(-r*tau) (European) or K (American)."""
    return option.K if option.american else option.K * np.exp(-params.r * tau)


def _interior_three_point(x: np.ndarray, diff: np.ndarray, conv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return coefficients for diff*u_xx + conv*u_x along axis 0, plus the upwind mask.

    Rows 1..N-1 are filled; rows 0 and N are left at zero for the caller.
    diff and conv have shape (N+1, M).
    """
    D1 = first_derivative_weights(x)[:, :, None]  # (N+1, 3, 1)
    D2 = second_derivative_weights(x)[:, :, None]
    central = diff[:, None, :] * D2 + conv[:, None, :] * D1  # (N+1, 3, M)

    h = np.diff(x)
    hm = np.zeros_like(x)
    hp = np.zeros_like(x)
    hm[1:-1], hp[1:-1] = h[:-1], h[1:]
    hm, hp = hm[:, None], hp[:, None]
    interior = np.zeros(x.size, dtype=bool)
    interior[1:-1] = True
    interior = interior[:, None]

    # Information flows from larger x when conv > 0 (forward difference), from smaller x when conv < 0.
    with np.errstate(divide="ignore", invalid="ignore"):
        fwd = np.stack([np.zeros_like(diff), -conv / hp, conv / hp], axis=1)
        bwd = np.stack([-conv / hm, conv / hm, np.zeros_like(diff)], axis=1)
    upwind_conv = np.where((conv > 0)[:, None, :], fwd, bwd)
    upwind = diff[:, None, :] * D2 + upwind_conv

    needs_upwind = ((central[:, 0] < 0.0) | (central[:, 2] < 0.0)) & interior
    coef = np.where(needs_upwind[:, None, :], upwind, central)
    coef[0] = 0.0
    coef[-1] = 0.0
    return np.moveaxis(coef, 1, 0), needs_upwind  # (3, N+1, M)


def build_heston_operators(grid: Grid, params: HestonParams) -> HestonOperators:
    """Assemble A0, A1, A2 with all four boundary edges on ``grid``."""
    p = params
    s, v = grid.s, grid.v
    S, V = grid.mesh()
    half_r = 0.5 * p.r

    # ---- A1: s-direction (axis 0) ----
    diff_s = 0.5 * S**2 * V
    a1, upwind_s = _interior_three_point(s, diff_s, (p.r - p.q) * S)
    a1[1, 1:-1] -= half_r
    h_last = s[-1] - s[-2]  # S_max: mirror ghost node, u_s = 0
    a1[0, -1] = 2.0 * diff_s[-1] / h_last**2
    a1[1, -1] = -a1[0, -1] - half_r
    a1[:, 0] = 0.0  # s = 0: Dirichlet

    # ---- A2: v-direction (axis 1), built on transposed arrays ----
    diff_v = 0.5 * p.xi**2 * V
    a2_t, upwind_v_t = _interior_three_point(v, diff_v.T, (p.kappa * (p.theta - V)).T)
    a2 = np.ascontiguousarray(np.moveaxis(a2_t, 2, 1))  # (3, m+1, n+1)
    upwind_v = upwind_v_t.T
    a2[1, :, 1:-1] -= half_r
    h0 = v[1] - v[0]  # v = 0: PDE with 2-point forward difference for kappa*theta*u_v
    a2[2, :, 0] = p.kappa * p.theta / h0
    a2[1, :, 0] = -a2[2, :, 0] - half_r
    h_top = v[-1] - v[-2]  # V_max: mirror ghost node, u_v = 0
    a2[0, :, -1] = 2.0 * diff_v[:, -1] / h_top**2
    a2[1, :, -1] = -a2[0, :, -1] - half_r
    a2[:, 0, :] = 0.0  # s = 0: Dirichlet wins

    # ---- A0: mixed term, zero on all edges ----
    a0_coef = p.rho * p.xi * S * V
    a0_coef[0, :] = a0_coef[-1, :] = 0.0
    a0_coef[:, 0] = a0_coef[:, -1] = 0.0

    return HestonOperators(
        grid=grid,
        params=params,
        a1=a1,
        a2=a2,
        a0_coef=a0_coef,
        w_s=first_derivative_weights(s),
        w_v=first_derivative_weights(v),
        upwind_s=upwind_s,
        upwind_v=upwind_v,
    )
