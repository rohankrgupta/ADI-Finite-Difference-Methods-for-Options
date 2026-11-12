"""Time marching for the Heston PDE with ADI schemes.

The solver starts from the payoff at tau = 0 and marches to tau = T. Each scheme is a
step function with the same signature, so new schemes plug into the same driver,
read-off and tests.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property
from typing import Callable

import numpy as np
from scipy.interpolate import RectBivariateSpline

from .grid import Grid
from .operators import HestonOperators, build_heston_operators, dirichlet_value
from .params import HestonParams, OptionSpec

StepFunction = Callable[[HestonOperators, np.ndarray, float, float, float], np.ndarray]


def douglas_step(ops: HestonOperators, U: np.ndarray, dtau: float, theta_adi: float, boundary_new: float) -> np.ndarray:
    """Advance U by one Douglas ADI step of size dtau.

    Args:
        ops: The Heston operators on the grid.
        U: Values at the old time, shape (m+1, n+1).
        dtau: Step size.
        theta_adi: The ADI weight. Use 1/2 normally, and 1 for the Rannacher start-up.
        boundary_new: Dirichlet value on the s = 0 row at the new time.

    The stages are (a) an explicit predictor with all of A, (b) an implicit correction
    in s, and (c) an implicit correction in v. The cost is 3 stencil sweeps plus 2
    batched tridiagonal solves. Douglas is 1st order in time when the mixed term is
    present (rho != 0) and 2nd order without it (with theta_adi = 1/2).
    """
    A0U, A1U, A2U = ops.apply_A0(U), ops.apply_A1(U), ops.apply_A2(U)
    c = theta_adi * dtau
    Y0 = U + dtau * (A0U + A1U + A2U)  # (a) predict
    R1 = Y0 - c * A1U
    R1[0] = boundary_new  # the Dirichlet row gets the new boundary value
    Y1 = ops.solve_A1(c, R1)  # (b) correct in s
    return ops.solve_A2(c, Y1 - c * A2U)  # (c) correct in v; row 0 passes through unchanged


SCHEMES: dict[str, StepFunction] = {"douglas": douglas_step}


@dataclass(frozen=True)
class SolveResult:
    """The solution at tau = T on the grid, with a spline read-off."""

    grid: Grid
    U: np.ndarray
    T: float

    @cached_property
    def _spline(self) -> RectBivariateSpline:
        return RectBivariateSpline(self.grid.s, self.grid.v, self.U, kx=3, ky=3, s=0)

    def price(self, S0: float, v0: float) -> float:
        """Return the option value at today's stock price S0 and variance v0 (bicubic spline)."""
        if not (0.0 <= S0 <= self.grid.s[-1] and 0.0 <= v0 <= self.grid.v[-1]):
            raise ValueError(f"(S0, v0) = ({S0}, {v0}) lies outside the grid")
        return float(self._spline.ev(S0, v0))


def solve(
    params: HestonParams,
    option: OptionSpec,
    grid: Grid,
    n_steps: int,
    scheme: str = "douglas",
    theta_adi: float = 0.5,
    rannacher_half_steps: int = 2,
) -> SolveResult:
    """Price a European put under Heston by marching from the payoff to tau = T.

    Args:
        params: Heston parameters.
        option: The put. American options are not supported yet (they are added in G5).
        grid: The space grid.
        n_steps: Number of full time steps N, so dtau = T/N.
        scheme: Name of the ADI scheme in ``SCHEMES``.
        theta_adi: ADI weight for the main steps.
        rannacher_half_steps: Number of Douglas half-steps with theta_adi = 1 at the start.
            They replace the first rannacher_half_steps/2 full steps, which damps the
            error from the payoff kink. Must be even; 0 turns smoothing off.
    """
    if option.american:
        raise NotImplementedError("American constraint (Ikonen-Toivanen) is not implemented yet; see task G5")
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; choose from {sorted(SCHEMES)}")
    if rannacher_half_steps < 0 or rannacher_half_steps % 2:
        raise ValueError(f"rannacher_half_steps must be even and >= 0, got {rannacher_half_steps}")
    replaced = rannacher_half_steps // 2
    if n_steps < max(1, replaced):
        raise ValueError(f"need n_steps >= {max(1, replaced)}, got {n_steps}")

    step = SCHEMES[scheme]
    ops = build_heston_operators(grid, params)
    dtau = option.T / n_steps
    S, _ = grid.mesh()
    U = option.payoff(S)  # row 0 holds g(0) = K, the Dirichlet value at tau = 0

    for k in range(1, rannacher_half_steps + 1):  # start-up: Douglas half-steps with theta_adi = 1
        U = douglas_step(ops, U, 0.5 * dtau, 1.0, dirichlet_value(params, option, 0.5 * k * dtau))
    for k in range(replaced + 1, n_steps + 1):
        U = step(ops, U, dtau, theta_adi, dirichlet_value(params, option, k * dtau))

    return SolveResult(grid=grid, U=U, T=option.T)
