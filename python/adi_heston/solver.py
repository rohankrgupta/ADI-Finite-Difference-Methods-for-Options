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


def _implicit_corrections(
    ops: HestonOperators, Y0: np.ndarray, c: float, A1X: np.ndarray, A2X: np.ndarray, boundary_new: float
) -> np.ndarray:
    """Run the two implicit stages shared by every ADI scheme and return Y2.

    Solves (I - c*A1) Y1 = Y0 - c*A1X along s, then (I - c*A2) Y2 = Y1 - c*A2X along v,
    where A1X and A2X are A1 and A2 applied to the scheme's base vector. Row 0 of the
    s-system is set to the Dirichlet value at the new time; its coefficients are zero,
    so the v-stage leaves it unchanged.
    """
    R1 = Y0 - c * A1X
    R1[0] = boundary_new
    Y1 = ops.solve_A1(c, R1)
    return ops.solve_A2(c, Y1 - c * A2X)


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
    Y0 = U + dtau * (A0U + A1U + A2U)  # (a) predict
    return _implicit_corrections(ops, Y0, theta_adi * dtau, A1U, A2U, boundary_new)  # (b) in s, (c) in v


def craig_sneyd_step(
    ops: HestonOperators, U: np.ndarray, dtau: float, theta_adi: float, boundary_new: float
) -> np.ndarray:
    """Advance U by one Craig-Sneyd ADI step of size dtau.

    A Douglas step gives Y2. The mixed term is then corrected with the average of A0 at
    the old and new values, and the two implicit stages are repeated:

        Y0~ = Y0 + dtau/2 * (A0 Y2 - A0 U),  followed by stages (b) and (c) from Y0~.

    The cost is 4 stencil sweeps plus 4 batched tridiagonal solves. The scheme is 2nd
    order in time for theta_adi = 1/2, also with the mixed term. With A0 = 0 it reduces
    to Douglas. See Haentjens & in 't Hout (2015), eq. (4.2).
    """
    A0U, A1U, A2U = ops.apply_A0(U), ops.apply_A1(U), ops.apply_A2(U)
    c = theta_adi * dtau
    Y0 = U + dtau * (A0U + A1U + A2U)
    Y2 = _implicit_corrections(ops, Y0, c, A1U, A2U, boundary_new)
    Y0_tilde = Y0 + 0.5 * dtau * (ops.apply_A0(Y2) - A0U)
    return _implicit_corrections(ops, Y0_tilde, c, A1U, A2U, boundary_new)


def modified_craig_sneyd_step(
    ops: HestonOperators, U: np.ndarray, dtau: float, theta_adi: float, boundary_new: float
) -> np.ndarray:
    """Advance U by one modified Craig-Sneyd ADI step of size dtau.

    As Craig-Sneyd, but the correction before the second pass also includes all of A:

        Y0~ = Y0 + theta_adi*dtau * A0 (Y2 - U) + (1/2 - theta_adi)*dtau * A (Y2 - U).

    The extra term makes the scheme 2nd order in time for every theta_adi, and it vanishes
    for theta_adi = 1/2, where the scheme reduces to Craig-Sneyd. The cost is 6 stencil
    sweeps plus 4 batched tridiagonal solves. See Haentjens & in 't Hout (2015), eq. (4.3).
    """
    A0U, A1U, A2U = ops.apply_A0(U), ops.apply_A1(U), ops.apply_A2(U)
    c = theta_adi * dtau
    Y0 = U + dtau * (A0U + A1U + A2U)
    Y2 = _implicit_corrections(ops, Y0, c, A1U, A2U, boundary_new)
    A0Y2 = ops.apply_A0(Y2)
    AY2_minus_AU = (A0Y2 + ops.apply_A1(Y2) + ops.apply_A2(Y2)) - (A0U + A1U + A2U)
    Y0_tilde = Y0 + c * (A0Y2 - A0U) + (0.5 - theta_adi) * dtau * AY2_minus_AU
    return _implicit_corrections(ops, Y0_tilde, c, A1U, A2U, boundary_new)


def hundsdorfer_verwer_step(
    ops: HestonOperators, U: np.ndarray, dtau: float, theta_adi: float, boundary_new: float
) -> np.ndarray:
    """Advance U by one Hundsdorfer-Verwer ADI step of size dtau.

    A Douglas step gives Y2. All of A is then corrected with its average over the old and
    new values, and the implicit stages are repeated relative to Y2 instead of U:

        Y0~ = Y0 + dtau/2 * A (Y2 - U),  then (I - c*A_j) Y_j~ = Y_{j-1}~ - c*A_j Y2.

    The scheme is 2nd order in time for every theta_adi. The cost is 6 stencil sweeps plus
    4 batched tridiagonal solves. See Haentjens & in 't Hout (2015), eq. (4.4).
    """
    A0U, A1U, A2U = ops.apply_A0(U), ops.apply_A1(U), ops.apply_A2(U)
    c = theta_adi * dtau
    Y0 = U + dtau * (A0U + A1U + A2U)
    Y2 = _implicit_corrections(ops, Y0, c, A1U, A2U, boundary_new)
    A1Y2, A2Y2 = ops.apply_A1(Y2), ops.apply_A2(Y2)
    Y0_tilde = Y0 + 0.5 * dtau * ((ops.apply_A0(Y2) + A1Y2 + A2Y2) - (A0U + A1U + A2U))
    return _implicit_corrections(ops, Y0_tilde, c, A1Y2, A2Y2, boundary_new)


SCHEMES: dict[str, StepFunction] = {
    "douglas": douglas_step,
    "craig_sneyd": craig_sneyd_step,
    "modified_craig_sneyd": modified_craig_sneyd_step,
    "hundsdorfer_verwer": hundsdorfer_verwer_step,
}

# theta_adi used by solve() when none is given (Haentjens & in 't Hout, Section 4).
DEFAULT_THETA: dict[str, float] = {
    "douglas": 0.5,
    "craig_sneyd": 0.5,
    "modified_craig_sneyd": 1.0 / 3.0,
    "hundsdorfer_verwer": 0.5 + np.sqrt(3.0) / 6.0,
}


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
    theta_adi: float | None = None,
    rannacher_half_steps: int = 2,
) -> SolveResult:
    """Price a European put under Heston by marching from the payoff to tau = T.

    Args:
        params: Heston parameters.
        option: The put. American options are not supported yet (they are added in G5).
        grid: The space grid.
        n_steps: Number of full time steps N, so dtau = T/N.
        scheme: Name of the ADI scheme in ``SCHEMES``.
        theta_adi: ADI weight for the main steps. None uses the scheme's ``DEFAULT_THETA``.
        rannacher_half_steps: Number of Douglas half-steps with theta_adi = 1 at the start.
            They replace the first rannacher_half_steps/2 full steps, which damps the
            error from the payoff kink. Must be even; 0 turns smoothing off.
    """
    if option.american:
        raise NotImplementedError("American constraint (Ikonen-Toivanen) is not implemented yet; see task G5")
    if scheme not in SCHEMES:
        raise ValueError(f"unknown scheme {scheme!r}; choose from {sorted(SCHEMES)}")
    if theta_adi is None:
        theta_adi = DEFAULT_THETA[scheme]
    schedule =time_schedule(params, option, n_steps, theta_adi, rannacher_half_steps)

    step = SCHEMES[scheme]
    ops = build_heston_operators(grid, params)
    S, _ = grid.mesh()
    U = option.payoff(S)  # row 0 holds g(0) = K, the Dirichlet value at tau = 0

    for row, (dtau, theta, boundary) in enumerate(schedule):
        step_fn = douglas_step if row < rannacher_half_steps else step  # start-up always uses Douglas
        U = step_fn(ops, U, dtau, theta, boundary)

    return SolveResult(grid=grid, U=U, T=option.T)


def time_schedule(
    params: HestonParams,
    option: OptionSpec,
    n_steps: int,
    theta_adi: float = 0.5,
    rannacher_half_steps: int = 2,
) -> np.ndarray:
    """Return the time steps as an array of rows (dtau, theta_adi, boundary value at the step's end).

    The first ``rannacher_half_steps`` rows are the start-up: half-steps with theta_adi = 1,
    replacing the first rannacher_half_steps/2 full steps. The remaining rows are full
    steps with the given theta_adi. ``solve()`` runs exactly these rows, and the C++/CUDA
    fixtures store them as schedule.bin.
    """
    if rannacher_half_steps < 0 or rannacher_half_steps % 2:
        raise ValueError(f"rannacher_half_steps must be even and >= 0, got {rannacher_half_steps}")
    replaced = rannacher_half_steps // 2
    if n_steps < max(1, replaced):
        raise ValueError(f"need n_steps >= {max(1, replaced)}, got {n_steps}")
    dtau = option.T / n_steps
    rows = [(0.5 * dtau, 1.0, dirichlet_value(params, option, 0.5 * k * dtau)) for k in range(1, rannacher_half_steps + 1)]
    rows += [(dtau, theta_adi, dirichlet_value(params, option, k * dtau)) for k in range(replaced + 1, n_steps + 1)]
    return np.array(rows, dtype=np.float64)
