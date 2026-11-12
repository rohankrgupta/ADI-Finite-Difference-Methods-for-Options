"""Batched tridiagonal solver (Thomas algorithm)."""

from __future__ import annotations

import numpy as np


def solve_tridiagonal(
    lower: np.ndarray,
    diag: np.ndarray,
    upper: np.ndarray,
    rhs: np.ndarray,
    axis: int = 0,
) -> np.ndarray:
    """Solve many independent tridiagonal systems at once.

    Along ``axis``, each line k of the system reads

        lower[k] * x[k-1] + diag[k] * x[k] + upper[k] * x[k+1] = rhs[k],

    with ``lower[0]`` and ``upper[-1]`` ignored. Every other axis indexes an
    independent system: the loop runs along ``axis``, and all lines are handled
    together in each iteration.

    There is no pivoting, so the systems should be diagonally dominant. The ADI
    systems (I - c*A_j) always are (see ``operators.py``).

    Args:
        lower, diag, upper: Coefficients that broadcast to ``rhs.shape``.
        rhs: Right-hand side(s).
        axis: The axis along which the systems couple.

    Returns:
        The solution, with the same shape as ``rhs``.
    """
    rhs = np.asarray(rhs, dtype=np.float64)
    a, b, c = (np.moveaxis(np.broadcast_to(x, rhs.shape), axis, 0) for x in (lower, diag, upper))
    d = np.moveaxis(rhs, axis, 0)
    N = d.shape[0]

    c_star = np.empty_like(d)
    d_star = np.empty_like(d)
    c_star[0] = c[0] / b[0]
    d_star[0] = d[0] / b[0]
    for k in range(1, N):
        denom = b[k] - a[k] * c_star[k - 1]
        c_star[k] = c[k] / denom
        d_star[k] = (d[k] - a[k] * d_star[k - 1]) / denom

    x = np.empty_like(d)
    x[-1] = d_star[-1]
    for k in range(N - 2, -1, -1):
        x[k] = d_star[k] - c_star[k] * x[k + 1]
    return np.moveaxis(x, 0, axis)
