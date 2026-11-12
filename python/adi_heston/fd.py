"""Finite-difference weights on non-uniform 1D grids.

Every weight set comes from fitting a parabola through three neighbouring nodes and
differentiating it, so the weights are exact for quadratics.

Three-point weights are stored as arrays of shape (N+1, 3). Row i holds the
coefficients for (u_{i-1}, u_i, u_{i+1}). Rows 0 and N are left at zero; the
operator assembly decides what happens at the edges (Dirichlet, Neumann, or a
one-sided formula).

Accuracy (h- = x_i - x_{i-1}, h+ = x_{i+1} - x_i):
  * First derivative: error is about -(h- * h+ / 6) u''', second order on any grid.
  * Second derivative: error is about ((h+ - h-) / 3) u''' + O(h^2). This is second
    order only when neighbouring spacings change gradually, which is why the grids
    use a smooth sinh stretch.
"""

from __future__ import annotations

import numpy as np


def _spacings(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1 or x.size < 3:
        raise ValueError("need a 1D grid with at least 3 nodes")
    h = np.diff(x)
    if np.any(h <= 0.0):
        raise ValueError("grid must be strictly increasing")
    return h[:-1], h[1:]  # h_minus, h_plus at interior nodes 1..N-1


def first_derivative_weights(x: np.ndarray) -> np.ndarray:
    """Return central weights for u'(x_i) at the interior nodes, shape (N+1, 3)."""
    hm, hp = _spacings(x)
    w = np.zeros((hm.size + 2, 3))
    w[1:-1, 0] = -hp / (hm * (hm + hp))
    w[1:-1, 1] = (hp - hm) / (hm * hp)
    w[1:-1, 2] = hm / (hp * (hm + hp))
    return w


def second_derivative_weights(x: np.ndarray) -> np.ndarray:
    """Return central weights for u''(x_i) at the interior nodes, shape (N+1, 3)."""
    hm, hp = _spacings(x)
    w = np.zeros((hm.size + 2, 3))
    w[1:-1, 0] = 2.0 / (hm * (hm + hp))
    w[1:-1, 1] = -2.0 / (hm * hp)
    w[1:-1, 2] = 2.0 / (hp * (hm + hp))
    return w


def forward_first_derivative_weights(x: np.ndarray) -> np.ndarray:
    """Return one-sided weights for u'(x_0) on (u_0, u_1, u_2), shape (3,).

    This formula is second order. We use it at v = 0, where it points into the domain
    (the upwind direction).
    """
    h1, h2 = _spacings(x)
    h1, h2 = h1[0], h2[0]
    return np.array(
        [
            -(2.0 * h1 + h2) / (h1 * (h1 + h2)),
            (h1 + h2) / (h1 * h2),
            -h1 / (h2 * (h1 + h2)),
        ]
    )


def apply_three_point(w: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Apply three-point weights along axis 0 of ``u``. Rows 0 and N of the result are zero.

    ``u`` may have extra trailing axes; for example, a 2D solution with weights in s.
    """
    u = np.asarray(u, dtype=np.float64)
    if u.shape[0] != w.shape[0]:
        raise ValueError(f"weights have {w.shape[0]} rows but u has {u.shape[0]}")
    col = (slice(1, -1),) + (None,) * (u.ndim - 1)
    out = np.zeros_like(u)
    out[1:-1] = w[col + (0,)] * u[:-2] + w[col + (1,)] * u[1:-1] + w[col + (2,)] * u[2:]
    return out


def mixed_derivative(w_s: np.ndarray, w_v: np.ndarray, U: np.ndarray) -> np.ndarray:
    """Approximate u_sv at the interior nodes as the s-derivative of the v-derivative.

    Args:
        w_s: First-derivative weights in s, shape (m+1, 3).
        w_v: First-derivative weights in v, shape (n+1, 3).
        U: Grid values, shape (m+1, n+1).

    Returns:
        An array of shape (m+1, n+1). It is zero on all four edges, where the mixed term
        is either not needed or vanishes.

    The equivalent 9-point stencil puts weight ``w_s[i, k+1] * w_v[j, l+1]`` on
    ``U[i+k, j+l]`` for k, l in {-1, 0, 1}. On an even grid only the four corners are
    nonzero. On a non-uniform grid all nine are, because the centre weight of a first
    derivative is (h+ - h-) / (h- * h+).
    """
    U = np.asarray(U, dtype=np.float64)
    if U.ndim != 2:
        raise ValueError(f"U must be 2D, got shape {U.shape}")
    dU_dv = apply_three_point(w_v, U.T).T  # along axis 1; columns 0 and n are zero
    return apply_three_point(w_s, dU_dv)  # along axis 0; rows 0 and m are zero
