"""Non-uniform grids in the stock price s and the variance v.

Both grids come from evenly spaced points in a helper variable, mapped through a
smooth sinh stretch. The smooth map makes neighbouring spacings change gradually
(their ratio tends to 1 as the grid is refined). That is what keeps central
differences second-order accurate on a non-uniform grid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def stock_grid(K: float, S_max: float, m: int, c: float | None = None) -> np.ndarray:
    """Return m+1 stock-price nodes on [0, S_max], clustered around the strike K.

    The nodes are s_i = K + c*sinh(xi_i). The helper variable xi is evenly spaced
    on each side of xi = 0, which makes K exactly a node. The left and right
    xi-steps differ slightly (their ratio is 1 + O(1/m)), so that both s = 0 and
    s = S_max are hit exactly.

    Args:
        K: Strike. Must satisfy 0 < K < S_max.
        S_max: Right end of the truncated domain.
        m: Number of intervals (at least 2).
        c: Clustering width. A smaller c crowds more points near K. The default is K/5.
    """
    if not 0.0 < K < S_max:
        raise ValueError(f"need 0 < K < S_max, got K={K}, S_max={S_max}")
    if m < 2:
        raise ValueError(f"need at least 2 intervals, got m={m}")
    c = K / 5.0 if c is None else float(c)
    if c <= 0.0:
        raise ValueError(f"c must be positive, got {c}")

    xi_left = np.arcsinh(-K / c)  # maps to s = 0
    xi_right = np.arcsinh((S_max - K) / c)  # maps to s = S_max
    m_left = int(round(m * (-xi_left) / (xi_right - xi_left)))
    m_left = min(max(m_left, 1), m - 1)

    xi = np.concatenate(
        [
            np.linspace(xi_left, 0.0, m_left + 1),
            np.linspace(0.0, xi_right, m - m_left + 1)[1:],
        ]
    )
    s = K + c * np.sinh(xi)
    s[0], s[m_left], s[-1] = 0.0, K, S_max  # remove rounding at the special nodes
    return s


def variance_grid(V_max: float, n: int, d: float | None = None) -> np.ndarray:
    """Return n+1 variance nodes on [0, V_max], clustered near v = 0.

    The nodes are v_j = d*sinh(eta_j), with eta evenly spaced.

    Args:
        V_max: Top of the truncated domain.
        n: Number of intervals (at least 1).
        d: Clustering width. A smaller d crowds more points near v = 0. The default is V_max/500.
    """
    if V_max <= 0.0:
        raise ValueError(f"V_max must be positive, got {V_max}")
    if n < 1:
        raise ValueError(f"need at least 1 interval, got n={n}")
    d = V_max / 500.0 if d is None else float(d)
    if d <= 0.0:
        raise ValueError(f"d must be positive, got {d}")

    eta = np.linspace(0.0, np.arcsinh(V_max / d), n + 1)
    v = d * np.sinh(eta)
    v[0], v[-1] = 0.0, V_max
    return v


@dataclass(frozen=True)
class Grid:
    """A tensor-product grid. Solution arrays have shape (m+1, n+1), indexed U[i, j] = u(s_i, v_j).

    The node arrays are stored read-only so that they cannot be changed by accident.
    """

    s: np.ndarray
    v: np.ndarray

    def __post_init__(self) -> None:
        for name in ("s", "v"):
            x = np.array(getattr(self, name), dtype=np.float64)
            if x.ndim != 1 or x.size < 2:
                raise ValueError(f"{name} must be a 1D array with at least 2 nodes")
            if x[0] != 0.0:
                raise ValueError(f"{name} must start at 0, got {x[0]}")
            if np.any(np.diff(x) <= 0.0):
                raise ValueError(f"{name} must be strictly increasing")
            x.setflags(write=False)
            object.__setattr__(self, name, x)

    @classmethod
    def build(
        cls,
        K: float,
        S_max: float,
        V_max: float,
        m: int,
        n: int,
        c: float | None = None,
        d: float | None = None,
    ) -> Grid:
        """Build the standard clustered grid: s clustered at K, and v clustered at 0."""
        return cls(stock_grid(K, S_max, m, c), variance_grid(V_max, n, d))

    @property
    def m(self) -> int:
        """Number of intervals in s."""
        return self.s.size - 1

    @property
    def n(self) -> int:
        """Number of intervals in v."""
        return self.v.size - 1

    @property
    def shape(self) -> tuple[int, int]:
        """Shape of a solution array, (m+1, n+1)."""
        return (self.s.size, self.v.size)

    def mesh(self) -> tuple[np.ndarray, np.ndarray]:
        """Return the arrays S, V of shape (m+1, n+1) with S[i, j] = s_i and V[i, j] = v_j."""
        return np.meshgrid(self.s, self.v, indexing="ij")
