"""Model and contract parameters shared by every solver."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HestonParams:
    """Parameters of the Heston model, all per year.

    Attributes:
        r: Risk-free interest rate.
        q: Dividend yield.
        kappa: Speed at which the variance is pulled back toward ``theta``.
        theta: Long-run variance. This is a variance, not a volatility:
            ``theta = 0.04`` means 20% volatility.
        xi: Volatility of the variance ("vol of vol"). ``xi = 0`` is allowed;
            with ``v0 = theta`` it reduces the model to Black-Scholes.
        rho: Correlation between price and variance moves, in (-1, 1).
    """

    r: float
    q: float
    kappa: float
    theta: float
    xi: float
    rho: float

    def __post_init__(self) -> None:
        if not -1.0 < self.rho < 1.0:
            raise ValueError(f"rho must lie in (-1, 1), got {self.rho}")
        for name in ("kappa", "theta", "xi"):
            if getattr(self, name) < 0.0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")

    @property
    def feller_ratio(self) -> float:
        """Return 2*kappa*theta / xi**2; the Feller condition holds when this is >= 1."""
        if self.xi == 0.0:
            return float("inf")
        return 2.0 * self.kappa * self.theta / self.xi**2

    @property
    def satisfies_feller(self) -> bool:
        """True when 2*kappa*theta >= xi**2, so the variance never reaches zero."""
        return 2.0 * self.kappa * self.theta >= self.xi**2


@dataclass(frozen=True)
class OptionSpec:
    """A put option, the only contract type in this project.

    Attributes:
        K: Strike price.
        T: Maturity in years.
        american: True if the option can be exercised at any time up to ``T``.
    """

    K: float
    T: float
    american: bool = False

    def __post_init__(self) -> None:
        if self.K <= 0.0:
            raise ValueError(f"K must be positive, got {self.K}")
        if self.T <= 0.0:
            raise ValueError(f"T must be positive, got {self.T}")

    def payoff(self, s: np.ndarray | float) -> np.ndarray:
        """Return the put payoff g(s) = max(K - s, 0)."""
        return np.maximum(self.K - np.asarray(s, dtype=np.float64), 0.0)
