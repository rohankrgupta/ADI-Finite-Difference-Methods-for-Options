"""Reference prices that are independent of the PDE solver.

* Black-Scholes European put and Gamma (exact formulas).
* Heston European put (semi-analytic: one numerical integral per probability).

Heston's put has the same shape as Black-Scholes,

    P = K e^{-rT} (1 - P2) - S0 e^{-qT} (1 - P1),

where P1 and P2 are probabilities recovered from the characteristic function
phi(u) = E[exp(iu ln S_T)] by the Gil-Pelaez integral. phi is evaluated in the
"little Heston trap" form (Albrecher et al., 2007). The original form takes a
complex logarithm that can jump to the wrong branch at long maturities.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad
from scipy.stats import norm

from .params import HestonParams, OptionSpec


# ---- Black-Scholes ---------------------------------------------------------------


def _d1_d2(S, K, T, r, q, sigma):
    S = np.asarray(S, dtype=np.float64)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    return d1, d1 - sigma * np.sqrt(T)


def bs_put(S, K: float, T: float, r: float, q: float, sigma: float):
    """Return the Black-Scholes European put price."""
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    return K * np.exp(-r * T) * norm.cdf(-d2) - np.asarray(S) * np.exp(-q * T) * norm.cdf(-d1)


def bs_gamma(S, K: float, T: float, r: float, q: float, sigma: float):
    """Return the Black-Scholes Gamma (the same for puts and calls)."""
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    return np.exp(-q * T) * norm.pdf(d1) / (np.asarray(S) * sigma * np.sqrt(T))


# ---- Heston ---------------------------------------------------------------------------


def heston_char_func(u, params: HestonParams, T: float, S0: float, v0: float):
    """Return E[exp(iu ln S_T)] under Heston, in the little-trap form. ``u`` may be complex."""
    p = params
    iu = 1j * u
    b = p.kappa - p.rho * p.xi * iu
    d = np.sqrt(b**2 + p.xi**2 * (iu + u**2))  # principal root, Re(d) >= 0
    g = (b - d) / (b + d)
    e = np.exp(-d * T)
    C = (p.r - p.q) * iu * T + p.kappa * p.theta / p.xi**2 * ((b - d) * T - 2.0 * np.log((1.0 - g * e) / (1.0 - g)))
    D = (b - d) / p.xi**2 * (1.0 - e) / (1.0 - g * e)
    return np.exp(C + D * v0 + iu * np.log(S0))


def _deterministic_variance_put(params: HestonParams, option: OptionSpec, S0: float, v0: float) -> float:
    """xi = 0: the variance follows a known path, so the price is Black-Scholes with its time average."""
    p, T = params, option.T
    if p.kappa * T > 0.0:
        avg = p.theta + (v0 - p.theta) * (1.0 - np.exp(-p.kappa * T)) / (p.kappa * T)
    else:
        avg = v0
    return float(bs_put(S0, option.K, T, p.r, p.q, np.sqrt(avg)))


def heston_put(params: HestonParams, option: OptionSpec, S0: float, v0: float) -> float:
    """Return the Heston European put price at today's (S0, v0).

    Accuracy is about 1e-9: this is meant as the reference value for the ADI solver,
    not as a fast pricer.
    """
    if option.american:
        raise ValueError("the semi-analytic formula prices European options only")
    if params.xi == 0.0:
        return _deterministic_variance_put(params, option, S0, v0)

    K, T, p = option.K, option.T, params
    log_K = np.log(K)
    forward = S0 * np.exp((p.r - p.q) * T)  # E[S_T] = phi(-i)

    def integrand_P1(u):
        return (np.exp(-1j * u * log_K) * heston_char_func(u - 1j, p, T, S0, v0) / (1j * u * forward)).real

    def integrand_P2(u):
        return (np.exp(-1j * u * log_K) * heston_char_func(u, p, T, S0, v0) / (1j * u)).real

    opts = dict(epsabs=1e-12, epsrel=1e-10, limit=1000)
    P1 = 0.5 + quad(integrand_P1, 0.0, np.inf, **opts)[0] / np.pi
    P2 = 0.5 + quad(integrand_P2, 0.0, np.inf, **opts)[0] / np.pi
    return float(K * np.exp(-p.r * T) * (1.0 - P2) - S0 * np.exp(-p.q * T) * (1.0 - P1))
