import numpy as np
import pytest

from adi_heston import HestonParams, OptionSpec
from adi_heston.reference import bs_gamma, bs_put, heston_put

K = 100.0

# Fang & Oosterlee (COS paper) Heston test case; parameters from a Bakshi-Cao-Chen fit.
# The Feller condition fails (2*kappa*theta = 0.126 < xi^2 = 0.331). With r = q = 0 and S0 = K, put = call.
BENCHMARK = HestonParams(r=0.0, q=0.0, kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711)
BENCHMARK_V0 = 0.0175
BENCHMARK_VALUE = 5.785155450


# ---- Black-Scholes helpers -------------------------------------------------------


def test_bs_put_call_parity_and_limits():
    S, T, r, q, sigma = np.array([80.0, 100.0, 120.0]), 1.0, 0.03, 0.01, 0.2
    put = bs_put(S, K, T, r, q, sigma)
    assert np.all(put >= np.maximum(K * np.exp(-r * T) - S * np.exp(-q * T), 0.0))
    assert np.all(np.diff(put) < 0.0)


def test_bs_gamma_matches_finite_difference_of_price():
    h, args = 1e-3, (K, 0.5, 0.03, 0.01, 0.25)
    fd = (bs_put(100.0 + h, *args) - 2 * bs_put(100.0, *args) + bs_put(100.0 - h, *args)) / h**2
    assert bs_gamma(100.0, *args) == pytest.approx(fd, rel=1e-5)


# ---- A1. Black-Scholes limit ------------------------------------------------------


def test_heston_tends_to_black_scholes_as_xi_vanishes():
    def gap(xi):
        p = HestonParams(r=0.03, q=0.01, kappa=2.0, theta=0.04, xi=xi, rho=0.0)
        return heston_put(p, OptionSpec(K, 1.0), 100.0, 0.04) - float(bs_put(100.0, K, 1.0, 0.03, 0.01, 0.2))

    assert abs(gap(1e-4)) < 1e-7
    assert gap(1e-2) / gap(1e-3) == pytest.approx(100.0, rel=0.05)  # the gap shrinks like xi^2


def test_xi_zero_uses_time_averaged_variance_and_is_continuous():
    p0 = HestonParams(r=0.03, q=0.01, kappa=2.0, theta=0.04, xi=0.0, rho=-0.5)
    p_small = HestonParams(r=0.03, q=0.01, kappa=2.0, theta=0.04, xi=1e-4, rho=-0.5)
    opt = OptionSpec(K, 1.0)
    assert heston_put(p0, opt, 100.0, 0.09) == pytest.approx(heston_put(p_small, opt, 100.0, 0.09), abs=5e-5)


# ---- A2. Published benchmark ------------------------------------------------------


def test_matches_published_benchmark_to_eight_digits():
    value = heston_put(BENCHMARK, OptionSpec(K, 1.0), 100.0, BENCHMARK_V0)
    assert value == pytest.approx(BENCHMARK_VALUE, abs=5e-8)


# ---- A3. No-arbitrage bounds and monotonicity ----------------------------------


def test_bounds_and_monotonicity():
    p = HestonParams(r=0.025, q=0.0, kappa=1.5, theta=0.04, xi=0.3, rho=-0.9)
    opt = OptionSpec(K, 1.0)
    by_stock = [heston_put(p, opt, S, 0.04) for S in (80.0, 100.0, 120.0)]
    by_variance = [heston_put(p, opt, K, v0) for v0 in (0.02, 0.04, 0.08)]
    assert by_stock[0] > by_stock[1] > by_stock[2] > 0.0
    assert by_variance[0] < by_variance[1] < by_variance[2]
    assert by_stock[0] >= K * np.exp(-p.r) - 80.0
    assert max(by_stock) <= K * np.exp(-p.r)


# ---- A4. No branch jumps at long maturities ---------------------------------------


def test_long_maturities_are_smooth():
    maturities = np.linspace(1.0, 30.0, 59)
    values = np.array([heston_put(BENCHMARK, OptionSpec(K, T), 100.0, BENCHMARK_V0) for T in maturities])
    assert np.all(np.isfinite(values))
    assert np.all(np.diff(values) > 0.0)  # with r = q = 0 the put gains value with maturity
    assert np.abs(np.diff(values, 2)).max() < 0.25  # no jumps


def test_american_rejected():
    with pytest.raises(ValueError):
        heston_put(BENCHMARK, OptionSpec(K, 1.0, american=True), 100.0, 0.04)
