import numpy as np
import pytest

from adi_heston import HestonParams, OptionSpec
from adi_heston.fd import apply_three_point, second_derivative_weights
from adi_heston.grid import Grid, stock_grid
from adi_heston.reference import bs_gamma, bs_put, heston_put
from adi_heston.solver import solve

K, T = 100.0, 1.0


# Black-Scholes reduction: xi = 0 and v0 = theta, with theta exactly on a v-node, so that
# grid line evolves as a pure 1D Black-Scholes problem with sigma^2 = theta.
V_NODES = np.linspace(0.0, 0.2, 11)
THETA_INDEX = 2
BS_PARAMS = HestonParams(r=0.03, q=0.01, kappa=2.0, theta=float(V_NODES[THETA_INDEX]), xi=0.0, rho=0.0)
BS_SIGMA = np.sqrt(BS_PARAMS.theta)


def bs_reduction_value(m, n_steps, T=T, rannacher_half_steps=2):
    grid = Grid(stock_grid(K, 4 * K, m), V_NODES)
    U = solve(BS_PARAMS, OptionSpec(K, T), grid, n_steps, rannacher_half_steps=rannacher_half_steps).U
    return grid, U[:, THETA_INDEX]


def heston_params(rho):
    return HestonParams(r=0.025, q=0.0, kappa=1.5, theta=0.04, xi=0.3, rho=rho)


# ---- 1. Black-Scholes reduction ---------------------------------------------


def test_black_scholes_reduction_is_second_order():
    exact = bs_put(K, K, T, BS_PARAMS.r, BS_PARAMS.q, BS_SIGMA)
    errors = []
    for m, n_steps in ((100, 50), (200, 100)):
        grid, u = bs_reduction_value(m, n_steps)
        errors.append(abs(u[np.flatnonzero(grid.s == K)[0]] - exact))
    assert errors[1] < 2e-3
    assert errors[0] / errors[1] > 3.5  # doubling the grid and the steps cuts the error by about 4x


# ---- 2. Rannacher smoothing --------------------------------------------------


def test_rannacher_removes_gamma_oscillations():
    T_short = 0.25
    errors = {}
    for half_steps in (0, 2):
        grid, u = bs_reduction_value(200, 10, T=T_short, rannacher_half_steps=half_steps)
        gamma = apply_three_point(second_derivative_weights(grid.s), u)
        exact = bs_gamma(grid.s[1:-1], K, T_short, BS_PARAMS.r, BS_PARAMS.q, BS_SIGMA)
        near_K = (grid.s[1:-1] > 0.8 * K) & (grid.s[1:-1] < 1.2 * K)
        errors[half_steps] = np.abs(gamma[1:-1] - exact)[near_K].max()
        max_gamma = exact[near_K].max()
    assert errors[0] > max_gamma  # without smoothing the oscillation is larger than Gamma itself
    assert errors[2] < 0.05 * max_gamma
    assert errors[2] < errors[0] / 50


# ---- 3. Order in time for Heston ---------------------------------------------


def time_order(rho):
    grid = Grid.build(K, 8 * K, 5.0, 60, 30)
    prices = [solve(heston_params(rho), OptionSpec(K, T), grid, N).price(K, 0.04) for N in (40, 80, 160)]
    d = np.abs(np.diff(prices))
    return np.log2(d[0] / d[1])


def test_douglas_is_second_order_in_time_without_mixed_term():
    assert time_order(0.0) > 1.8


def test_douglas_is_first_order_in_time_with_mixed_term():
    assert 0.8 < time_order(-0.9) < 1.2


# ---- 4. Sanity for a Heston put with strong correlation ----------------------


@pytest.fixture(scope="module")
def heston_result():
    return solve(heston_params(-0.9), OptionSpec(K, T), Grid.build(K, 8 * K, 5.0, 100, 50), 100)


def test_price_within_no_arbitrage_bounds(heston_result):
    p = heston_params(-0.9)
    price = heston_result.price(K, 0.04)
    assert max(K * np.exp(-p.r * T) - K * np.exp(-p.q * T), 0.0) <= price <= K * np.exp(-p.r * T)


def test_price_falls_with_stock_and_rises_with_variance(heston_result):
    by_stock = [heston_result.price(S, 0.04) for S in (80.0, 100.0, 120.0)]
    by_variance = [heston_result.price(K, v0) for v0 in (0.02, 0.04, 0.08)]
    assert by_stock[0] > by_stock[1] > by_stock[2]
    assert by_variance[0] < by_variance[1] < by_variance[2]


# ---- 5. Agreement with the semi-analytic Heston price ------------------------------


def adi_error(rho, m, n, n_steps, scheme="douglas"):
    p = heston_params(rho)
    ref = heston_put(p, OptionSpec(K, T), K, 0.04)
    grid = Grid.build(K, 8 * K, 5.0, m, n)
    return abs(solve(p, OptionSpec(K, T), grid, n_steps, scheme=scheme).price(K, 0.04) - ref)


def test_adi_converges_to_reference_at_second_order_without_mixed_term():
    errors = [adi_error(0.0, m, n, m) for m, n in ((50, 25), (100, 50), (200, 100))]
    assert errors[-1] < 2e-3
    assert errors[0] / errors[1] > 3.5 and errors[1] / errors[2] > 3.5


def test_adi_converges_in_space_with_strong_correlation():
    # Many time steps make Douglas's first-order time error negligible, isolating the space error.
    errors = [adi_error(-0.9, m, n, 1000) for m, n in ((50, 25), (100, 50))]
    assert errors[1] < 1e-2
    assert errors[0] / errors[1] > 3.5


def test_adi_on_published_benchmark_where_feller_fails():
    p, v0, value = HestonParams(r=0.0, q=0.0, kappa=1.5768, theta=0.0398, xi=0.5751, rho=-0.5711), 0.0175, 5.785155450
    errors = [
        abs(solve(p, OptionSpec(K, T), Grid.build(K, 8 * K, 5.0, m, n), N).price(K, v0) - value)
        for m, n, N in ((100, 50, 200), (200, 100, 400))
    ]
    assert errors[0] < 2e-2
    assert errors[0] / errors[1] > 3.0


# ---- 6. Craig-Sneyd -----------------------------------------------------------


def test_craig_sneyd_equals_douglas_without_mixed_term():
    # With rho = 0 the mixed-term correction is exactly zero, so the second pass repeats the first.
    p, option, grid = heston_params(0.0), OptionSpec(K, T), Grid.build(K, 8 * K, 5.0, 40, 20)
    U_do = solve(p, option, grid, 20, scheme="douglas").U
    U_cs = solve(p, option, grid, 20, scheme="craig_sneyd").U
    np.testing.assert_array_equal(U_cs, U_do)


# ---- 7. Modified Craig-Sneyd ----------------------------------------------------


def test_modified_craig_sneyd_equals_craig_sneyd_at_theta_half():
    # The extra correction is scaled by (1/2 - theta_adi), so it vanishes at theta_adi = 1/2.
    p, option, grid = heston_params(-0.9), OptionSpec(K, T), Grid.build(K, 8 * K, 5.0, 40, 20)
    U_cs = solve(p, option, grid, 20, scheme="craig_sneyd", theta_adi=0.5).U
    U_mcs = solve(p, option, grid, 20, scheme="modified_craig_sneyd", theta_adi=0.5).U
    np.testing.assert_array_equal(U_mcs, U_cs)


def test_modified_craig_sneyd_defaults_to_theta_one_third():
    p, option, grid = heston_params(-0.9), OptionSpec(K, T), Grid.build(K, 8 * K, 5.0, 20, 10)
    U_default = solve(p, option, grid, 10, scheme="modified_craig_sneyd").U
    U_third = solve(p, option, grid, 10, scheme="modified_craig_sneyd", theta_adi=1.0 / 3.0).U
    np.testing.assert_array_equal(U_default, U_third)


# ---- 8. Hundsdorfer-Verwer ------------------------------------------------------


def test_hundsdorfer_verwer_defaults_to_theta_half_plus_sqrt3_over_6():
    p, option, grid = heston_params(-0.9), OptionSpec(K, T), Grid.build(K, 8 * K, 5.0, 20, 10)
    U_default = solve(p, option, grid, 10, scheme="hundsdorfer_verwer").U
    U_explicit = solve(p, option, grid, 10, scheme="hundsdorfer_verwer", theta_adi=0.5 + np.sqrt(3.0) / 6.0).U
    np.testing.assert_array_equal(U_default, U_explicit)


# ---- 9. Second-order schemes with the mixed term --------------------------------

SECOND_ORDER_SCHEMES = ["craig_sneyd", "modified_craig_sneyd", "hundsdorfer_verwer"]


@pytest.mark.parametrize("scheme", SECOND_ORDER_SCHEMES)
def test_second_order_in_time_with_mixed_term(scheme):
    # The time error at a single node is small enough to change sign, so measure it over
    # the whole grid against a run with many more steps.
    p, option, grid = heston_params(-0.9), OptionSpec(K, T), Grid.build(K, 8 * K, 5.0, 60, 30)
    U_fine = solve(p, option, grid, 640, scheme=scheme).U
    errors = [np.abs(solve(p, option, grid, N, scheme=scheme).U - U_fine).max() for N in (20, 40, 80)]
    assert errors[0] / errors[1] > 3.8 and errors[1] / errors[2] > 3.8


@pytest.mark.parametrize("scheme", SECOND_ORDER_SCHEMES)
def test_converges_to_reference_with_strong_correlation(scheme):
    errors = [adi_error(-0.9, m, n, m, scheme=scheme) for m, n in ((50, 25), (100, 50), (200, 100))]
    assert errors[-1] < 2e-3
    assert errors[0] / errors[1] > 3.5 and errors[1] / errors[2] > 3.5


# ---- 10. Interface checks ------------------------------------------------------


def small_problem():
    return heston_params(-0.5), Grid.build(K, 8 * K, 5.0, 20, 10)


def test_american_not_supported_yet():
    p, g = small_problem()
    with pytest.raises(NotImplementedError):
        solve(p, OptionSpec(K, T, american=True), g, 10)


@pytest.mark.parametrize("kwargs", [dict(scheme="nope"), dict(rannacher_half_steps=3), dict(rannacher_half_steps=-2)])
def test_bad_settings_rejected(kwargs):
    p, g = small_problem()
    with pytest.raises(ValueError):
        solve(p, OptionSpec(K, T), g, 10, **kwargs)


def test_read_off_outside_grid_rejected():
    p, g = small_problem()
    result = solve(p, OptionSpec(K, T), g, 10)
    with pytest.raises(ValueError):
        result.price(K, 6.0)


def test_read_off_matches_grid_value_at_a_node():
    p, g = small_problem()
    result = solve(p, OptionSpec(K, T), g, 10)
    i, j = int(np.flatnonzero(g.s == K)[0]), 3
    assert result.price(K, g.v[j]) == pytest.approx(result.U[i, j], rel=1e-10)
