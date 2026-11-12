import numpy as np
import pytest

from adi_heston import HestonParams, OptionSpec
from adi_heston.grid import Grid
from adi_heston.operators import build_heston_operators, dirichlet_value

K = 100.0
CASE = HestonParams(r=0.025, q=0.01, kappa=1.5, theta=0.04, xi=0.3, rho=-0.9)
HARD = HestonParams(r=0.03, q=0.0, kappa=0.6, theta=0.07, xi=0.9, rho=-0.9)  # Feller fails


def make_ops(params=CASE, m=50, n=25):
    return build_heston_operators(Grid.build(K=K, S_max=8 * K, V_max=5.0, m=m, n=n), params)


def heston_L(params, S, V):
    """Return u and the exact L u for u = sin(2s/K) cos(3v)."""
    p = params
    a, b = 2.0 / K, 3.0
    u = np.sin(a * S) * np.cos(b * V)
    u_s = a * np.cos(a * S) * np.cos(b * V)
    u_ss = -(a**2) * u
    u_v = -b * np.sin(a * S) * np.sin(b * V)
    u_vv = -(b**2) * u
    u_sv = -a * b * np.cos(a * S) * np.sin(b * V)
    Lu = (
        0.5 * S**2 * V * u_ss
        + p.rho * p.xi * S * V * u_sv
        + 0.5 * p.xi**2 * V * u_vv
        + (p.r - p.q) * S * u_s
        + p.kappa * (p.theta - V) * u_v
        - p.r * u
    )
    return u, Lu


# ---- simple exact checks ----------------------------------------------------


def test_constant_function():
    ops = make_ops()
    one = np.ones(ops.grid.shape)
    for A1_or_A2 in (ops.apply_A1(one), ops.apply_A2(one)):
        np.testing.assert_allclose(A1_or_A2[1:], -CASE.r / 2, rtol=0, atol=1e-9)  # large coefficients cancel
        assert not A1_or_A2[0].any()  # Dirichlet row
    assert np.abs(ops.apply_A0(one)).max() < 1e-12  # weights sum to zero up to roundoff


def test_linear_in_s_gives_minus_q_s():
    ops = make_ops()
    S, _ = ops.grid.mesh()
    Lu = ops.apply(S.copy())
    np.testing.assert_allclose(Lu[1:-1], (-CASE.q * S)[1:-1], rtol=1e-9, atol=1e-9)


def test_A1_matches_black_scholes_operator_on_quadratics_where_central():
    ops = make_ops()
    S, V = ops.grid.mesh()
    got = ops.apply_A1(S**2)
    expected = (V + 2 * (CASE.r - CASE.q) - CASE.r / 2) * S**2  # 1D BS operator with sigma^2 = v, minus r/2
    check = ~ops.upwind_s
    check[[0, -1], :] = False
    np.testing.assert_allclose(got[check], expected[check], rtol=1e-9)


# ---- second-order consistency -----------------------------------------------


def test_full_operator_is_second_order_where_central():
    errors = []
    for m, n in ((50, 25), (100, 50)):
        ops = make_ops(m=m, n=n)
        S, V = ops.grid.mesh()
        u, Lu = heston_L(CASE, S, V)
        box = (S >= 0.5 * K) & (S <= 1.5 * K) & (V >= 0.01) & (V <= 0.2)
        assert not (ops.upwind_s | ops.upwind_v)[box].any()  # the box uses central differences only
        errors.append(np.max(np.abs(ops.apply(u) - Lu)[box]))
    assert np.log2(errors[0] / errors[1]) > 1.8


# ---- boundary edges ---------------------------------------------------------


def test_dirichlet_row_is_zero_in_every_operator():
    ops = make_ops()
    assert not ops.a1[:, 0].any() and not ops.a2[:, 0].any() and not ops.a0_coef[0].any()


def test_mixed_coefficient_is_zero_on_all_edges():
    c = make_ops().a0_coef
    assert not (c[0].any() or c[-1].any() or c[:, 0].any() or c[:, -1].any())


def test_s_max_mirror_row():
    ops = make_ops()
    s, v = ops.grid.s, ops.grid.v
    h = s[-1] - s[-2]
    np.testing.assert_allclose(ops.a1[0, -1], s[-1] ** 2 * v / h**2)
    np.testing.assert_allclose(ops.a1[1, -1], -(s[-1] ** 2) * v / h**2 - CASE.r / 2)
    assert not ops.a1[2, -1].any()


def test_v_max_mirror_column():
    ops = make_ops()
    v = ops.grid.v
    h = v[-1] - v[-2]
    np.testing.assert_allclose(ops.a2[0, 1:, -1], CASE.xi**2 * v[-1] / h**2)
    np.testing.assert_allclose(ops.a2[1, 1:, -1], -CASE.xi**2 * v[-1] / h**2 - CASE.r / 2)
    assert not ops.a2[2, :, -1].any()


def test_v_zero_column_uses_forward_difference():
    ops = make_ops()
    h0 = ops.grid.v[1]
    kt = CASE.kappa * CASE.theta
    np.testing.assert_allclose(ops.a2[2, 1:, 0], kt / h0)
    np.testing.assert_allclose(ops.a2[1, 1:, 0], -kt / h0 - CASE.r / 2)
    assert not ops.a2[0, :, 0].any()


def test_v_zero_column_upwinds_s_convection():
    ops = make_ops()
    assert ops.upwind_s[1:-1, 0].all()  # no s-diffusion at v = 0, so convection must be upwinded
    assert not ops.a1[0, 1:-1, 0].any()  # r - q > 0: forward difference, so no lower coefficient


def test_corner_s_max_v_zero_is_pure_decay_in_s():
    ops = make_ops()
    assert ops.a1[0, -1, 0] == 0.0 and ops.a1[1, -1, 0] == pytest.approx(-CASE.r / 2)


# ---- solver safety ------------------------------------------------------------


@pytest.mark.parametrize("params", [CASE, HARD], ids=["feller-holds", "feller-fails"])
def test_operators_have_nonnegative_offdiagonals_and_row_sums_minus_half_r(params):
    ops = make_ops(params)
    for a in (ops.a1, ops.a2):
        assert (a[0] >= 0).all() and (a[2] >= 0).all()
        sums = a.sum(axis=0)[1:]  # skip the Dirichlet row
        np.testing.assert_allclose(sums, -params.r / 2, atol=1e-9)


@pytest.mark.parametrize("params", [CASE, HARD], ids=["feller-holds", "feller-fails"])
@pytest.mark.parametrize("c", [1e-3, 1.0, 1e3])
def test_implicit_systems_are_strictly_diagonally_dominant(params, c):
    ops = make_ops(params)
    for lo, di, up in (ops.a1, ops.a2):
        assert (np.abs(1.0 - c * di) > np.abs(c * lo) + np.abs(c * up)).all()


@pytest.mark.parametrize("which", ["A1", "A2"])
def test_solve_inverts_I_minus_cA(which):
    ops = make_ops()
    rhs = np.random.default_rng(0).normal(size=ops.grid.shape)
    c = 0.05
    solve, apply = (ops.solve_A1, ops.apply_A1) if which == "A1" else (ops.solve_A2, ops.apply_A2)
    Y = solve(c, rhs)
    np.testing.assert_allclose(Y - c * apply(Y), rhs, rtol=1e-10, atol=1e-10)


def test_dirichlet_value():
    euro, amer = OptionSpec(K=100.0, T=1.0), OptionSpec(K=100.0, T=1.0, american=True)
    assert dirichlet_value(CASE, euro, 0.5) == pytest.approx(100.0 * np.exp(-0.025 * 0.5))
    assert dirichlet_value(CASE, amer, 0.5) == 100.0
