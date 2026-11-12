import numpy as np
import pytest

from adi_heston.grid import Grid, stock_grid, variance_grid


def max_ratio_deviation(x):
    """Return the largest |h_{k+1}/h_k - 1| over neighbouring spacings h of the nodes x."""
    h = np.diff(x)
    return np.max(np.abs(h[1:] / h[:-1] - 1.0))


# ---- stock grid -------------------------------------------------------------


def test_stock_grid_endpoints_and_strike_are_exact_nodes():
    s = stock_grid(K=100.0, S_max=800.0, m=50)
    assert s.size == 51
    assert s[0] == 0.0 and s[-1] == 800.0
    assert np.count_nonzero(s == 100.0) == 1


@pytest.mark.parametrize("m", [2, 3, 10, 51, 200])
def test_stock_grid_is_strictly_increasing_and_keeps_strike(m):
    s = stock_grid(K=10.0, S_max=20.0, m=m)
    assert np.all(np.diff(s) > 0.0)
    assert 10.0 in s


def test_stock_grid_is_finest_at_the_strike():
    s = stock_grid(K=100.0, S_max=800.0, m=50)
    h = np.diff(s)
    k = int(np.flatnonzero(s == 100.0)[0])
    assert np.argmin(h) in (k - 1, k)
    assert h.max() / h.min() > 10.0  # clearly clustered, not nearly uniform


def test_stock_grid_spacing_ratios_tend_to_one():
    coarse = max_ratio_deviation(stock_grid(100.0, 800.0, 50))
    fine = max_ratio_deviation(stock_grid(100.0, 800.0, 200))
    assert fine < coarse / 2.5  # O(1/m): 4x the points gives about 4x smaller


def test_large_c_gives_nearly_uniform_grid():
    s = stock_grid(K=100.0, S_max=400.0, m=40, c=1e6)
    h = np.diff(s)
    assert h.max() / h.min() < 1.01


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(K=100.0, S_max=100.0, m=10),
        dict(K=0.0, S_max=100.0, m=10),
        dict(K=50.0, S_max=100.0, m=1),
        dict(K=50.0, S_max=100.0, m=10, c=0.0),
    ],
)
def test_stock_grid_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        stock_grid(**kwargs)


# ---- variance grid ----------------------------------------------------------


def test_variance_grid_endpoints_are_exact():
    v = variance_grid(V_max=5.0, n=25)
    assert v.size == 26
    assert v[0] == 0.0 and v[-1] == 5.0


def test_variance_grid_is_finest_at_zero():
    v = variance_grid(V_max=5.0, n=25)
    h = np.diff(v)
    assert np.all(np.diff(h) > 0.0)  # spacing grows monotonically away from v = 0
    assert h[0] < 0.01


def test_variance_grid_spacing_ratios_tend_to_one():
    coarse = max_ratio_deviation(variance_grid(5.0, 25))
    fine = max_ratio_deviation(variance_grid(5.0, 100))
    assert fine < coarse / 2.5


@pytest.mark.parametrize("kwargs", [dict(V_max=0.0, n=10), dict(V_max=1.0, n=0), dict(V_max=1.0, n=10, d=-1.0)])
def test_variance_grid_rejects_bad_input(kwargs):
    with pytest.raises(ValueError):
        variance_grid(**kwargs)


# ---- Grid -------------------------------------------------------------------


def test_grid_build_shape_and_mesh():
    g = Grid.build(K=100.0, S_max=800.0, V_max=5.0, m=50, n=25)
    assert (g.m, g.n, g.shape) == (50, 25, (51, 26))
    S, V = g.mesh()
    assert S.shape == V.shape == g.shape
    np.testing.assert_array_equal(S[:, 7], g.s)
    np.testing.assert_array_equal(V[3, :], g.v)


def test_grid_nodes_are_read_only():
    g = Grid.build(K=100.0, S_max=800.0, V_max=5.0, m=10, n=10)
    with pytest.raises(ValueError):
        g.s[1] = 1.0


@pytest.mark.parametrize("s", [[0.0, 2.0, 1.0], [1.0, 2.0, 3.0], [0.0]])
def test_grid_rejects_bad_nodes(s):
    with pytest.raises(ValueError):
        Grid(s=np.array(s), v=np.array([0.0, 1.0]))
