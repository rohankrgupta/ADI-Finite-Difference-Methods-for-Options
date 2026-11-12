import numpy as np
import pytest

from adi_heston.fd import (
    apply_three_point,
    first_derivative_weights,
    forward_first_derivative_weights,
    second_derivative_weights,
)
from adi_heston.grid import stock_grid


def sinh_grid(m):
    """Return a smooth clustered grid on [0, 4] (clustered at 1.5), like the real s-grid."""
    return stock_grid(K=1.5, S_max=4.0, m=m, c=0.5)


def jumpy_grid(m):
    """Return a grid on [0, 4] whose spacing alternates h, 2h, h, 2h, ... (m must be even)."""
    h = np.tile([1.0, 2.0], m // 2)
    return np.concatenate([[0.0], np.cumsum(h)]) * (4.0 / h.sum())


def max_interior_error(x, w, f, exact):
    return np.max(np.abs(apply_three_point(w, f(x))[1:-1] - exact(x)[1:-1]))


def observed_order(grid, weights, f, exact, m=40):
    e1 = max_interior_error(grid(m), weights(grid(m)), f, exact)
    e2 = max_interior_error(grid(2 * m), weights(grid(2 * m)), f, exact)
    return np.log2(e1 / e2)


# ---- exactness on quadratics -----------------------------------------------


def test_central_weights_exact_for_quadratics_on_nonuniform_grid():
    x = stock_grid(K=10.0, S_max=40.0, m=20)
    u = 3.0 - 2.0 * x + 0.25 * x**2
    np.testing.assert_allclose(apply_three_point(first_derivative_weights(x), u)[1:-1], (-2.0 + 0.5 * x)[1:-1], rtol=1e-10, atol=1e-10)
    np.testing.assert_allclose(apply_three_point(second_derivative_weights(x), u)[1:-1], 0.5, rtol=1e-9)


def test_forward_weights_exact_for_quadratics():
    x = np.array([0.0, 0.1, 0.35, 1.0])
    u = 3.0 - 2.0 * x + 0.25 * x**2
    assert forward_first_derivative_weights(x) @ u[:3] == pytest.approx(-2.0)


def test_weights_reduce_to_textbook_formulas_on_uniform_grid():
    h = 0.5
    x = np.arange(5) * h
    np.testing.assert_allclose(first_derivative_weights(x)[2], [-1 / (2 * h), 0.0, 1 / (2 * h)])
    np.testing.assert_allclose(second_derivative_weights(x)[2], [1 / h**2, -2 / h**2, 1 / h**2])
    np.testing.assert_allclose(forward_first_derivative_weights(x), [-3 / (2 * h), 2 / h, -1 / (2 * h)])


# ---- convergence order ------------------------------------------------------


def test_first_derivative_is_second_order_on_sinh_grid():
    assert observed_order(sinh_grid, first_derivative_weights, np.sin, np.cos) > 1.8


def test_second_derivative_is_second_order_on_sinh_grid():
    assert observed_order(sinh_grid, second_derivative_weights, np.sin, lambda x: -np.sin(x)) > 1.8


def test_second_derivative_drops_to_first_order_on_jumpy_grid():
    """A sudden change in spacing costs one order: this is why the grids use a smooth stretch."""
    p = observed_order(jumpy_grid, second_derivative_weights, np.sin, lambda x: -np.sin(x))
    assert 0.7 < p < 1.3


def test_first_derivative_stays_second_order_on_jumpy_grid():
    assert observed_order(jumpy_grid, first_derivative_weights, np.sin, np.cos) > 1.8


def test_forward_weights_are_second_order():
    errors = []
    for m in (40, 80):
        x = sinh_grid(m)
        errors.append(abs(forward_first_derivative_weights(x) @ np.exp(x[:3]) - 1.0))
    assert np.log2(errors[0] / errors[1]) > 1.8


# ---- shapes and edges -------------------------------------------------------


def test_edge_rows_are_zero_and_shape_is_n_plus_one_by_three():
    x = stock_grid(K=10.0, S_max=40.0, m=12)
    for w in (first_derivative_weights(x), second_derivative_weights(x)):
        assert w.shape == (13, 3)
        assert not w[0].any() and not w[-1].any()


def test_apply_three_point_works_along_axis_0_of_a_2d_array():
    x = stock_grid(K=10.0, S_max=40.0, m=12)
    U = np.outer(x**2, [1.0, 2.0, 3.0])  # column j is (j+1) * s^2
    D = apply_three_point(first_derivative_weights(x), U)
    np.testing.assert_allclose(D[1:-1], np.outer(2 * x, [1.0, 2.0, 3.0])[1:-1], rtol=1e-10, atol=1e-10)


@pytest.mark.parametrize("x", [[0.0, 1.0], [0.0, 2.0, 1.0], [[0.0, 1.0, 2.0]]])
def test_bad_grids_rejected(x):
    with pytest.raises(ValueError):
        first_derivative_weights(np.array(x))
