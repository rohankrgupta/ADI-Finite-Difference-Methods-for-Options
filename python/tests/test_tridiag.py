import numpy as np
import pytest
from scipy.linalg import solve_banded

from adi_heston.tridiag import solve_tridiagonal


def random_dominant_system(rng, shape):
    lower = rng.uniform(0.0, 1.0, shape)
    upper = rng.uniform(0.0, 1.0, shape)
    diag = 1.0 + lower + upper + rng.uniform(0.0, 1.0, shape)  # strictly diagonally dominant
    rhs = rng.normal(size=shape)
    return -lower, diag, -upper, rhs


def reference_line(lower, diag, upper, rhs):
    ab = np.zeros((3, rhs.size))
    ab[0, 1:] = upper[:-1]
    ab[1] = diag
    ab[2, :-1] = lower[1:]
    return solve_banded((1, 1), ab, rhs)


@pytest.mark.parametrize("N", [1, 2, 3, 17])
def test_single_system_matches_scipy(N):
    lo, di, up, rhs = random_dominant_system(np.random.default_rng(N), (N,))
    np.testing.assert_allclose(solve_tridiagonal(lo, di, up, rhs), reference_line(lo, di, up, rhs), rtol=1e-12)


@pytest.mark.parametrize("axis", [0, 1])
def test_batched_systems_match_scipy_line_by_line(axis):
    lo, di, up, rhs = random_dominant_system(np.random.default_rng(7), (9, 6))
    x = solve_tridiagonal(lo, di, up, rhs, axis=axis)
    for k in range(rhs.shape[1 - axis]):
        line = (slice(None), k) if axis == 0 else (k, slice(None))
        np.testing.assert_allclose(x[line], reference_line(lo[line], di[line], up[line], rhs[line]), rtol=1e-12)


def test_coefficients_broadcast_against_rhs():
    rng = np.random.default_rng(3)
    lo, di, up, _ = random_dominant_system(rng, (8,))
    rhs = rng.normal(size=(8, 4))
    x = solve_tridiagonal(lo[:, None], di[:, None], up[:, None], rhs, axis=0)
    for k in range(4):
        np.testing.assert_allclose(x[:, k], reference_line(lo, di, up, rhs[:, k]), rtol=1e-12)
