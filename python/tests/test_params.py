import dataclasses

import numpy as np
import pytest

from adi_heston import HestonParams, OptionSpec


def make_params(**overrides):
    base = dict(r=0.025, q=0.0, kappa=1.5, theta=0.04, xi=0.3, rho=-0.9)
    base.update(overrides)
    return HestonParams(**base)


def test_payoff_matches_definition():
    opt = OptionSpec(K=100.0, T=1.0)
    assert opt.payoff(80.0) == pytest.approx(20.0)
    assert opt.payoff(120.0) == pytest.approx(0.0)
    np.testing.assert_allclose(opt.payoff([0.0, 50.0, 100.0, 150.0]), [100.0, 50.0, 0.0, 0.0])


def test_payoff_is_float64():
    assert OptionSpec(K=10.0, T=0.25).payoff(np.arange(5)).dtype == np.float64


def test_feller_condition():
    holds = make_params(kappa=1.5, theta=0.04, xi=0.3)  # 0.12 >= 0.09
    fails = make_params(kappa=0.5, theta=0.04, xi=0.5)  # 0.04 <  0.25
    assert holds.satisfies_feller and holds.feller_ratio == pytest.approx(0.12 / 0.09)
    assert not fails.satisfies_feller and fails.feller_ratio < 1.0


def test_xi_zero_is_allowed_for_black_scholes_reduction():
    p = make_params(xi=0.0)
    assert p.satisfies_feller and p.feller_ratio == float("inf")


@pytest.mark.parametrize("bad", [dict(rho=1.0), dict(rho=-1.0), dict(kappa=-1.0), dict(theta=-0.01), dict(xi=-0.1)])
def test_invalid_heston_params_rejected(bad):
    with pytest.raises(ValueError):
        make_params(**bad)


@pytest.mark.parametrize("bad", [dict(K=0.0, T=1.0), dict(K=100.0, T=0.0)])
def test_invalid_option_rejected(bad):
    with pytest.raises(ValueError):
        OptionSpec(**bad)


def test_params_are_immutable():
    with pytest.raises(dataclasses.FrozenInstanceError):
        make_params().rho = 0.0
