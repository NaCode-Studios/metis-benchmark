"""Inverse-variance assist blend: weight limits and log-space correctness."""

import numpy as np
import pytest

from metis_benchmark.assist import (
    blend_log_space,
    inverse_variance_weight,
    ivw_blend,
    log_residual_sigma,
)


# --- inverse_variance_weight: the limits that define the estimator ----------


def test_weight_is_half_when_sigmas_equal():
    # Equal information -> equal say: w = 0.5 exactly, for any common sigma.
    assert inverse_variance_weight(0.7, 0.7) == pytest.approx(0.5)
    assert inverse_variance_weight(1e-3, 1e-3) == pytest.approx(0.5)


def test_weight_vanishes_when_expert_perfect():
    # sigma_e -> 0 means the expert is noiseless: all weight to the expert.
    assert inverse_variance_weight(0.8, 0.0) == 0.0
    assert inverse_variance_weight(0.8, 1e-9) == pytest.approx(0.0, abs=1e-15)


def test_weight_saturates_when_model_perfect():
    # The mirror limit: a noiseless model takes all the weight.
    assert inverse_variance_weight(0.0, 0.8) == 1.0


def test_weight_monotone_in_expert_noise():
    # A noisier expert must shift weight toward the model, monotonically.
    weights = [inverse_variance_weight(0.5, s) for s in (0.1, 0.3, 0.5, 1.0, 2.0)]
    assert all(a < b for a, b in zip(weights, weights[1:]))
    # And the exact textbook value: w = sigma_e^2 / (sigma_e^2 + sigma_m^2).
    assert inverse_variance_weight(0.5, 1.0) == pytest.approx(1.0 / 1.25)


def test_weight_degenerate_and_invalid():
    # Two perfect estimators is ill-posed; the symmetric convention is 0.5.
    assert inverse_variance_weight(0.0, 0.0) == 0.5
    # Negative "standard deviations" are caller bugs, not cases.
    with pytest.raises(ValueError):
        inverse_variance_weight(-0.1, 0.5)


# --- blend_log_space / ivw_blend: log-space algebra --------------------------


def test_blend_endpoints_return_inputs():
    model = np.array([100.0, 500.0])
    expert = np.array([80.0, 900.0])
    # w=1 -> pure model, w=0 -> pure expert (exactly, not approximately).
    assert blend_log_space(model, expert, 1.0) == pytest.approx(model)
    assert blend_log_space(model, expert, 0.0) == pytest.approx(expert)


def test_blend_half_is_geometric_mean():
    # w=0.5 in log space IS the geometric mean — the defining identity.
    model = np.array([100.0])
    expert = np.array([400.0])
    assert blend_log_space(model, expert, 0.5) == pytest.approx(np.sqrt(100.0 * 400.0))


def test_blend_matches_manual_log_combination():
    # General w: verify against the literal exp(w log m + (1-w) log e).
    rng = np.random.default_rng(0)
    model = rng.uniform(50, 5000, size=20)
    expert = rng.uniform(50, 5000, size=20)
    w = 0.3
    manual = np.exp(w * np.log(model) + (1 - w) * np.log(expert))
    assert blend_log_space(model, expert, w) == pytest.approx(manual)


def test_blend_rejects_bad_inputs():
    with pytest.raises(ValueError):
        blend_log_space(np.array([1.0]), np.array([1.0]), 1.5)  # w out of [0,1]
    with pytest.raises(ValueError):
        blend_log_space(np.array([-1.0]), np.array([1.0]), 0.5)  # non-positive


def test_ivw_blend_composes_weight_and_blend():
    # sigma_e = sigma_m -> w = 0.5 -> geometric mean; the composition must
    # equal calling the two pieces by hand.
    model = np.array([200.0, 1000.0])
    expert = np.array([50.0, 4000.0])
    out = ivw_blend(model, expert, sigma_model=0.6, sigma_expert=0.6)
    assert out == pytest.approx(np.sqrt(model * expert))
    # Perfect expert -> blend IS the expert (w -> 0).
    assert ivw_blend(model, expert, 0.6, 0.0) == pytest.approx(expert)


# --- log_residual_sigma: the measured input to the weight --------------------


def test_log_residual_sigma_known_value():
    # Estimates off by exactly x2 and /2 around truth: residuals are
    # ±log(2), whose ddof=1 SD is log(2)·sqrt(2·... ) — computed directly.
    actual = np.array([100.0, 100.0])
    estimate = np.array([50.0, 200.0])
    resid = np.log(actual) - np.log(estimate)  # [log2, -log2]
    assert log_residual_sigma(actual, estimate) == pytest.approx(np.std(resid, ddof=1))


def test_log_residual_sigma_ignores_constant_bias():
    # A constant multiplicative bias shifts the mean log residual, not its SD:
    # sigma must be identical whether the estimator is unbiased or 3x low.
    rng = np.random.default_rng(1)
    actual = rng.lognormal(5.0, 0.4, size=200)
    noise = rng.lognormal(0.0, 0.3, size=200)
    unbiased = actual * noise
    biased = actual * noise / 3.0
    s1 = log_residual_sigma(actual, unbiased)
    s2 = log_residual_sigma(actual, biased)
    assert s1 == pytest.approx(s2)


def test_log_residual_sigma_guards():
    with pytest.raises(ValueError):
        log_residual_sigma(np.array([1.0, 2.0]), np.array([1.0, -2.0]))
    with pytest.raises(ValueError):
        log_residual_sigma(np.array([1.0]), np.array([1.0]))
