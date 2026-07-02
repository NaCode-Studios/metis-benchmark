"""Intercept random-effects correctness (post-G0 addition to pooling.py).

The claim under test: DerSimonian-Laird applied to per-dataset mean
log-productivity recovers a KNOWN between-dataset spread tau_a from synthetic
data, and the degenerate corners (2 datasets, zero within-variance) stay
finite and honest instead of crashing or fabricating spread.
"""

import numpy as np
import pytest

from metis_benchmark.track_a.pooling import (
    InterceptPrior,
    fit_intercept_prior,
    log_productivity_intercept,
)


def _synthetic_contributions(rng, k, n, a0, tau, noise):
    """k datasets whose true intercepts are draws from N(a0, tau^2).

    effort = exp(a_d) * size * lognormal(0, noise): elasticity exactly 1, so
    mean log(effort/size) estimates a_d with se = noise/sqrt(n) — the model
    fit_intercept_prior assumes, letting the test isolate the DL estimator.
    """
    contribs = []
    for _ in range(k):
        a_d = rng.normal(a0, tau)  # this dataset's true productivity level
        size = rng.uniform(10, 1000, size=n)
        effort = np.exp(a_d) * size * rng.lognormal(0.0, noise, size=n)
        contribs.append((size, effort))
    return contribs


def test_intercept_estimate_and_se():
    # Deterministic productivity (no noise): intercept must equal log(e/s)
    # exactly and the SE must be 0 (every project agrees perfectly).
    size = np.array([10.0, 100.0, 1000.0])
    effort = size * np.exp(2.0)
    a, se, n = log_productivity_intercept(size, effort)
    assert a == pytest.approx(2.0)
    assert se == pytest.approx(0.0)
    assert n == 3


def test_intercept_rejects_bad_inputs():
    # Non-positive values make the log target undefined -> hard error, and a
    # single project cannot support a standard error.
    with pytest.raises(ValueError):
        log_productivity_intercept(np.array([1.0, -2.0]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        log_productivity_intercept(np.array([1.0, 2.0]), np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        log_productivity_intercept(np.array([5.0]), np.array([50.0]))


def test_recovers_known_tau():
    # Many datasets, tight within-noise: DL must land near the true tau and
    # the Q-profile CI must cover it. k=40 keeps the estimator's own sampling
    # error (~tau/sqrt(2(k-1)) ~ 0.06) well inside the assertion tolerance.
    rng = np.random.default_rng(7)
    true_tau = 0.5
    prior = fit_intercept_prior(_synthetic_contributions(rng, 40, 200, a0=2.0, tau=true_tau, noise=0.4))
    assert isinstance(prior, InterceptPrior)
    assert abs(prior.tau_a - true_tau) < 0.15
    assert abs(prior.a_0 - 2.0) < 0.25
    lo, hi = prior.tau_a_ci
    assert lo < true_tau < hi


def test_homogeneous_datasets_give_near_zero_tau():
    # tau=0 world: every dataset shares one intercept, spread is pure within
    # noise. DL must attribute (almost) nothing to the between component and
    # the CI lower bound must sit at 0 (tau=0 not rejectable).
    rng = np.random.default_rng(11)
    prior = fit_intercept_prior(_synthetic_contributions(rng, 20, 300, a0=1.5, tau=0.0, noise=0.5))
    assert prior.tau_a < 0.05
    assert prior.tau_a_ci[0] == 0.0


def test_two_datasets_degenerate_but_finite():
    # The k=2 corner: DL still has 1 degree of freedom for Q, so the point
    # estimate exists, but the Q-profile CI must be honestly wide (the upper
    # bound far above the point) — 2 datasets cannot pin down a variance.
    rng = np.random.default_rng(3)
    contribs = _synthetic_contributions(rng, 2, 100, a0=2.0, tau=0.8, noise=0.3)
    prior = fit_intercept_prior(contribs)
    assert np.isfinite(prior.tau_a)
    lo, hi = prior.tau_a_ci
    assert np.isfinite(lo) and np.isfinite(hi)
    assert hi > prior.tau_a  # uncertainty dominates at k=2


def test_zero_within_variance_degenerates_to_sample_sd():
    # Exact productivities (se=0): all observed spread is between-dataset, so
    # DL's within-correction vanishes and tau must equal the ddof=1 sample SD
    # of the intercepts — the analytic limit the variance floor preserves.
    intercepts = [1.0, 2.0, 3.0, 4.0]
    contribs = []
    for a_d in intercepts:
        size = np.linspace(10, 500, 50)
        contribs.append((size, size * np.exp(a_d)))
    prior = fit_intercept_prior(contribs)
    assert prior.tau_a == pytest.approx(np.std(intercepts, ddof=1), rel=1e-6)
    assert prior.std_errors == pytest.approx((0.0, 0.0, 0.0, 0.0), abs=1e-12)


def test_single_dataset_refused():
    # One dataset has no between spread to measure; returning tau=0 would be
    # fabrication, so the function must refuse.
    size = np.linspace(10, 100, 20)
    with pytest.raises(ValueError):
        fit_intercept_prior([(size, size * 2.0)])
