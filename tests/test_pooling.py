"""Hierarchical pooling correctness: shrinkage behaviour and the empirical-Bayes
between-dataset variance estimate."""

import numpy as np

from metis_benchmark.track_a.pooling import PooledPowerLaw, _dersimonian_laird, _ols_loglog


def _powerlaw(rng, n, a, b, noise):
    size = rng.uniform(10, 1000, size=n)
    effort = np.exp(a) * size**b * rng.lognormal(0.0, noise, size=n)
    return size, effort


def test_ols_recovers_slope():
    rng = np.random.default_rng(0)
    size, effort = _powerlaw(rng, 500, a=1.0, b=0.85, noise=0.05)
    intercept, slope, var = _ols_loglog(size, effort)
    assert abs(slope - 0.85) < 0.02
    assert var > 0


def test_dersimonian_laird_zero_between_variance_when_homogeneous():
    # All "datasets" share the same slope with only sampling noise -> tau2 ~ 0
    # and the global slope equals the common value.
    slopes = np.array([0.90, 0.91, 0.89, 0.905])
    variances = np.array([1e-3, 1e-3, 1e-3, 1e-3])
    b0, tau2 = _dersimonian_laird(slopes, variances)
    assert abs(b0 - 0.90) < 0.02
    assert tau2 < 1e-3


def test_dersimonian_laird_detects_real_heterogeneity():
    # Genuinely different slopes with tight within-variance -> tau2 > 0.
    slopes = np.array([0.6, 0.9, 1.2, 1.5])
    variances = np.array([1e-3, 1e-3, 1e-3, 1e-3])
    _, tau2 = _dersimonian_laird(slopes, variances)
    assert tau2 > 0.05


def test_pooling_shrinks_noisy_dataset_more_than_precise_one():
    rng = np.random.default_rng(1)
    # Global built from datasets with genuinely different slopes (spread around
    # ~0.9) so the between-dataset variance tau2 is > 0 and shrinkage is partial.
    # (When all contributors agree, tau2 ~ 0 forces full shrinkage — correct
    # empirical-Bayes behaviour, but it would make this test non-discriminating.)
    contribs = [
        _powerlaw(rng, 300, a=1.0, b=b, noise=0.05) for b in (0.7, 0.9, 1.1)
    ]
    pooled = PooledPowerLaw.fit_global(contribs)
    assert abs(pooled.b_0 - 0.9) < 0.15
    assert pooled.tau2 > 0  # real heterogeneity detected

    # A precise local dataset (many points, low noise) barely shrinks; a noisy
    # local dataset (few points, high noise) shrinks a large fraction of the way
    # toward the global. We measure the shrinkage FRACTION relative to each
    # dataset's own raw local slope (the noisy local slope is itself random, so
    # comparing absolute distance to the global would be meaningless).
    size_precise, effort_precise = _powerlaw(rng, 400, a=2.0, b=0.5, noise=0.05)
    size_noisy, effort_noisy = _powerlaw(rng, 10, a=2.0, b=0.5, noise=0.6)

    def pooled_slope(size_tr, effort_tr):
        s = np.array([10.0, 1000.0])
        base = pooled.baseline_log(size_tr, effort_tr, s)
        return (base[1] - base[0]) / (np.log(1000.0) - np.log(10.0))

    def shrink_fraction(size_tr, effort_tr):
        _, b_raw, _ = _ols_loglog(size_tr, effort_tr)
        b_pool = pooled_slope(size_tr, effort_tr)
        return abs(b_pool - b_raw) / abs(pooled.b_0 - b_raw)

    # The noisy dataset moves a larger fraction toward the global than the
    # precise one — the defining behaviour of partial pooling.
    assert shrink_fraction(size_noisy, effort_noisy) > shrink_fraction(size_precise, effort_precise)
