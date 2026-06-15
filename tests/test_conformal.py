"""CQR correctness: the calibrated interval must attain nominal coverage on
fresh data drawn from the same distribution, regardless of the base quantiles."""

import numpy as np

from metis_benchmark.conformal import ConformalizedQuantile
from metis_benchmark.evaluation import empirical_coverage


def test_cqr_attains_nominal_coverage_from_miscalibrated_quantiles():
    # Truth: y ~ N(100, 15). Deliberately too-narrow base quantiles (±5) so the
    # raw interval under-covers; CQR must widen it to ~90% on held-out data.
    rng = np.random.default_rng(0)
    y_cal = rng.normal(100, 15, size=2000)
    y_test = rng.normal(100, 15, size=5000)
    lower_cal, upper_cal = np.full_like(y_cal, 95.0), np.full_like(y_cal, 105.0)
    lower_te, upper_te = np.full_like(y_test, 95.0), np.full_like(y_test, 105.0)

    cqr = ConformalizedQuantile(alpha=0.10).calibrate(lower_cal, upper_cal, y_cal)
    lo, hi = cqr.interval(lower_te, upper_te)
    cov = empirical_coverage(y_test, lo, hi)
    # Distribution-free guarantee -> empirical coverage close to nominal 90%.
    assert 0.88 <= cov <= 0.92


def test_cqr_can_tighten_overwide_intervals():
    # Base interval far too wide (±60 around a N(100,15) target): the score
    # quantile is negative, so CQR pulls the bounds inward.
    rng = np.random.default_rng(1)
    y = rng.normal(100, 15, size=3000)
    lower, upper = np.full_like(y, 40.0), np.full_like(y, 160.0)
    cqr = ConformalizedQuantile(alpha=0.10).calibrate(lower, upper, y)
    assert cqr.q_ < 0  # negative correction = tightening
    lo, hi = cqr.interval(lower, upper)
    assert hi[0] < 160.0 and lo[0] > 40.0


def test_interval_requires_calibration_first():
    cqr = ConformalizedQuantile()
    try:
        cqr.interval(np.array([1.0]), np.array([2.0]))
        raise AssertionError("expected RuntimeError before calibration")
    except RuntimeError:
        pass
