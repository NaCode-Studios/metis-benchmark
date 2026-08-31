"""CQR correctness: the calibrated interval must attain nominal coverage on
fresh data drawn from the same distribution, regardless of the base quantiles."""

import numpy as np
import pytest

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


def test_over_tightening_collapses_to_a_point_instead_of_inverting():
    # A base band far wider than the calibration residuals drives q strongly
    # negative; tightening by more than half the width would put the lower bound
    # above the upper one, which is not an interval at all — evaluation's
    # empirical_coverage rejects it. The guard collapses those rows to their
    # midpoint so the pipeline gets a defined (and deservedly useless) band
    # rather than a crash. Surfaced by the metis-core verdict simulation at n=60.
    rng = np.random.default_rng(0)
    y_cal = rng.normal(100.0, 0.1, size=200)
    # Calibration band: absurdly wide around the truth -> q << 0.
    cqr = ConformalizedQuantile(alpha=0.10).calibrate(
        y_cal - 500.0, y_cal + 500.0, y_cal
    )
    assert cqr.q_ < 0
    # Apply it to a band far narrower than the calibration one: 2|q| > width.
    lo, hi = cqr.interval(np.array([99.0]), np.array([101.0]))
    assert lo[0] <= hi[0]
    assert lo[0] == pytest.approx(hi[0])
    assert lo[0] == pytest.approx(100.0)


def test_guard_leaves_well_behaved_intervals_untouched():
    # The guard must be invisible in the normal case: same numbers as the plain
    # shift, so nothing produced before it existed moves.
    rng = np.random.default_rng(1)
    y_cal = rng.normal(50.0, 5.0, size=200)
    cqr = ConformalizedQuantile(alpha=0.10).calibrate(y_cal - 6.0, y_cal + 6.0, y_cal)
    lo_in, hi_in = np.array([40.0, 60.0]), np.array([60.0, 80.0])
    lo, hi = cqr.interval(lo_in, hi_in)
    assert np.allclose(lo, lo_in - cqr.q_)
    assert np.allclose(hi, hi_in + cqr.q_)
