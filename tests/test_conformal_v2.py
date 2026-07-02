"""Post-G0 conformal extensions: log-space CQR and Mondrian grouping.

Three families of guarantees are locked here:
1. Regression — the default (space='raw') reproduces the G0 behavior EXACTLY,
   value-for-value, so `make reproduce-g0` still reproduces the frozen verdict.
2. Log-space properties — the correction is a uniform *multiplicative* shift in
   raw units (equivalently, an additive shift in log units), so the interval
   width in log space is invariant across project scales.
3. Mondrian — per-group corrections where the group is large enough, explicit
   marginal fallback where it is not.
"""

import numpy as np
import pytest

from metis_benchmark.conformal import (
    ConformalizedQuantile,
    MondrianConformalizedQuantile,
    assign_terciles,
    log_size_tercile_edges,
)
from metis_benchmark.evaluation import empirical_coverage


# ---------- 1. Regression: space='raw' is the G0 behavior, exactly ----------


def test_default_space_is_raw():
    # The G0 code had no `space`; the default must not change the contract.
    assert ConformalizedQuantile().space == "raw"


def test_raw_q_matches_hand_computed_value_exactly():
    # Hand-computable case: scores = max(90-y, y-110) = [-10, 10, 10, -5, -5].
    # n=5, level = min(1, ceil(6*0.9)/5) = min(1, 1.2) = 1.0 -> q = max = 10.0.
    lower = np.full(5, 90.0)
    upper = np.full(5, 110.0)
    y = np.array([100.0, 120.0, 80.0, 105.0, 95.0])
    cqr = ConformalizedQuantile(alpha=0.10).calibrate(lower, upper, y)
    assert cqr.q_ == 10.0
    lo, hi = cqr.interval(np.array([90.0]), np.array([110.0]))
    # The raw shift is exactly additive: [90-10, 110+10].
    assert lo[0] == 80.0 and hi[0] == 120.0


def test_raw_regression_against_g0_values_on_fixed_data():
    # Same construction as the pre-existing suite test (seed 0, N(100,15),
    # +-5 base band): the calibrated q must be bit-identical between the
    # legacy constructor call and the explicit space='raw' call.
    rng = np.random.default_rng(0)
    y_cal = rng.normal(100, 15, size=2000)
    lower_cal, upper_cal = np.full_like(y_cal, 95.0), np.full_like(y_cal, 105.0)
    legacy = ConformalizedQuantile(alpha=0.10).calibrate(lower_cal, upper_cal, y_cal)
    explicit = ConformalizedQuantile(alpha=0.10, space="raw").calibrate(
        lower_cal, upper_cal, y_cal
    )
    # Exact float equality: the raw code path must be the SAME computation.
    assert legacy.q_ == explicit.q_
    lo_a, hi_a = legacy.interval(np.array([95.0, 50.0]), np.array([105.0, 300.0]))
    lo_b, hi_b = explicit.interval(np.array([95.0, 50.0]), np.array([105.0, 300.0]))
    assert np.array_equal(lo_a, lo_b) and np.array_equal(hi_a, hi_b)


def test_unknown_space_rejected_at_construction():
    with pytest.raises(ValueError):
        ConformalizedQuantile(space="sqrt")
    with pytest.raises(ValueError):
        MondrianConformalizedQuantile(space="sqrt")


# ---------- 2. Log-space properties ----------


def test_log_space_shift_is_uniformly_multiplicative_in_hours():
    # Whatever q the calibration finds, the back-transformed interval must be
    # [lower*exp(-q), upper*exp(q)] — the SAME ratio at every project scale.
    rng = np.random.default_rng(2)
    y = np.exp(rng.normal(7, 1, size=500))  # log-normal effort
    lower, upper = 0.8 * y, 1.2 * y
    cqr = ConformalizedQuantile(alpha=0.10, space="log").calibrate(lower, upper, y)
    # Test points spanning three orders of magnitude.
    lo_in = np.array([10.0, 1_000.0, 100_000.0])
    hi_in = np.array([20.0, 2_000.0, 200_000.0])
    lo, hi = cqr.interval(lo_in, hi_in)
    # Multiplicative shift: output/input ratio equals exp(-q)/exp(q) everywhere.
    assert np.allclose(lo / lo_in, np.exp(-cqr.q_))
    assert np.allclose(hi / hi_in, np.exp(cqr.q_))
    # Lower bound stays positive by construction — no negative-hours intervals.
    assert np.all(lo > 0)


def test_log_space_width_invariant_in_log_units():
    # The log-width log(hi) - log(lo) must grow by exactly 2q for EVERY point,
    # regardless of scale: that is the equalization the raw shift cannot do.
    rng = np.random.default_rng(3)
    y = np.exp(rng.normal(6, 1, size=300))
    lower, upper = 0.9 * y, 1.1 * y
    cqr = ConformalizedQuantile(alpha=0.10, space="log").calibrate(lower, upper, y)
    lo_in = np.array([5.0, 500.0, 50_000.0])
    hi_in = np.array([9.0, 900.0, 90_000.0])
    lo, hi = cqr.interval(lo_in, hi_in)
    widening = (np.log(hi) - np.log(lo)) - (np.log(hi_in) - np.log(lo_in))
    assert np.allclose(widening, 2 * cqr.q_)


def test_log_space_attains_nominal_coverage_on_multiplicative_noise():
    # Log-normal target with multiplicative noise: exactly the regime where a
    # raw additive shift fails per-scale but a log shift is the right geometry.
    rng = np.random.default_rng(4)
    mu = rng.uniform(4, 9, size=6000)  # wide range of project scales
    y = np.exp(mu + rng.normal(0, 0.5, size=6000))
    lower, upper = np.exp(mu - 0.3), np.exp(mu + 0.3)  # too narrow on purpose
    cqr = ConformalizedQuantile(alpha=0.10, space="log").calibrate(
        lower[:3000], upper[:3000], y[:3000]
    )
    lo, hi = cqr.interval(lower[3000:], upper[3000:])
    cov = empirical_coverage(y[3000:], lo, hi)
    # Distribution-free guarantee holds in log space too (~90% +- sampling noise).
    assert 0.88 <= cov <= 0.92


def test_log_space_rejects_nonpositive_inputs():
    cqr = ConformalizedQuantile(space="log")
    with pytest.raises(ValueError):
        cqr.calibrate(np.array([-1.0, 1.0]), np.array([2.0, 2.0]), np.array([1.0, 1.0]))


# ---------- 3. Mondrian grouping and the explicit fallback ----------


def test_tercile_edges_and_assignment_from_training_sizes():
    # 9 training sizes spanning three decades -> edges split them 3/3/3.
    size_train = np.array([1, 2, 3, 10, 20, 30, 100, 200, 300], dtype=float)
    edges = log_size_tercile_edges(size_train)
    groups = assign_terciles(size_train, edges)
    # Terciles of the training distribution itself must be balanced.
    assert [int((groups == g).sum()) for g in (0, 1, 2)] == [3, 3, 3]
    # New points fall into the group their size dictates.
    assert assign_terciles(np.array([1.5]), edges)[0] == 0
    assert assign_terciles(np.array([15.0]), edges)[0] == 1
    assert assign_terciles(np.array([1_000.0]), edges)[0] == 2


def test_tercile_helpers_reject_nonpositive_sizes():
    with pytest.raises(ValueError):
        log_size_tercile_edges(np.array([0.0, 1.0]))
    with pytest.raises(ValueError):
        assign_terciles(np.array([-3.0]), np.array([0.0, 1.0]))


def test_mondrian_uses_per_group_q_when_groups_are_large_enough():
    # Two groups with visibly different error scales: group 0 needs a small
    # correction, group 1 a large one. With enough calibration points each
    # group must get ITS OWN q, not a shared compromise.
    rng = np.random.default_rng(5)
    n = 200
    y0 = 100 + rng.normal(0, 1, n)  # tight residuals
    y1 = 100 + rng.normal(0, 25, n)  # wide residuals
    lower = np.full(2 * n, 99.0)
    upper = np.full(2 * n, 101.0)
    y = np.concatenate([y0, y1])
    groups = np.concatenate([np.zeros(n, dtype=int), np.ones(n, dtype=int)])
    m = MondrianConformalizedQuantile(alpha=0.10, min_group_cal=15).calibrate(
        lower, upper, y, groups
    )
    assert m.fallback_groups_ == set()  # both groups self-calibrated
    # The wide-residual group demands a strictly larger correction.
    assert m.q_by_group_[1] > m.q_by_group_[0]
    # And the interval applies each point's own group correction.
    lo, hi = m.interval(np.array([99.0, 99.0]), np.array([101.0, 101.0]), np.array([0, 1]))
    assert hi[1] - lo[1] > hi[0] - lo[0]


def test_mondrian_falls_back_to_marginal_below_threshold():
    # Group 1 has only 5 calibration points (< 15): its per-group quantile has
    # no finite-sample meaning, so it must inherit the MARGINAL q and be
    # reported in fallback_groups_.
    rng = np.random.default_rng(6)
    y_big = 100 + rng.normal(0, 5, 100)
    y_small = 100 + rng.normal(0, 50, 5)
    lower = np.full(105, 98.0)
    upper = np.full(105, 102.0)
    y = np.concatenate([y_big, y_small])
    groups = np.array([0] * 100 + [1] * 5)
    m = MondrianConformalizedQuantile(alpha=0.10, min_group_cal=15).calibrate(
        lower, upper, y, groups
    )
    assert m.fallback_groups_ == {1}
    assert m.q_by_group_[1] == m.q_marginal_
    # Group 0 is large enough to keep its own calibration.
    assert 0 not in m.fallback_groups_


def test_mondrian_unseen_group_uses_marginal_q():
    # A test point whose group never appeared in calibration (size drift) must
    # get the marginal correction, never a KeyError.
    y = 100 + np.arange(30, dtype=float) - 15.0
    lower = np.full(30, 95.0)
    upper = np.full(30, 105.0)
    m = MondrianConformalizedQuantile(alpha=0.10, min_group_cal=15).calibrate(
        lower, upper, y, np.zeros(30, dtype=int)
    )
    lo, hi = m.interval(np.array([95.0]), np.array([105.0]), np.array([2]))
    assert lo[0] == 95.0 - m.q_marginal_ and hi[0] == 105.0 + m.q_marginal_


def test_mondrian_requires_calibration_first():
    m = MondrianConformalizedQuantile()
    with pytest.raises(RuntimeError):
        m.interval(np.array([1.0]), np.array([2.0]), np.array([0]))


def test_mondrian_log_space_matches_scalar_log_cqr_per_group():
    # Cross-check: with one single group holding all points, Mondrian in log
    # space must reduce exactly to the scalar log-space CQR.
    rng = np.random.default_rng(7)
    y = np.exp(rng.normal(6, 0.8, 100))
    lower, upper = 0.7 * y, 1.3 * y
    scalar = ConformalizedQuantile(alpha=0.10, space="log").calibrate(lower, upper, y)
    mondrian = MondrianConformalizedQuantile(alpha=0.10, space="log").calibrate(
        lower, upper, y, np.zeros(100, dtype=int)
    )
    assert mondrian.q_by_group_[0] == scalar.q_
    lo_s, hi_s = scalar.interval(y[:5], 2 * y[:5])
    lo_m, hi_m = mondrian.interval(y[:5], 2 * y[:5], np.zeros(5, dtype=int))
    assert np.allclose(lo_s, lo_m) and np.allclose(hi_s, hi_m)
