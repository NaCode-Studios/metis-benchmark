"""Metric correctness tests: wrong metrics would invalidate the whole gate,
so each one is checked against hand-computed values."""

import numpy as np
import pandas as pd
import pytest

from metis_benchmark.evaluation import (
    ape,
    bootstrap_metric_ci,
    empirical_coverage,
    group_split,
    mdape,
    ordered_kfold,
    paired_bootstrap_diff,
    pred_at,
    rolling_origin_split,
    temporal_split,
)


def test_ape_hand_computed():
    # 110 vs 100 -> 10% error; 150 vs 200 -> 25% error.
    out = ape(np.array([100.0, 200.0]), np.array([110.0, 150.0]))
    assert out == pytest.approx([0.10, 0.25])


def test_ape_rejects_non_positive_actuals():
    # Division by a non-positive actual must fail loudly, not return inf/NaN.
    with pytest.raises(ValueError):
        ape(np.array([0.0, 10.0]), np.array([1.0, 1.0]))


def test_pred25_counts_boundary_as_hit():
    # Errors: 25%, 25%, 30%, 0% -> exactly-25% counts as a hit, so 3/4.
    y_true = np.array([100.0, 100.0, 100.0, 100.0])
    y_pred = np.array([125.0, 75.0, 130.0, 100.0])
    assert pred_at(y_true, y_pred, level=0.25) == pytest.approx(0.75)


def test_mdape_is_median_not_mean():
    # Errors: 10%, 20%, 300% -> the median (20%) must ignore the outlier.
    y_true = np.array([100.0, 100.0, 100.0])
    y_pred = np.array([110.0, 120.0, 400.0])
    assert mdape(y_true, y_pred) == pytest.approx(0.20)


def test_coverage_hand_computed():
    # Intervals capture samples 1, 3 and 4 but miss sample 2 -> 3/4 coverage.
    y = np.array([5.0, 10.0, 15.0, 20.0])
    lo = np.array([4.0, 11.0, 10.0, 19.0])
    hi = np.array([6.0, 12.0, 20.0, 21.0])
    assert empirical_coverage(y, lo, hi) == pytest.approx(0.75)


def test_coverage_rejects_inverted_intervals():
    # lower > upper is a bug in interval construction, not a scoring case.
    with pytest.raises(ValueError):
        empirical_coverage(np.array([1.0]), np.array([2.0]), np.array([1.0]))


def test_temporal_split_is_chronological():
    # Dates arrive unsorted on purpose: the split must order them itself.
    df = pd.DataFrame(
        {"date": pd.to_datetime(["2021-03-01", "2020-01-01", "2022-06-01", "2021-09-01", "2020-07-01"])}
    )
    split = temporal_split(df, "date", calibration_fraction=0.2, test_fraction=0.2)
    train_dates = df.loc[split.train, "date"]
    cal_dates = df.loc[split.calibration, "date"]
    test_dates = df.loc[split.test, "date"]
    # Chronological ordering between blocks: train <= calibration <= test.
    assert train_dates.max() <= cal_dates.min()
    assert cal_dates.max() <= test_dates.min()
    # Partition property: every row used exactly once across the three blocks.
    used = np.concatenate([split.train, split.calibration, split.test])
    assert sorted(used) == list(range(len(df)))


def test_temporal_split_rejects_missing_dates():
    # A NaT cannot be placed on the timeline; the split must refuse it.
    df = pd.DataFrame({"date": [pd.Timestamp("2021-01-01"), pd.NaT]})
    with pytest.raises(ValueError):
        temporal_split(df, "date")


def test_group_split_keeps_projects_disjoint():
    # 10 projects, 5 rows each. No project may appear in two blocks at once.
    groups = pd.Series([f"P{p}" for p in range(10) for _ in range(5)])
    split = group_split(groups, calibration_fraction=0.2, test_fraction=0.2, seed=0)
    g = groups.to_numpy()
    train_g = set(g[split.train])
    cal_g = set(g[split.calibration])
    test_g = set(g[split.test])
    # Pairwise disjoint group sets is the whole point of the cold-start split.
    assert train_g.isdisjoint(cal_g)
    assert train_g.isdisjoint(test_g)
    assert cal_g.isdisjoint(test_g)
    # Every row is used exactly once across the three blocks.
    used = np.concatenate([split.train, split.calibration, split.test])
    assert sorted(used) == list(range(len(groups)))


def test_group_split_is_deterministic():
    # Same seed must reproduce the exact same partition.
    groups = pd.Series([f"P{p}" for p in range(12) for _ in range(3)])
    a = group_split(groups, seed=7)
    b = group_split(groups, seed=7)
    assert np.array_equal(a.test, b.test)


def test_group_split_needs_three_groups():
    # With only two projects there is no room for train+cal+test.
    groups = pd.Series(["A", "A", "B", "B"])
    with pytest.raises(ValueError):
        group_split(groups)


def test_ordered_kfold_partitions_test_blocks():
    # Across all folds, each row is a test sample exactly once.
    folds = ordered_kfold(n_samples=50, n_splits=5, seed=0)
    assert len(folds) == 5
    all_test = np.concatenate([f.test for f in folds])
    assert sorted(all_test) == list(range(50))


def test_ordered_kfold_blocks_are_disjoint_within_fold():
    # Inside one fold, train / calibration / test must not overlap.
    folds = ordered_kfold(n_samples=50, n_splits=5, seed=0)
    for f in folds:
        idx = np.concatenate([f.train, f.calibration, f.test])
        assert len(idx) == len(set(idx.tolist()))


def test_ordered_kfold_is_deterministic():
    # Fixed seed -> identical folds across runs (reproducible, not random).
    a = ordered_kfold(40, n_splits=4, seed=3)
    b = ordered_kfold(40, n_splits=4, seed=3)
    for fa, fb in zip(a, b):
        assert np.array_equal(fa.test, fb.test)


def test_rolling_origin_is_strictly_past_to_future():
    # Each fold's entire training+calibration window must end before its test
    # block begins (no future leakage), and folds expand over time.
    dates = pd.date_range("2020-01-01", periods=100, freq="D")
    df = pd.DataFrame({"date": dates.to_numpy()})
    folds = rolling_origin_split(df, "date", initial_train_fraction=0.5, n_folds=5)
    assert len(folds) == 5
    d = df["date"].to_numpy()
    prev_train_size = -1
    for f in folds:
        train_block = np.concatenate([f.train, f.calibration])
        # Strict temporal order: newest training row precedes oldest test row.
        assert d[train_block].max() < d[f.test].min()
        # Expanding window: the training pool grows fold over fold.
        assert len(train_block) > prev_train_size
        prev_train_size = len(train_block)


def test_rolling_origin_pools_to_back_half():
    # The pooled test rows cover the back 50% of the data exactly once, with no
    # overlap between fold test blocks.
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=100, freq="D").to_numpy()})
    folds = rolling_origin_split(df, "date", initial_train_fraction=0.5, n_folds=5)
    pooled = np.concatenate([f.test for f in folds])
    assert len(pooled) == len(set(pooled.tolist()))  # no row tested twice
    assert len(pooled) == 50  # back half


def test_rolling_origin_respects_min_test_per_fold():
    # With few back rows, the fold count must drop so each fold keeps >=5 rows.
    df = pd.DataFrame({"date": pd.date_range("2020-01-01", periods=40, freq="D").to_numpy()})
    folds = rolling_origin_split(
        df, "date", initial_train_fraction=0.5, n_folds=5, min_test_per_fold=5
    )
    for f in folds:
        assert len(f.test) >= 5


def test_bootstrap_ci_brackets_point_estimate():
    # The CI must contain the full-sample point estimate, with low <= high.
    rng = np.random.default_rng(0)
    y = rng.uniform(100, 1000, size=200)
    pred = y * rng.uniform(0.7, 1.3, size=200)
    ci = bootstrap_metric_ci(y, pred, pred_at, n_boot=1000, seed=1)
    assert ci.low <= ci.point <= ci.high


def test_bootstrap_ci_zero_width_when_perfect():
    # Perfect predictions -> PRED(25) is 1.0 on every resample, so the CI
    # collapses to a point.
    y = np.array([100.0, 200.0, 300.0, 400.0])
    ci = bootstrap_metric_ci(y, y.copy(), pred_at, n_boot=500, seed=0)
    assert ci.point == pytest.approx(1.0)
    assert ci.low == pytest.approx(1.0)
    assert ci.high == pytest.approx(1.0)


def test_paired_bootstrap_diff_detects_clear_winner():
    # pred_a is near-perfect, pred_b is badly biased: the PRED(25) difference
    # must be positive with a CI that excludes 0.
    rng = np.random.default_rng(2)
    y = rng.uniform(100, 1000, size=300)
    pred_a = y * rng.uniform(0.95, 1.05, size=300)
    pred_b = y * 2.0
    diff = paired_bootstrap_diff(y, pred_a, pred_b, pred_at, n_boot=1000, seed=3)
    assert diff.point > 0
    assert diff.low > 0


def test_paired_bootstrap_diff_straddles_zero_for_ties():
    # Two equally-noisy models (noise wide enough that PRED(25) varies, not a
    # saturated 1.0) -> the difference CI should contain 0.
    rng = np.random.default_rng(4)
    y = rng.uniform(100, 1000, size=300)
    pred_a = y * rng.uniform(0.6, 1.4, size=300)
    pred_b = y * rng.uniform(0.6, 1.4, size=300)
    diff = paired_bootstrap_diff(y, pred_a, pred_b, pred_at, n_boot=1000, seed=5)
    assert diff.low < 0 < diff.high
