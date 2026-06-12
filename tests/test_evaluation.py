"""Metric correctness tests: wrong metrics would invalidate the whole gate,
so each one is checked against hand-computed values."""

import numpy as np
import pandas as pd
import pytest

from metis_benchmark.evaluation import (
    ape,
    empirical_coverage,
    mdape,
    pred_at,
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
