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
    out = ape(np.array([100.0, 200.0]), np.array([110.0, 150.0]))
    assert out == pytest.approx([0.10, 0.25])


def test_ape_rejects_non_positive_actuals():
    with pytest.raises(ValueError):
        ape(np.array([0.0, 10.0]), np.array([1.0, 1.0]))


def test_pred25_counts_boundary_as_hit():
    y_true = np.array([100.0, 100.0, 100.0, 100.0])
    y_pred = np.array([125.0, 75.0, 130.0, 100.0])  # 25% errors are hits
    assert pred_at(y_true, y_pred, level=0.25) == pytest.approx(0.75)


def test_mdape_is_median_not_mean():
    y_true = np.array([100.0, 100.0, 100.0])
    y_pred = np.array([110.0, 120.0, 400.0])  # outlier must not dominate
    assert mdape(y_true, y_pred) == pytest.approx(0.20)


def test_coverage_hand_computed():
    y = np.array([5.0, 10.0, 15.0, 20.0])
    lo = np.array([4.0, 11.0, 10.0, 19.0])
    hi = np.array([6.0, 12.0, 20.0, 21.0])
    assert empirical_coverage(y, lo, hi) == pytest.approx(0.75)


def test_coverage_rejects_inverted_intervals():
    with pytest.raises(ValueError):
        empirical_coverage(np.array([1.0]), np.array([2.0]), np.array([1.0]))


def test_temporal_split_is_chronological():
    df = pd.DataFrame(
        {"date": pd.to_datetime(["2021-03-01", "2020-01-01", "2022-06-01", "2021-09-01", "2020-07-01"])}
    )
    split = temporal_split(df, "date", calibration_fraction=0.2, test_fraction=0.2)
    train_dates = df.loc[split.train, "date"]
    cal_dates = df.loc[split.calibration, "date"]
    test_dates = df.loc[split.test, "date"]
    assert train_dates.max() <= cal_dates.min()
    assert cal_dates.max() <= test_dates.min()
    # every row used exactly once
    used = np.concatenate([split.train, split.calibration, split.test])
    assert sorted(used) == list(range(len(df)))


def test_temporal_split_rejects_missing_dates():
    df = pd.DataFrame({"date": [pd.Timestamp("2021-01-01"), pd.NaT]})
    with pytest.raises(ValueError):
        temporal_split(df, "date")
