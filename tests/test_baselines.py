"""Baseline correctness tests against hand-computed or analytically known values."""

import numpy as np
import pandas as pd
import pytest

from metis_benchmark.baselines import LogSizeRegression, MedianByCategory, expert_estimates


def test_median_by_category_with_fallback():
    cats = pd.Series(["web", "web", "mobile", "web"])
    effort = pd.Series([100.0, 200.0, 50.0, 300.0])
    model = MedianByCategory().fit(cats, effort)
    out = model.predict(pd.Series(["web", "mobile", "embedded"]))
    # Known category: median of its training efforts (100/200/300 -> 200).
    assert out[0] == pytest.approx(200.0)
    assert out[1] == pytest.approx(50.0)
    # Unseen category must fall back to the global median, not NaN.
    assert out[2] == pytest.approx(150.0)


def test_log_size_regression_recovers_power_law():
    # Generate exact power-law data effort = 3 * size^0.9: the log-log OLS
    # must recover slope (exponent) and predictions essentially exactly.
    rng = np.random.default_rng(0)
    size = rng.uniform(10, 1000, size=200)
    effort = 3.0 * size**0.9
    model = LogSizeRegression().fit(size, effort)
    assert model.slope_ == pytest.approx(0.9, abs=1e-6)
    assert model.predict(np.array([100.0]))[0] == pytest.approx(3.0 * 100**0.9, rel=1e-6)


def test_log_size_regression_rejects_non_positive():
    # log() is undefined at zero: the fit must refuse rather than crash later.
    with pytest.raises(ValueError):
        LogSizeRegression().fit(np.array([0.0, 1.0]), np.array([1.0, 2.0]))


def test_expert_estimates_mask():
    # Mixed column: numbers, missing, numeric strings, junk text. The mask
    # must mark exactly the usable values and parse them as floats.
    df = pd.DataFrame({"expert": [40.0, None, "80", "n/a"]})
    mask, values = expert_estimates(df, "expert")
    assert mask.tolist() == [True, False, True, False]
    assert values[0] == pytest.approx(40.0)
    assert values[2] == pytest.approx(80.0)
