"""Coherence tests for the Track A experiment runner (run_dataset).

The critical invariant: the "gate (selected)" series must be *identical* to the
out-of-fold predictions of the regressor the pre-registered rule selected. The
gate verdict is read off that series, so any divergence (e.g. predicting from a
non-imputed matrix) silently corrupts the gate number without failing anywhere.
"""

import numpy as np
import pandas as pd
import pytest

from metis_benchmark.track_a import experiment
from metis_benchmark.track_a.features import FEATURE_COLUMNS, SPLIT_METHOD


class NaNPropagatingRegressor:
    """Minimal Regressor whose predictions poison on NaN inputs.

    predict() sums the feature row, so a single NaN feature yields a NaN
    prediction. This makes the test sharp: if run_dataset ever feeds a model
    the raw (non-imputed) test matrix, the gate series grows NaNs and/or stops
    matching the selected model's own out-of-fold series.
    """

    def __init__(self, offset: float = 0.0) -> None:
        # A per-model offset so "GBM" and "GP" fakes give different predictions,
        # which lets the equality assertion distinguish the two.
        self._offset = offset
        self._mean = 0.0

    def fit(self, X: np.ndarray, effort: np.ndarray) -> "NaNPropagatingRegressor":
        # Remember the training mean so predictions live near the target scale.
        self._mean = float(np.mean(effort))
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Row-sum propagates any NaN feature straight into the prediction.
        return self._mean + self._offset + np.asarray(X, dtype=float).sum(axis=1)


@pytest.fixture()
def synthetic_gate_dataset(monkeypatch):
    """Register a synthetic temporal dataset with NaNs in the feature matrix."""
    n = 40
    rng = np.random.default_rng(0)
    df = pd.DataFrame(
        {
            # Monotonic dates -> the rolling-origin (temporal) split applies.
            "date": pd.date_range("2020-01-01", periods=n, freq="7D"),
            # Positive size and effort (required by the log-space baselines).
            "size": rng.uniform(10.0, 500.0, n),
            "effort": rng.uniform(100.0, 5000.0, n),
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
        }
    )
    # Sprinkle NaNs over both features, including rows that will fall in the
    # rolling-origin test folds (the back half): these are the rows where the
    # imputation bug — predicting from the raw matrix — becomes observable.
    df.loc[::5, "f1"] = np.nan
    df.loc[3::7, "f2"] = np.nan

    key = "synthetic_gate"
    # Register the synthetic dataset in the (shared) protocol tables and stub
    # the loader, so run_dataset exercises its real code path end to end.
    monkeypatch.setitem(FEATURE_COLUMNS, key, ["f1", "f2"])
    monkeypatch.setitem(SPLIT_METHOD, key, "temporal")
    monkeypatch.setattr(experiment, "load", lambda k: df)
    return key


def test_gate_selected_matches_selected_regressor_exactly(synthetic_gate_dataset):
    # Both engine candidates present, as the selection rule expects.
    models = {
        "GBM (engine)": lambda: NaNPropagatingRegressor(offset=0.0),
        "GP (engine)": lambda: NaNPropagatingRegressor(offset=100.0),
    }
    run = experiment.run_dataset(synthetic_gate_dataset, models, mode="rolling")

    # The rolling temporal mode must have produced a gate selection.
    assert run.gate_regressor in models
    gate = run.predictions["gate (selected)"]
    selected = run.predictions[run.gate_regressor]

    # Exact coherence: the gate series IS the selected regressor's series, row
    # by row (NaN outside the tested rows on both, identical values inside).
    np.testing.assert_array_equal(gate, selected)

    # No NaN may survive on tested rows for ANY model: features were imputed
    # with the training-fold median before every fit and predict.
    for name, pred in run.predictions.items():
        assert not np.isnan(pred[run.tested_mask]).any(), f"NaN predictions in {name!r}"
