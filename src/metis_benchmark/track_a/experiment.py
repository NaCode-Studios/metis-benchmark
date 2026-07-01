"""Track A experiment runner: out-of-fold predictions for engine and baselines.

One place builds the feature matrix, applies the protocol-assigned split, and
produces out-of-fold predictions for every model on the same held-out rows.
Both the console runner and the confidence-interval analysis consume the same
DatasetRun, so the numbers can never drift between "what we printed" and "what
we tested".
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol

import numpy as np
import pandas as pd

from metis_benchmark.baselines import LogSizeRegression, MedianByCategory
from metis_benchmark.datasets.loaders import load
from metis_benchmark.evaluation import (
    ordered_kfold,
    pred_at,
    rolling_origin_split,
    temporal_split,
)
from metis_benchmark.track_a.features import (
    FEATURE_COLUMNS,
    SPLIT_METHOD,
    encode_features,
    impute_train_median,
)


class Regressor(Protocol):
    """Minimal interface a feature-based Track A regressor must expose."""

    def fit(self, X: np.ndarray, effort: np.ndarray) -> "Regressor": ...
    def predict(self, X: np.ndarray) -> np.ndarray: ...


# A model factory builds a fresh, unfitted regressor for each fold.
ModelFactory = Callable[[], Regressor]


@dataclass
class DatasetRun:
    """Out-of-fold predictions and metadata for one dataset."""

    key: str
    split: str  # human-readable split label
    n_test: int  # number of rows that served as test
    features: list[str]
    y: np.ndarray
    tested_mask: np.ndarray
    # model name -> out-of-fold predictions (NaN outside that model's test rows)
    predictions: dict[str, np.ndarray] = field(default_factory=dict)
    # which engine regressor the pre-registered rule selected for the gate
    gate_regressor: str | None = None

    def scored(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (y_true, y_pred) for the rows where `name` made a prediction."""
        pred = self.predictions[name]
        keep = ~np.isnan(pred)
        return self.y[keep], pred[keep]


def _fold_indices(df: pd.DataFrame, key: str, mode: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train_idx, test_idx) pairs per the dataset's split.

    mode="single" -> the v1.1 single 80/20 temporal split (retained for the
    side-by-side report). mode="rolling" -> the v1.2 pre-registered
    rolling-origin CV. Both pool train+calibration for the point models (CQR
    consumes the calibration block separately in Step 3). Dateless datasets
    (China) ignore mode and stay on ordered k-fold CV.
    """
    if SPLIT_METHOD[key] != "temporal":
        return [
            (np.concatenate([f.train, f.calibration]), f.test)
            for f in ordered_kfold(len(df), n_splits=5, seed=0)
        ]
    if mode == "rolling":
        return [
            (np.concatenate([f.train, f.calibration]), f.test)
            for f in rolling_origin_split(df, "date")
        ]
    s = temporal_split(df, "date")
    return [(np.concatenate([s.train, s.calibration]), s.test)]


def select_gate_regressor(
    X: np.ndarray, y: np.ndarray, size: np.ndarray, models: dict[str, ModelFactory]
) -> str:
    """Pick GBM or GP for the gate by test-blind train-internal validation.

    Pre-registered rule (protocol v1.2 sec. 8): GBM is primary; fall back to GP
    when GBM does not beat the log-size baseline on a held-out validation slice
    drawn from the *training* data only. The validation slice here is the last
    20% of the rows passed in (which the caller restricts to the initial
    rolling-origin training window, so no test-fold row is ever seen).
    """
    n = len(y)
    cut = int(round(n * 0.8))
    tr = slice(0, cut)
    va = slice(cut, n)
    if cut < 2 or n - cut < 1:
        return "GBM (engine)"  # too small to validate -> keep the primary
    # log-size baseline on the validation slice (the bar GBM must clear).
    base = LogSizeRegression().fit(size[tr], y[tr]).predict(size[va])
    gbm = models["GBM (engine)"]().fit(X[tr], y[tr]).predict(X[va])
    # Keep GBM only if it is at least as accurate as the baseline; else GP,
    # following the cold-start doctrine (GP for tiny, overfit-prone samples).
    if pred_at(y[va], gbm) >= pred_at(y[va], base):
        return "GBM (engine)"
    return "GP (engine)"


def run_dataset(key: str, models: dict[str, ModelFactory], mode: str = "single") -> DatasetRun:
    """Fit every model and the protocol baselines out-of-fold on one dataset.

    `models` maps a display name to a factory of feature-based regressors.
    `mode` selects the split: "single" (v1.1, one 80/20 temporal split) or
    "rolling" (v1.2 pre-registered rolling-origin CV). Baselines are always
    added: log-size regression, median-by-category (where a category exists)
    and the expert estimate (where recorded), each scored on the same rows.

    In "rolling" mode a "gate (selected)" prediction series is also produced,
    using the regressor chosen by the pre-registered test-blind rule.
    """
    df = load(key).reset_index(drop=True)
    feature_cols = FEATURE_COLUMNS[key]
    # Require only the target, size and (for temporal datasets) the date. Missing
    # *features* are no longer a reason to drop a row — they are imputed with the
    # training-fold median (v1.3), so datasets with sparse soft attributes (seera)
    # keep their rows instead of being decimated.
    needed = ["effort", "size"]
    if SPLIT_METHOD[key] == "temporal":
        needed.append("date")
    df = df.dropna(subset=needed).reset_index(drop=True)
    df = df[df["effort"] > 0].reset_index(drop=True)

    # Encode features to numeric (ordinal for COCOMO, coercion elsewhere); NaNs
    # are imputed per fold below.
    X = encode_features(df, key).to_numpy(dtype=float)
    y = df["effort"].to_numpy(dtype=float)
    size = df["size"].to_numpy(dtype=float)
    n = len(df)

    folds = _fold_indices(df, key, mode)

    # Pre-registered gate-regressor selection (rolling, temporal datasets only).
    # The selection uses the FIRST fold's training window, which never contains
    # a test-fold row, so it is test-blind.
    gate_regressor: str | None = None
    if mode == "rolling" and SPLIT_METHOD[key] == "temporal" and "GBM (engine)" in models:
        first_train = folds[0][0]
        # Impute on the first training window only (test-blind selection input).
        X_sel = impute_train_median(X[first_train])[0]
        gate_regressor = select_gate_regressor(
            X_sel, y[first_train], size[first_train], models
        )

    # Prediction vectors, NaN-initialized; filled only on each model's test rows.
    preds: dict[str, np.ndarray] = {name: np.full(n, np.nan) for name in models}
    preds["log-size regression"] = np.full(n, np.nan)
    if gate_regressor is not None:
        preds["gate (selected)"] = np.full(n, np.nan)
    has_category = "category" in df and df["category"].notna().any()
    if has_category:
        preds["median-by-category"] = np.full(n, np.nan)
    tested_mask = np.zeros(n, dtype=bool)

    for tr, te in folds:
        tested_mask[te] = True
        # Impute missing features with the training-fold median (test rows get
        # the train medians too — no test statistics leak into imputation).
        Xtr, Xte = impute_train_median(X[tr], X[te])
        # Feature-based regressors (engine candidates).
        fitted = {}
        for name, factory in models.items():
            model = factory().fit(Xtr, y[tr])
            fitted[name] = model
            preds[name][te] = model.predict(Xte)
        # The selected gate regressor reuses the already-fitted model. It must
        # predict on the *imputed* test matrix (Xte), exactly like the model it
        # points to: the model was fitted on imputed data, so feeding it the raw
        # X[te] (with NaNs) would silently change — or NaN out — its predictions
        # and desynchronize "gate (selected)" from the selected regressor's row.
        if gate_regressor is not None:
            preds["gate (selected)"][te] = fitted[gate_regressor].predict(Xte)
        # Baseline: log-size regression on the canonical size measure.
        preds["log-size regression"][te] = (
            LogSizeRegression().fit(size[tr], y[tr]).predict(size[te])
        )
        # Baseline: median-by-category, where the dataset has categories.
        if has_category:
            cat = MedianByCategory().fit(df.loc[tr, "category"], pd.Series(y[tr]))
            preds["median-by-category"][te] = cat.predict(df.loc[te, "category"])

    # Expert baseline: the recorded estimate, scored only on the tested rows so
    # the comparison is apples-to-apples with the models (never on its own
    # training rows).
    if "expert_estimate" in df:
        expert = np.full(n, np.nan)
        col = df["expert_estimate"].to_numpy(dtype=float)
        expert[tested_mask] = col[tested_mask]
        preds["expert estimate"] = expert

    if SPLIT_METHOD[key] != "temporal":
        split_label = "ordered 5-fold CV"
    else:
        split_label = "rolling-origin CV" if mode == "rolling" else "single temporal"
    return DatasetRun(
        key=key,
        split=split_label,
        n_test=int(tested_mask.sum()),
        features=feature_cols,
        y=y,
        tested_mask=tested_mask,
        predictions=preds,
        gate_regressor=gate_regressor,
    )
