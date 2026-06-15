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
from metis_benchmark.evaluation import ordered_kfold, temporal_split
from metis_benchmark.track_a.features import FEATURE_COLUMNS, SPLIT_METHOD


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

    def scored(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        """Return (y_true, y_pred) for the rows where `name` made a prediction."""
        pred = self.predictions[name]
        keep = ~np.isnan(pred)
        return self.y[keep], pred[keep]


def _fold_indices(df: pd.DataFrame, key: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train_idx, test_idx) pairs per the dataset's frozen-protocol split."""
    if SPLIT_METHOD[key] == "temporal":
        # Single chronological split; train absorbs the calibration block here
        # (the point models do not need a separate calibration set — CQR does,
        # added in Step 3).
        s = temporal_split(df, "date")
        return [(np.concatenate([s.train, s.calibration]), s.test)]
    # Dateless, ungrouped (China) -> 5-fold CV, every row tested once.
    return [
        (np.concatenate([f.train, f.calibration]), f.test)
        for f in ordered_kfold(len(df), n_splits=5, seed=0)
    ]


def run_dataset(key: str, models: dict[str, ModelFactory]) -> DatasetRun:
    """Fit every model and the protocol baselines out-of-fold on one dataset.

    `models` maps a display name to a factory of feature-based regressors
    (GP now, GBM added in Step 2). Baselines are always added: log-size
    regression, median-by-category (where a category exists) and the expert
    estimate (where recorded), each scored on the same held-out rows.
    """
    df = load(key).reset_index(drop=True)
    feature_cols = FEATURE_COLUMNS[key]
    needed = ["effort", "size", *feature_cols]
    if SPLIT_METHOD[key] == "temporal":
        needed.append("date")
    # Drop rows lacking the target or any model input, so all models compete on
    # exactly the same rows.
    df = df.dropna(subset=needed).reset_index(drop=True)
    df = df[df["effort"] > 0].reset_index(drop=True)

    X = df[feature_cols].to_numpy(dtype=float)
    y = df["effort"].to_numpy(dtype=float)
    size = df["size"].to_numpy(dtype=float)
    n = len(df)

    # Prediction vectors, NaN-initialized; filled only on each model's test rows.
    preds: dict[str, np.ndarray] = {name: np.full(n, np.nan) for name in models}
    preds["log-size regression"] = np.full(n, np.nan)
    has_category = "category" in df and df["category"].notna().any()
    if has_category:
        preds["median-by-category"] = np.full(n, np.nan)
    tested_mask = np.zeros(n, dtype=bool)

    for tr, te in _fold_indices(df, key):
        tested_mask[te] = True
        # Feature-based regressors (engine candidates).
        for name, factory in models.items():
            preds[name][te] = factory().fit(X[tr], y[tr]).predict(X[te])
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

    split_label = "temporal" if SPLIT_METHOD[key] == "temporal" else "ordered 5-fold CV"
    return DatasetRun(
        key=key,
        split=split_label,
        n_test=int(tested_mask.sum()),
        features=feature_cols,
        y=y,
        tested_mask=tested_mask,
        predictions=preds,
    )
