"""Engine-vs-baseline run on the Track A datasets.

Fits the GP and the protocol baselines on the same split and reports
PRED(25)/MdAPE on the pooled test rows. The split method follows the frozen
protocol per dataset (temporal where a date exists, ordered k-fold CV for the
dateless China). Every row is predicted exactly once, by a model that did not
train on it.

Usage: python scripts/run_track_a.py [china desharnais kitchenham maxwell]
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from metis_benchmark.baselines import LogSizeRegression, MedianByCategory
from metis_benchmark.datasets.loaders import load
from metis_benchmark.evaluation import mdape, ordered_kfold, pred_at, temporal_split
from metis_benchmark.track_a.features import FEATURE_COLUMNS, SPLIT_METHOD


def _folds(df: pd.DataFrame, key: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return (train_idx, test_idx) pairs per the dataset's protocol split."""
    if SPLIT_METHOD[key] == "temporal":
        # One chronological split: a single (train, test) pair. The calibration
        # block is reserved for conformal intervals (next step), not used here.
        s = temporal_split(df, "date")
        return [(np.concatenate([s.train, s.calibration]), s.test)]
    # Dateless, ungrouped -> 5-fold CV; pool train+calibration for the point model.
    return [(np.concatenate([f.train, f.calibration]), f.test) for f in ordered_kfold(len(df), 5, seed=0)]


def run(key: str) -> None:
    df = load(key).reset_index(drop=True)
    feature_cols = FEATURE_COLUMNS[key]
    # Require a positive target and complete features/size for a fair comparison.
    needed = ["effort", "size", *feature_cols]
    if SPLIT_METHOD[key] == "temporal":
        needed.append("date")
    df = df.dropna(subset=needed).reset_index(drop=True)
    df = df[df["effort"] > 0].reset_index(drop=True)

    X = df[feature_cols].to_numpy(dtype=float)
    y = df["effort"].to_numpy(dtype=float)
    size = df["size"].to_numpy(dtype=float)
    n = len(df)

    # Out-of-fold prediction vectors for the engine and each baseline.
    gp_pred = np.full(n, np.nan)
    size_pred = np.full(n, np.nan)
    cat_pred = np.full(n, np.nan)
    # The expert estimate is a recorded column, but it must be scored on the
    # same held-out test rows as the models, never on the training rows it was
    # made for; tested_mask marks the rows that served as test somewhere.
    expert_col = df["expert_estimate"].to_numpy(dtype=float) if "expert_estimate" in df else None
    expert_pred = np.full(n, np.nan) if expert_col is not None else None
    tested_mask = np.zeros(n, dtype=bool)

    from metis_benchmark.track_a.gp import LogGaussianProcess

    for tr, te in _folds(df, key):
        tested_mask[te] = True
        # Engine: GP on the dataset's features, in log space.
        gp_pred[te] = LogGaussianProcess(seed=0).fit(X[tr], y[tr]).predict(X[te])
        # Baseline 1: log-size regression on the canonical size measure.
        size_pred[te] = LogSizeRegression().fit(size[tr], y[tr]).predict(size[te])
        # Baseline 2: median-by-category, only where the dataset has categories.
        if "category" in df and df["category"].notna().any():
            cat = MedianByCategory().fit(df.loc[tr, "category"], pd.Series(y[tr]))
            cat_pred[te] = cat.predict(df.loc[te, "category"])

    if expert_pred is not None:
        # Restrict the expert baseline to the tested rows for a fair comparison.
        expert_pred[tested_mask] = expert_col[tested_mask]

    split_label = "temporal" if SPLIT_METHOD[key] == "temporal" else "ordered 5-fold CV"
    print(f"== {key} (n={n}, {split_label}) ==")
    rows = [("GP (engine)", gp_pred), ("log-size regression", size_pred)]
    if not np.all(np.isnan(cat_pred)):
        rows.append(("median-by-category", cat_pred))
    if expert_pred is not None:
        rows.append(("expert estimate", expert_pred))
    for name, pred in rows:
        # Score only rows where the model produced a prediction.
        keep = ~np.isnan(pred)
        print(f"  {name:<22} PRED(25)={pred_at(y[keep], pred[keep]):5.1%}  "
              f"MdAPE={mdape(y[keep], pred[keep]):5.1%}  (n={keep.sum()})")


if __name__ == "__main__":
    for key in sys.argv[1:] or ["china", "desharnais", "kitchenham", "maxwell"]:
        run(key)
