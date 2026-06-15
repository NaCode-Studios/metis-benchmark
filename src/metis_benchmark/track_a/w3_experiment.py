"""W3 evaluation harness for the mean-function and pooled GP variants.

The generic run_dataset() drives models with a fit(X, y)/predict(X) contract, so
it cannot feed a size-law baseline to the mean-function GP, nor coordinate a
pooled baseline across datasets. This harness exposes the per-fold raw
materials (imputed train/test features, size, effort, indices) so each variant
— GP-isolated, GP-mean-function, GP-pooled, GBM — can be scored on identical
folds. SEERA is never loaded here during development; it is the sealed holdout.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from metis_benchmark.baselines import LogSizeRegression
from metis_benchmark.datasets.loaders import load
from metis_benchmark.evaluation import ordered_kfold, rolling_origin_split, temporal_split
from metis_benchmark.track_a.features import (
    FEATURE_COLUMNS,
    SPLIT_METHOD,
    encode_features,
    impute_train_median,
)


@dataclass
class Fold:
    """Imputed raw materials for one fold, shared by every model variant."""

    train: np.ndarray
    test: np.ndarray
    Xtr: np.ndarray
    Xte: np.ndarray
    ytr: np.ndarray
    yte: np.ndarray
    size_tr: np.ndarray
    size_te: np.ndarray


@dataclass
class DatasetData:
    key: str
    split: str
    y: np.ndarray
    size: np.ndarray
    X: np.ndarray  # encoded, NOT imputed (imputation is per fold)
    folds: list[Fold]


def _split_indices(df, key, mode):
    if SPLIT_METHOD[key] != "temporal":
        return [(np.concatenate([f.train, f.calibration]), f.test)
                for f in ordered_kfold(len(df), n_splits=5, seed=0)]
    if mode == "rolling":
        return [(np.concatenate([f.train, f.calibration]), f.test)
                for f in rolling_origin_split(df, "date")]
    s = temporal_split(df, "date")
    return [(np.concatenate([s.train, s.calibration]), s.test)]


def load_dataset(key: str, mode: str = "rolling") -> DatasetData:
    """Load a dataset and build its per-fold imputed materials."""
    df = load(key).reset_index(drop=True)
    needed = ["effort", "size"] + (["date"] if SPLIT_METHOD[key] == "temporal" else [])
    df = df.dropna(subset=needed).reset_index(drop=True)
    df = df[df["effort"] > 0].reset_index(drop=True)

    X = encode_features(df, key).to_numpy(dtype=float)
    y = df["effort"].to_numpy(dtype=float)
    size = df["size"].to_numpy(dtype=float)

    folds = []
    for tr, te in _split_indices(df, key, mode):
        Xtr, Xte = impute_train_median(X[tr], X[te])
        folds.append(Fold(
            train=tr, test=te, Xtr=Xtr, Xte=Xte,
            ytr=y[tr], yte=y[te], size_tr=size[tr], size_te=size[te],
        ))
    label = ("rolling-origin CV" if (SPLIT_METHOD[key] == "temporal" and mode == "rolling")
             else "single temporal" if SPLIT_METHOD[key] == "temporal"
             else "ordered 5-fold CV")
    return DatasetData(key=key, split=label, y=y, size=size, X=X, folds=folds)


def size_law_baseline_log(size_tr, y_tr, size_eval) -> np.ndarray:
    """log-space prediction of the per-dataset log-size power law, fit on train."""
    reg = LogSizeRegression().fit(size_tr, y_tr)
    return np.log(reg.predict(size_eval))


def initial_window(key: str, fraction: float = 0.5) -> tuple[np.ndarray, np.ndarray]:
    """(size, effort) of the earliest `fraction` of a temporal dataset's rows.

    These never-test initial-window rows are the only ones allowed to feed the
    pooled global slope (protocol v1.3 addendum), so the global is leakage-free.
    Only defined for temporal datasets.
    """
    if SPLIT_METHOD[key] != "temporal":
        raise ValueError(f"{key} is not temporal; it cannot contribute to the global")
    df = load(key).dropna(subset=["effort", "size", "date"]).reset_index(drop=True)
    df = df[df["effort"] > 0].reset_index(drop=True)
    order = np.argsort(df["date"].to_numpy(), kind="stable")
    n_init = int(round(len(order) * fraction))
    idx = order[:n_init]
    return df["size"].to_numpy(dtype=float)[idx], df["effort"].to_numpy(dtype=float)[idx]
