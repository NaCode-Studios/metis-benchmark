"""Track B evaluation harness: embeddings + leakage-free folds + k/τ selection.

Loads a task dataset, attaches its cached embeddings, and builds out-of-fold
folds per the frozen protocol split (group-by-project for josse/deepse,
rolling-origin for sip). For each fold the k-NN index is the training tasks
only — held-out projects (or future tasks) never appear in it — so retrieval is
strictly cold-start. k and τ are chosen by test-blind validation inside the
training rows.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from metis_benchmark.datasets.loaders import load
from metis_benchmark.evaluation import grouped_kfold, pred_at, rolling_origin_split
from metis_benchmark.track_b.embeddings import embed_texts
from metis_benchmark.track_b.knn import NadarayaWatsonKNN

# Frozen v2.0 split assignment for the Track B datasets.
SPLIT_B = {"josse": "group", "deepse": "group", "sip": "rolling"}

# Pre-registered k/τ grid for test-blind selection (protocol v2.0).
K_GRID = (5, 10, 20, 50)
TAU_GRID = (0.05, 0.1, 0.2)


@dataclass
class FoldB:
    train: np.ndarray
    test: np.ndarray
    cal: np.ndarray


@dataclass
class DatasetB:
    key: str
    split: str
    emb: np.ndarray
    y: np.ndarray
    project: np.ndarray
    expert: np.ndarray | None
    category: np.ndarray | None
    folds: list[FoldB]


def load_emb_dataset(key: str, n_splits: int = 5) -> DatasetB:
    df = load(key).reset_index(drop=True)
    keep = df["effort"].notna() & (df["effort"] > 0) & df["text"].notna()
    if SPLIT_B[key] == "rolling":
        keep &= df["date"].notna()
    df = df[keep].reset_index(drop=True)

    texts = df["text"].astype(str).tolist()
    emb = embed_texts(texts, key)
    y = df["effort"].to_numpy(dtype=float)
    project = df["project"].astype(str).to_numpy() if "project" in df else None
    expert = df["expert_estimate"].to_numpy(dtype=float) if "expert_estimate" in df else None
    category = df["category"].astype("string").to_numpy() if "category" in df else None

    if SPLIT_B[key] == "group":
        splits = grouped_kfold(pd.Series(project), n_splits=n_splits, seed=0)
        folds = [FoldB(train=s.train, test=s.test, cal=s.calibration) for s in splits]
    else:  # rolling-origin
        splits = rolling_origin_split(df, "date")
        folds = [FoldB(train=s.train, test=s.test, cal=s.calibration) for s in splits]
    return DatasetB(key=key, split=SPLIT_B[key], emb=emb, y=y, project=project,
                    expert=expert, category=category, folds=folds)


def select_k_tau(emb_tr: np.ndarray, y_tr: np.ndarray, seed: int = 0) -> tuple[int, float]:
    """Pick (k, τ) by a random 80/20 split inside the training rows (test-blind).

    The validation split is on rows here (the training projects are already a
    cold-start pool relative to the test projects); it never touches the fold's
    test set, so selection cannot peek at the gate.
    """
    n = len(y_tr)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    cut = int(round(n * 0.8))
    fit, val = perm[:cut], perm[cut:]
    if len(val) < 5 or len(fit) < 5:
        return 20, 0.1  # too small to validate -> grid centre
    best, best_p = (20, 0.1), -1.0
    for k in K_GRID:
        for tau in TAU_GRID:
            model = NadarayaWatsonKNN(k=k, tau=tau).fit(emb_tr[fit], y_tr[fit])
            p = pred_at(y_tr[val], model.predict(emb_tr[val]))
            if p > best_p:
                best, best_p = (k, tau), p
    return best


def knn_oof(data: DatasetB) -> tuple[np.ndarray, np.ndarray, tuple[int, float]]:
    """Out-of-fold k-NN-NW predictions (k/τ selected per fold). Returns
    (tested_mask, predictions, last (k,τ) for reporting)."""
    n = len(data.y)
    pred = np.full(n, np.nan)
    tested = np.zeros(n, dtype=bool)
    chosen = (20, 0.1)
    for f in data.folds:
        # Fit index = train rows (+ calibration rows, which are also past/other
        # projects) — never the test rows.
        idx = np.concatenate([f.train, f.cal])
        chosen = select_k_tau(data.emb[idx], data.y[idx])
        model = NadarayaWatsonKNN(k=chosen[0], tau=chosen[1]).fit(data.emb[idx], data.y[idx])
        pred[f.test] = model.predict(data.emb[f.test])
        tested[f.test] = True
    return tested, pred, chosen
