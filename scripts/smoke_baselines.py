"""End-to-end pipeline smoke test: baselines + temporal split + metrics.

Runs the mandatory G0 baselines on datasets with usable dates and prints
PRED(25)/MdAPE on the test block. These numbers are the bar the engine must
beat — not results of the engine itself.

Usage: python scripts/smoke_baselines.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metis_benchmark.baselines import LogSizeRegression, MedianByCategory
from metis_benchmark.datasets.loaders import load
from metis_benchmark.evaluation import mdape, pred_at, temporal_split


def report(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    keep = ~np.isnan(y_pred)
    if keep.sum() == 0:
        print(f"  {name:<22} n/a")
        return
    p = pred_at(y_true[keep], y_pred[keep])
    m = mdape(y_true[keep], y_pred[keep])
    print(f"  {name:<22} PRED(25)={p:5.1%}  MdAPE={m:5.1%}  (n={keep.sum()})")


def run(key: str) -> None:
    df = load(key).dropna(subset=["effort", "date"]).reset_index(drop=True)
    df = df[df["effort"] > 0].reset_index(drop=True)
    split = temporal_split(df, "date")
    train, test = df.loc[split.train], df.loc[split.test]
    y_test = test["effort"].to_numpy(dtype=float)
    print(f"\n== {key} (train={len(train)}, test={len(test)}) ==")

    if "category" in df and df["category"].notna().any():
        baseline = MedianByCategory().fit(train["category"], train["effort"])
        report("median-by-category", y_test, baseline.predict(test["category"]))

    if "size" in df:
        sized_train = train.dropna(subset=["size"])
        sized_train = sized_train[sized_train["size"] > 0]
        size_test = test["size"].to_numpy(dtype=float)
        ok = ~np.isnan(size_test) & (size_test > 0)
        reg = LogSizeRegression().fit(sized_train["size"], sized_train["effort"])
        preds = np.full(len(test), np.nan)
        preds[ok] = reg.predict(size_test[ok])
        report("log-size regression", y_test, preds)

    if "expert_estimate" in df:
        report("expert estimate", y_test, test["expert_estimate"].to_numpy(dtype=float))


if __name__ == "__main__":
    for key in ["desharnais", "kitchenham", "sip"]:
        run(key)
