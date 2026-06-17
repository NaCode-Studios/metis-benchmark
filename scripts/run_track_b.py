"""Track B semantic engine vs baselines, with bootstrap CI95.

k-NN Nadaraya-Watson over bge-small embeddings (k/τ test-blind selected) vs the
mandatory baselines: global median (text-blind floor — the engine must beat it
to prove text carries signal), median-by-category where present, and the human
expert (the binding bar). Out-of-fold under the protocol split.

Usage: python scripts/run_track_b.py [deepse sip josse]
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from metis_benchmark.evaluation import bootstrap_metric_ci, mdape, paired_bootstrap_diff, pred_at
from metis_benchmark.track_b.experiment import knn_oof, load_emb_dataset


def baselines_oof(data) -> dict[str, np.ndarray]:
    """Out-of-fold global-median and median-by-category predictions."""
    n = len(data.y)
    gmed = np.full(n, np.nan)
    cat = np.full(n, np.nan) if data.category is not None else None
    for f in data.folds:
        idx = np.concatenate([f.train, f.cal])
        gmed[f.test] = np.median(data.y[idx])
        if cat is not None:
            frame = pd.DataFrame({"c": data.category[idx], "y": data.y[idx]})
            med = frame.groupby("c")["y"].median()
            glob = float(frame["y"].median())
            cat[f.test] = pd.Series(data.category[f.test]).map(med).fillna(glob).to_numpy()
    out = {"global-median": gmed}
    if cat is not None:
        out["median-by-category"] = cat
    return out


def run(key: str) -> dict:
    data = load_emb_dataset(key)
    tested, knn_pred, chosen = knn_oof(data)
    preds = {"k-NN-NW (engine)": knn_pred, **baselines_oof(data)}
    if data.expert is not None:
        expert = np.full(len(data.y), np.nan)
        expert[tested] = data.expert[tested]
        preds["expert"] = expert

    print(f"\n== {key} ({data.split}, n_test={tested.sum()}, k/τ={chosen}) ==")
    for name, pred in preds.items():
        m = tested & ~np.isnan(pred)
        pci = bootstrap_metric_ci(data.y[m], pred[m], pred_at, seed=0)
        mci = bootstrap_metric_ci(data.y[m], pred[m], mdape, seed=0)
        print(f"  {name:<20} PRED(25)={pci}  MdAPE={mci}  (n={m.sum()})")

    # The decisive comparisons: engine vs the text-blind floor, and engine vs expert.
    m = tested & ~np.isnan(preds["global-median"])
    d = paired_bootstrap_diff(data.y[m], knn_pred[m], preds["global-median"][m], pred_at, seed=0)
    print(f"  -> engine − global-median (PRED25): {d}  "
          f"{'TEXT ADDS SIGNAL' if d.low > 0 else 'no proven signal'}")
    if "expert" in preds:
        me = tested & ~np.isnan(preds["expert"])
        de = paired_bootstrap_diff(data.y[me], knn_pred[me], preds["expert"][me], pred_at, seed=0)
        print(f"  -> engine − expert (PRED25): {de}  "
              f"{'beats expert' if de.low > 0 else 'below/tie expert'}")
    return {"key": key, "tested": tested, "y": data.y, "engine": knn_pred}


if __name__ == "__main__":
    for key in sys.argv[1:] or ["deepse", "sip", "josse"]:
        run(key)
