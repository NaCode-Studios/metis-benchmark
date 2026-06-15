"""W3 comparison: GP-isolated vs GP-mean-function vs GBM, with CI95.

Pooled out-of-fold predictions per dataset under the protocol split, scored
with bootstrap CIs. The GP-pooled variant is added in Phase 3. SEERA is the
sealed holdout and is NOT included here.

Usage: python scripts/run_track_a_w3.py
"""

from __future__ import annotations

import numpy as np

from metis_benchmark.evaluation import bootstrap_metric_ci, mdape, paired_bootstrap_diff, pred_at
from metis_benchmark.track_a.gbm import LogGBMQuantile
from metis_benchmark.track_a.gp import LogGaussianProcess
from metis_benchmark.track_a.w3_experiment import load_dataset, size_law_baseline_log

# Development datasets only — SEERA sealed. China kept as a no-signal reference.
DEV_DATASETS = ["desharnais", "kitchenham", "maxwell", "cocomo81", "china"]


def oof_predictions(key: str) -> dict[str, np.ndarray]:
    """Out-of-fold predictions for each model variant, aligned to test rows."""
    data = load_dataset(key, mode="rolling")
    n = len(data.y)
    preds = {m: np.full(n, np.nan) for m in
             ["GP-isolated", "GP-meanfn", "GBM", "log-size"]}
    for f in data.folds:
        # GP-isolated: zero-mean GP (the W2 model).
        preds["GP-isolated"][f.test] = LogGaussianProcess(seed=0).fit(f.Xtr, f.ytr).predict(f.Xte)
        # GP-mean-function: GP fits residuals of the per-dataset log-size law.
        base_tr = size_law_baseline_log(f.size_tr, f.ytr, f.size_tr)
        base_te = size_law_baseline_log(f.size_tr, f.ytr, f.size_te)
        gp = LogGaussianProcess(seed=0).fit(f.Xtr, f.ytr, baseline_log=base_tr)
        preds["GP-meanfn"][f.test] = gp.predict(f.Xte, baseline_log=base_te)
        # GBM comparator.
        preds["GBM"][f.test] = LogGBMQuantile(seed=0).fit(f.Xtr, f.ytr).predict(f.Xte)
        # log-size baseline.
        from metis_benchmark.baselines import LogSizeRegression
        preds["log-size"][f.test] = LogSizeRegression().fit(f.size_tr, f.ytr).predict(f.size_te)
    return {"y": data.y, "split": data.split, **preds}


def main() -> None:
    pooled = {m: [] for m in ["GP-isolated", "GP-meanfn", "GBM", "log-size"]}
    y_all = []
    for key in DEV_DATASETS:
        r = oof_predictions(key)
        y = r["y"]
        mask = ~np.isnan(r["GP-meanfn"])
        print(f"\n== {key} ({r['split']}, n_test={mask.sum()}) ==")
        for m in ["GP-isolated", "GP-meanfn", "GBM", "log-size"]:
            yt, yp = y[mask], r[m][mask]
            print(f"  {m:<12} PRED(25)={bootstrap_metric_ci(yt, yp, pred_at, seed=0)}  "
                  f"MdAPE={bootstrap_metric_ci(yt, yp, mdape, seed=0)}")
        # mean-function vs isolated, paired.
        d = paired_bootstrap_diff(y[mask], r["GP-meanfn"][mask], r["GP-isolated"][mask], pred_at, seed=0)
        print(f"  -> GP-meanfn − GP-isolated (PRED25): {d}")
        if key != "china":  # china is out-gate (Q11), excluded from the aggregate
            for m in pooled:
                pooled[m].append(r[m][mask])
            y_all.append(y[mask])

    print("\n== AGGREGATE (gate dev datasets: desharnais, kitchenham, maxwell, cocomo81) ==")
    y = np.concatenate(y_all)
    cols = {m: np.concatenate(v) for m, v in pooled.items()}
    for m in ["GP-isolated", "GP-meanfn", "GBM", "log-size"]:
        print(f"  {m:<12} PRED(25)={bootstrap_metric_ci(y, cols[m], pred_at, seed=0)}  "
              f"MdAPE={bootstrap_metric_ci(y, cols[m], mdape, seed=0)}")
    d = paired_bootstrap_diff(y, cols["GP-meanfn"], cols["log-size"], pred_at, seed=0)
    print(f"  -> GP-meanfn − log-size (PRED25): {d}")


if __name__ == "__main__":
    main()
