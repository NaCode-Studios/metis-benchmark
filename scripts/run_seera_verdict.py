"""W3 Phase 5 — open the SEERA holdout EXACTLY ONCE for the final verdict.

SEERA played no role in development: its data values were never used to build
features, choose models, or calibrate anything (only its schema/formulas, read
by the leakage audit). This script reveals SEERA's numbers a single time — the
honest ceiling, the gate engine (GP-isolated), the baselines, the expert, and
the CQR coverage including SEERA — as an out-of-sample confirmation of the
Phase-4 conclusion on data that did not shape it.

Usage: python scripts/run_seera_verdict.py
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

import numpy as np  # noqa: F811 (kept explicit for the wrapper import below)
from sklearn.ensemble import HistGradientBoostingRegressor as _HGB
from sklearn.ensemble import RandomForestRegressor as _RF

from metis_benchmark.baselines import LogSizeRegression
from metis_benchmark.datasets.loaders import load
from metis_benchmark.evaluation import bootstrap_metric_ci, mdape, pred_at
from metis_benchmark.track_a.gbm import LogGBMQuantile
from metis_benchmark.track_a.gp import LogGaussianProcess
from metis_benchmark.track_a.w3_experiment import load_dataset


class LogSklearn:
    """Wrap a scikit-learn regressor to predict log-effort (log1p features)."""

    def __init__(self, model):
        self._m = model

    def fit(self, X, y):
        self._m.fit(np.log1p(X), np.log(y))
        return self

    def predict(self, X):
        return np.exp(self._m.predict(np.log1p(X)))


def gate_metrics() -> None:
    data = load_dataset("seera", mode="rolling")
    n = len(data.y)
    gp = np.full(n, np.nan)
    size = np.full(n, np.nan)
    for f in data.folds:
        gp[f.test] = LogGaussianProcess(seed=0).fit(f.Xtr, f.ytr).predict(f.Xte)
        size[f.test] = LogSizeRegression().fit(f.size_tr, f.ytr).predict(f.size_te)
    m = ~np.isnan(gp)
    print(f"SEERA gate engine (GP-isolated), rolling-origin, n_test={m.sum()}")
    print(f"  GP        PRED(25)={bootstrap_metric_ci(data.y[m], gp[m], pred_at, seed=0)}  "
          f"MdAPE={bootstrap_metric_ci(data.y[m], gp[m], mdape, seed=0)}")
    print(f"  log-size  PRED(25)={bootstrap_metric_ci(data.y[m], size[m], pred_at, seed=0)}")

    # Honest ceiling on SEERA (best of the flexible zoo).
    zoo = {
        "GP": lambda: LogGaussianProcess(seed=0),
        "GBM": lambda: LogGBMQuantile(seed=0),
        "RF-500": lambda: LogSklearn(_RF(n_estimators=500, random_state=0)),
        "HistGB": lambda: LogSklearn(_HGB(random_state=0)),
    }
    best = None
    for name, fac in zoo.items():
        pred = np.full(n, np.nan)
        for f in data.folds:
            pred[f.test] = fac().fit(f.Xtr, f.ytr).predict(f.Xte)
        p = pred_at(data.y[m], pred[m])
        if best is None or p > best[1]:
            best = (name, p)
    print(f"  honest ceiling (best={best[0]}) PRED(25)={best[1]:.1%}  "
          f"{'<55%' if best[1] < 0.55 else '>=55%'}")

    # Expert baseline on SEERA (Estimated effort), scored on the tested rows.
    # Replicate the harness row filtering exactly (effort, size, date) so the
    # expert array aligns with the gate predictions.
    df = load("seera").dropna(subset=["effort", "size", "date"]).reset_index(drop=True)
    df = df[df["effort"] > 0].reset_index(drop=True)
    expert = df["expert_estimate"].to_numpy(dtype=float)
    em = m & ~np.isnan(expert)
    print(f"  expert    PRED(25)={bootstrap_metric_ci(data.y[em], expert[em], pred_at, seed=0)}  "
          f"MdAPE={bootstrap_metric_ci(data.y[em], expert[em], mdape, seed=0)}  (n={em.sum()})")


if __name__ == "__main__":
    print("=== OPENING SEERA HOLDOUT (once) ===\n")
    gate_metrics()
    print("\n(coverage including SEERA: see scripts/run_coverage.py)")
