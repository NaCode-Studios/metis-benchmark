"""W3 Phase 4 — honest feasibility ceiling and the pre-registered stop rule.

The W2 ceiling (a RandomForest fit AND scored on the same rows) is optimistic:
it measures memorization, not achievable accuracy. This replaces it with the
best *out-of-sample* PRED(25) over a zoo of model classes under the protocol
split — the honest answer to "how high can any model get on this data?" — plus
an irreducible-noise diagnostic (the productivity scatter that no model can
remove, in the spirit of the China Q11 analysis).

Pre-registered stop rule (protocol v1.3 sec. 8): if the honest ceiling is below
55% PRED(25) on MORE THAN ONE gate dataset, 55% is not achievable on this data
and Track A closes as "threshold unreachable, demonstrated".

SEERA is the sealed holdout and is NOT evaluated here.
Usage: python scripts/run_ceiling.py
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor

from metis_benchmark.baselines import LogSizeRegression
from metis_benchmark.evaluation import bootstrap_metric_ci, pred_at
from metis_benchmark.track_a.gbm import LogGBMQuantile
from metis_benchmark.track_a.gp import LogGaussianProcess
from metis_benchmark.track_a.w3_experiment import load_dataset

GATE_DEV = ["desharnais", "kitchenham", "maxwell", "cocomo81"]
THRESHOLD = 0.55


class LogSklearn:
    """Wrap a scikit-learn regressor to predict log-effort (like the GP/GBM)."""

    def __init__(self, model):
        self._m = model

    def fit(self, X, y):
        self._m.fit(np.log1p(X), np.log(y))
        return self

    def predict(self, X):
        return np.exp(self._m.predict(np.log1p(X)))


def model_zoo():
    # A deliberately diverse, flexible zoo so the ceiling is a genuine upper
    # bound on achievable accuracy, not a single conservative config.
    return {
        "GP": lambda: LogGaussianProcess(seed=0),
        "GBM": lambda: LogGBMQuantile(seed=0),
        "RF-500": lambda: LogSklearn(RandomForestRegressor(n_estimators=500, random_state=0)),
        "HistGB": lambda: LogSklearn(HistGradientBoostingRegressor(random_state=0)),
    }


def oof_for_model(key: str, factory) -> tuple[np.ndarray, np.ndarray]:
    """Out-of-fold (y_true, y_pred) for one model under the protocol split."""
    data = load_dataset(key, mode="rolling")
    n = len(data.y)
    pred = np.full(n, np.nan)
    for f in data.folds:
        pred[f.test] = factory().fit(f.Xtr, f.ytr).predict(f.Xte)
    mask = ~np.isnan(pred)
    return data.y[mask], pred[mask]


def size_only_residual_sigma(key: str) -> float:
    """Out-of-sample log-residual scatter after the size law ALONE.

    Compared with the best model's scatter, this shows how much the recorded
    features beyond size actually reduce the unexplained productivity variance
    (the China Q11 phenomenon, quantified)."""
    data = load_dataset(key, mode="rolling")
    resid = []
    for f in data.folds:
        reg = LogSizeRegression().fit(f.size_tr, f.ytr)
        resid.append(np.log(reg.predict(f.size_te)) - np.log(f.yte))
    return float(np.std(np.concatenate(resid)))


def main() -> None:
    print("Honest feasibility ceiling (best out-of-sample model per dataset)\n")
    below = []
    for key in GATE_DEV:
        # Best out-of-sample model = the honest ceiling: the highest PRED(25)
        # any of a diverse, flexible model zoo reaches without test peeking.
        scored = {name: oof_for_model(key, fac) for name, fac in model_zoo().items()}
        best_name = max(scored, key=lambda nm: pred_at(*scored[nm]))
        yt, yp = scored[best_name]
        ci = bootstrap_metric_ci(yt, yp, pred_at, seed=0)
        best_pred25 = pred_at(yt, yp)
        # Irreducible-noise diagnostic: unexplained log-residual scatter after
        # the best model, vs after size alone. A large best-model σ that barely
        # improves on the size-only σ means the recorded features cannot explain
        # the productivity variance — the accuracy ceiling is a data limit.
        sigma_best = float(np.std(np.log(yp) - np.log(yt)))
        sigma_size = size_only_residual_sigma(key)
        if best_pred25 < THRESHOLD:
            below.append(key)
        flag = "  <55%" if best_pred25 < THRESHOLD else "  >=55%"
        print(f"  {key:<11} best={best_name:<7} PRED(25)={ci}{flag}   "
              f"resid-σ(log): best={sigma_best:.2f} size-only={sigma_size:.2f}")

    print(f"\nDatasets with honest ceiling < 55%: {len(below)}/{len(GATE_DEV)} "
          f"({', '.join(below)})")
    if len(below) > 1:
        print("STOP RULE TRIGGERED (pre-registered): 55% is not achievable on >1 gate dataset.")
        print("Track A outcome: point-accuracy threshold unreachable on this data, demonstrated.")
    else:
        print("Stop rule NOT triggered: 55% remains reachable on the gate datasets.")


if __name__ == "__main__":
    main()
