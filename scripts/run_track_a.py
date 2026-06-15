"""Track A engine-vs-baseline run with bootstrap confidence intervals.

For each dataset: out-of-fold PRED(25)/MdAPE per model with a 95% bootstrap CI,
plus a paired bootstrap of the engine-minus-best-baseline difference to state
whether the win holds or the models are indistinguishable within the data.

Usage: python scripts/run_track_a.py [china desharnais kitchenham maxwell]
"""

from __future__ import annotations

import sys

import numpy as np

from metis_benchmark.evaluation import (
    bootstrap_metric_ci,
    mdape,
    paired_bootstrap_diff,
    pred_at,
)
from metis_benchmark.track_a.experiment import run_dataset
from metis_benchmark.track_a.gbm import LogGBMQuantile
from metis_benchmark.track_a.gp import LogGaussianProcess

# Engine candidates (feature-based regressors): GP and GBM compete per dataset.
MODELS = {
    "GP (engine)": lambda: LogGaussianProcess(seed=0),
    "GBM (engine)": lambda: LogGBMQuantile(seed=0),
}

# Statistical baselines the engine must beat (expert handled separately).
STAT_BASELINES = ["log-size regression", "median-by-category"]


def _verdict(diff) -> str:
    # Read the paired-difference CI: a win requires the interval to clear 0.
    if diff.low > 0:
        return "engine BEATS baseline"
    if diff.high < 0:
        return "engine LOSES to baseline"
    return "indistinguishable within CI"


def analyze(key: str) -> None:
    run = run_dataset(key, MODELS)
    print(f"\n== {key} (n_test={run.n_test}, {run.split}, features={len(run.features)}) ==")

    # Point estimate + 95% CI for every model present, on its tested rows.
    for name in run.predictions:
        y_true, y_pred = run.scored(name)
        pred_ci = bootstrap_metric_ci(y_true, y_pred, pred_at, seed=0)
        mdape_ci = bootstrap_metric_ci(y_true, y_pred, mdape, seed=0)
        print(f"  {name:<22} PRED(25)={pred_ci}  MdAPE={mdape_ci}")

    # Paired comparisons on the same resampled rows: each engine candidate vs
    # the best statistical baseline, so we can declare a win or a tie within CI.
    present = [b for b in STAT_BASELINES if b in run.predictions]
    best = max(present, key=lambda b: pred_at(*run.scored(b)))
    m = run.tested_mask
    for engine in MODELS:
        diff = paired_bootstrap_diff(
            run.y[m], run.predictions[engine][m], run.predictions[best][m], pred_at, seed=0
        )
        print(f"  -> {engine} − {best}: {diff}  => {_verdict(diff)}")


def analyze_aggregate(keys: list[str]) -> None:
    """Pooled engine-vs-baseline across datasets (the gate-relevant view).

    PRED(25) and the APEs behind MdAPE are unitless, so test rows from
    different datasets can be pooled even though their effort units differ.
    Pooling is exactly the aggregate-per-track judgement the protocol uses
    (Q7), and it lifts the sample size out of the tiny-per-dataset regime that
    makes single-split CIs uninformative.
    """
    # Pool tested rows across datasets for each model + the reference baseline.
    pooled: dict[str, list[np.ndarray]] = {name: [] for name in MODELS}
    pooled["log-size regression"] = []
    y_pool: list[np.ndarray] = []
    for key in keys:
        run = run_dataset(key, MODELS)
        m = run.tested_mask
        y_pool.append(run.y[m])
        for name in pooled:
            pooled[name].append(run.predictions[name][m])
    yb = np.concatenate(y_pool)
    cols = {name: np.concatenate(parts) for name, parts in pooled.items()}

    print(f"\n== AGGREGATE Track A {keys} (n_test={len(yb)}) ==")
    for name, pred in cols.items():
        pred_ci = bootstrap_metric_ci(yb, pred, pred_at, seed=0)
        mdape_ci = bootstrap_metric_ci(yb, pred, mdape, seed=0)
        print(f"  {name:<22} PRED(25)={pred_ci}  MdAPE={mdape_ci}")
    base = cols["log-size regression"]
    for engine in MODELS:
        diff = paired_bootstrap_diff(yb, cols[engine], base, pred_at, seed=0)
        print(f"  -> {engine} − log-size: {diff}  => {_verdict(diff)}")


if __name__ == "__main__":
    keys = sys.argv[1:] or ["china", "desharnais", "kitchenham", "maxwell"]
    for key in keys:
        analyze(key)
    # Aggregate over the gate-carrying temporal datasets (China excluded: Q11).
    analyze_aggregate(["desharnais", "kitchenham", "maxwell"])
