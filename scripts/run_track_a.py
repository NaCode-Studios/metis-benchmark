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
from metis_benchmark.track_a.gp import LogGaussianProcess

# Engine candidates (feature-based regressors). GBM is added in Step 2.
MODELS = {"GP (engine)": lambda: LogGaussianProcess(seed=0)}

# Statistical baselines the engine must beat (expert handled separately).
STAT_BASELINES = ["log-size regression", "median-by-category"]


def analyze(key: str) -> None:
    run = run_dataset(key, MODELS)
    print(f"\n== {key} (n_test={run.n_test}, {run.split}, features={len(run.features)}) ==")

    # Point estimate + 95% CI for every model present, on its tested rows.
    for name in run.predictions:
        y_true, y_pred = run.scored(name)
        pred_ci = bootstrap_metric_ci(y_true, y_pred, pred_at, seed=0)
        mdape_ci = bootstrap_metric_ci(y_true, y_pred, mdape, seed=0)
        print(f"  {name:<22} PRED(25)={pred_ci}  MdAPE={mdape_ci}")

    # Paired comparison: engine vs the best available statistical baseline, on
    # the same resampled rows, so we can declare a win or a tie within CI.
    engine = "GP (engine)"
    present = [b for b in STAT_BASELINES if b in run.predictions]
    # "Best" baseline = highest point PRED(25), the bar the engine must clear.
    best = max(present, key=lambda b: pred_at(*run.scored(b)))
    yb, _ = run.scored(engine)  # engine and baselines share tested rows here
    diff = paired_bootstrap_diff(
        run.y[run.tested_mask],
        run.predictions[engine][run.tested_mask],
        run.predictions[best][run.tested_mask],
        pred_at,
        seed=0,
    )
    verdict = (
        "engine BEATS baseline" if diff.low > 0
        else "engine LOSES to baseline" if diff.high < 0
        else "indistinguishable within CI"
    )
    print(f"  -> PRED(25) engine - {best}: {diff}  => {verdict}")


def analyze_aggregate(keys: list[str]) -> None:
    """Pooled engine-vs-baseline across datasets (the gate-relevant view).

    PRED(25) and the APEs behind MdAPE are unitless, so test rows from
    different datasets can be pooled even though their effort units differ.
    Pooling is exactly the aggregate-per-track judgement the protocol uses
    (Q7), and it lifts the sample size out of the tiny-per-dataset regime that
    makes single-split CIs uninformative.
    """
    yb, eng, base = [], [], []
    for key in keys:
        run = run_dataset(key, MODELS)
        m = run.tested_mask
        # Reference baseline = log-size regression (available on every dataset).
        yb.append(run.y[m])
        eng.append(run.predictions["GP (engine)"][m])
        base.append(run.predictions["log-size regression"][m])
    yb, eng, base = np.concatenate(yb), np.concatenate(eng), np.concatenate(base)

    print(f"\n== AGGREGATE Track A {keys} (n_test={len(yb)}) ==")
    for name, pred in [("GP (engine)", eng), ("log-size regression", base)]:
        pred_ci = bootstrap_metric_ci(yb, pred, pred_at, seed=0)
        mdape_ci = bootstrap_metric_ci(yb, pred, mdape, seed=0)
        print(f"  {name:<22} PRED(25)={pred_ci}  MdAPE={mdape_ci}")
    diff = paired_bootstrap_diff(yb, eng, base, pred_at, seed=0)
    verdict = (
        "engine BEATS baseline" if diff.low > 0
        else "engine LOSES to baseline" if diff.high < 0
        else "indistinguishable within CI"
    )
    print(f"  -> PRED(25) engine - log-size: {diff}  => {verdict}")


if __name__ == "__main__":
    keys = sys.argv[1:] or ["china", "desharnais", "kitchenham", "maxwell"]
    for key in keys:
        analyze(key)
    # Aggregate over the gate-carrying temporal datasets (China excluded: Q11).
    analyze_aggregate(["desharnais", "kitchenham", "maxwell"])
