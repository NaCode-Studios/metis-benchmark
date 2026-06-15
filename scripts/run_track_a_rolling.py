"""Rolling-origin Track A run (protocol v1.2) vs the single-split baseline.

Reports both estimators side by side, as the pre-registration requires. The
gate verdict reads the rolling-origin column; the single split stays for
transparency. Per dataset it shows GP, GBM and the pre-registered "gate
(selected)" regressor, each vs the best statistical baseline with CI95, plus
the pooled aggregate.

Usage: python scripts/run_track_a_rolling.py
"""

from __future__ import annotations

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

MODELS = {
    "GP (engine)": lambda: LogGaussianProcess(seed=0),
    "GBM (engine)": lambda: LogGBMQuantile(seed=0),
}

# Gate-carrying temporal datasets (China is dateless + out-gate per Q11).
GATE_DATASETS = ["desharnais", "kitchenham", "maxwell"]


def _verdict(diff) -> str:
    if diff.low > 0:
        return "BEATS baseline"
    if diff.high < 0:
        return "LOSES to baseline"
    return "indistinguishable"


def per_dataset(key: str) -> None:
    single = run_dataset(key, MODELS, mode="single")
    rolling = run_dataset(key, MODELS, mode="rolling")
    print(f"\n== {key} ==")
    print(f"  single  : n_test={single.n_test:3d}")
    print(f"  rolling : n_test={rolling.n_test:3d}  gate regressor = {rolling.gate_regressor}")

    for label, run in [("single", single), ("rolling", rolling)]:
        base = "log-size regression"
        # Report each engine candidate + the selected gate model where present.
        names = ["GP (engine)", "GBM (engine)"]
        if "gate (selected)" in run.predictions:
            names.append("gate (selected)")
        for name in names:
            y_t, y_p = run.scored(name)
            pci = bootstrap_metric_ci(y_t, y_p, pred_at, seed=0)
            mci = bootstrap_metric_ci(y_t, y_p, mdape, seed=0)
            m = run.tested_mask
            diff = paired_bootstrap_diff(
                run.y[m], run.predictions[name][m], run.predictions[base][m], pred_at, seed=0
            )
            print(f"  [{label:7}] {name:<16} PRED(25)={pci}  MdAPE={mci}  vs base: {diff} {_verdict(diff)}")


def aggregate(mode: str, gate_only: bool) -> None:
    """Pooled engine-vs-baseline across the gate datasets for one estimator."""
    y_pool, eng_pool, base_pool = [], [], []
    label_model = "gate (selected)" if gate_only else "GBM (engine)"
    for key in GATE_DATASETS:
        run = run_dataset(key, MODELS, mode=mode)
        m = run.tested_mask
        name = label_model if label_model in run.predictions else "GBM (engine)"
        y_pool.append(run.y[m])
        eng_pool.append(run.predictions[name][m])
        base_pool.append(run.predictions["log-size regression"][m])
    y = np.concatenate(y_pool)
    eng = np.concatenate(eng_pool)
    base = np.concatenate(base_pool)
    pci = bootstrap_metric_ci(y, eng, pred_at, seed=0)
    mci = bootstrap_metric_ci(y, eng, mdape, seed=0)
    diff = paired_bootstrap_diff(y, eng, base, pred_at, seed=0)
    tag = "gate-selected" if gate_only else "GBM"
    print(f"  [{mode:7}] AGGREGATE {tag:<13} n={len(y):3d}  PRED(25)={pci}  MdAPE={mci}  "
          f"vs base: {diff} {_verdict(diff)}")


if __name__ == "__main__":
    for key in GATE_DATASETS:
        per_dataset(key)
    print("\n== AGGREGATE (gate datasets, China excluded per Q11) ==")
    aggregate("single", gate_only=False)
    aggregate("rolling", gate_only=True)
