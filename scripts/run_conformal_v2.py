"""Post-G0 R&D — conditional coverage of raw vs log-space vs Mondrian CQR.

NOT gate-carrying: the G0 verdict (NO-GO) stands and is not re-litigated here.

Pre-declared question: the G0 interval applies ONE additive correction in raw
hours after exponentiation. For a log-normal target the same q hours dominates
small projects and vanishes on large ones, so marginal coverage can look fine
(94.6% at the verdict) while CONDITIONAL coverage is unbalanced: over-coverage
on small projects, under-coverage on large ones. Does calibrating and shifting
in log space equalize per-tercile coverage without losing the marginal one?
And does Mondrian (per-tercile) calibration add anything at gate-sized
calibration sets?

Setup (mirrors scripts/run_coverage.py, the Step-3 machinery):
- Datasets: the Track A gate datasets desharnais, kitchenham, maxwell, cocomo81.
  SEERA is the SEALED holdout, opened once for the verdict — it is explicitly
  EXCLUDED from this and every post-G0 experiment.
- Splits: the protocol-assigned ones — rolling-origin CV for the dated
  datasets, ordered 5-fold CV for the dateless cocomo81 — via the shared W3
  harness (same encoding, same train-median imputation). The calibration block
  is re-derived per fold as the most recent 20% slice of the training window,
  exactly as in run_coverage.py.
- Base quantiles: LogGBMQuantile p5/p95 (seed 0), identical for every variant,
  so the ONLY difference between variants is the conformal step.
- Variants: raw-marginal (the G0 baseline), log-marginal, log-Mondrian
  (training log-size terciles, min_group_cal=15, marginal fallback).
- Measured: marginal coverage, conditional coverage per training-derived size
  tercile, and the median interval width as the upper/lower ratio per tercile,
  each with a 2000-rep percentile-bootstrap 95% CI (seed 0).

Usage: python scripts/run_conformal_v2.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metis_benchmark.conformal import (
    ConformalizedQuantile,
    MondrianConformalizedQuantile,
    assign_terciles,
    log_size_tercile_edges,
)
from metis_benchmark.track_a.gbm import LogGBMQuantile
from metis_benchmark.track_a.w3_experiment import load_dataset

# Track A gate datasets. SEERA is deliberately absent: sealed holdout.
GATE_DATASETS = ["desharnais", "kitchenham", "maxwell", "cocomo81"]
ALPHA = 0.10  # nominal 90%, the gate level
QUANTILES = (0.05, 0.95)  # base p5/p95 pair for the 90% interval
GROUP_NAMES = {0: "small", 1: "medium", 2: "large"}
N_BOOT = 2000
SEED = 0


def collect_records(min_group_cal: int = 15) -> tuple[pd.DataFrame, dict[str, int]]:
    """One row per (variant, test point): hit, tercile, width ratio.

    Also returns the Mondrian fallback audit: how many (fold, tercile) cells
    self-calibrated vs fell back to the marginal correction — the honest
    measure of whether Mondrian can bite at gate calibration sizes.
    """
    rows: list[dict] = []
    audit = {"cells": 0, "fallback_cells": 0, "self_calibrated_cells": 0}
    for key in GATE_DATASETS:
        data = load_dataset(key, mode="rolling")
        for f in data.folds:
            # Re-derive the calibration block: most recent 20% of the training
            # window (identical to run_coverage.py, the Step-3 convention).
            n_cal = max(1, int(round(len(f.Xtr) * 0.2)))
            X_fit, X_cal = f.Xtr[:-n_cal], f.Xtr[-n_cal:]
            y_fit, y_cal = f.ytr[:-n_cal], f.ytr[-n_cal:]
            size_cal, size_te = f.size_tr[-n_cal:], f.size_te

            # One base model per fold; every variant consumes the SAME quantiles.
            model = LogGBMQuantile(quantiles=QUANTILES, seed=0).fit(X_fit, y_fit)
            qcal = model.predict_quantiles(X_cal)
            qte = model.predict_quantiles(f.Xte)
            lo_cal, hi_cal = qcal[QUANTILES[0]], qcal[QUANTILES[1]]
            lo_te, hi_te = qte[QUANTILES[0]], qte[QUANTILES[1]]

            # Mondrian partition: terciles of TRAINING log-size (fit+cal window),
            # fixed before any calibration/test label is used.
            edges = log_size_tercile_edges(f.size_tr)
            g_cal = assign_terciles(size_cal, edges)
            g_te = assign_terciles(size_te, edges)

            intervals: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            # Variant 1 — the G0 baseline: one additive correction in raw hours.
            raw = ConformalizedQuantile(alpha=ALPHA).calibrate(lo_cal, hi_cal, y_cal)
            intervals["raw-marginal"] = raw.interval(lo_te, hi_te)
            # Variant 2 — same marginal calibration, but in log space.
            logm = ConformalizedQuantile(alpha=ALPHA, space="log").calibrate(
                lo_cal, hi_cal, y_cal
            )
            intervals["log-marginal"] = logm.interval(lo_te, hi_te)
            # Variant 3 — log space + per-tercile (Mondrian) calibration.
            mond = MondrianConformalizedQuantile(
                alpha=ALPHA, space="log", min_group_cal=min_group_cal
            ).calibrate(lo_cal, hi_cal, y_cal, g_cal)
            intervals["log-Mondrian"] = mond.interval(lo_te, hi_te, g_te)

            # Fallback audit over the tercile cells present in this fold's
            # calibration block.
            for g in np.unique(g_cal):
                audit["cells"] += 1
                if int(g) in mond.fallback_groups_:
                    audit["fallback_cells"] += 1
                else:
                    audit["self_calibrated_cells"] += 1

            for variant, (lo, hi) in intervals.items():
                hit = (f.yte >= lo) & (f.yte <= hi)
                for i in range(len(f.yte)):
                    rows.append({
                        "dataset": key,
                        "variant": variant,
                        "tercile": GROUP_NAMES[int(g_te[i])],
                        "hit": bool(hit[i]),
                        "lower": float(lo[i]),
                        "upper": float(hi[i]),
                        # Width as the upper/lower ratio: scale-free, so it can
                        # be pooled across datasets with different effort units.
                        # Undefined (NaN) when the raw shift pushes lower <= 0.
                        "ratio": float(hi[i] / lo[i]) if lo[i] > 0 else np.nan,
                        "lower_nonpos": bool(lo[i] <= 0),
                    })
    return pd.DataFrame(rows), audit


def boot_ci(values: np.ndarray, stat, seed: int = SEED) -> tuple[float, float, float]:
    """Percentile-bootstrap 95% CI of `stat` over test points (2000 reps)."""
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    n = len(values)
    stats = np.empty(N_BOOT)
    for b in range(N_BOOT):
        stats[b] = stat(values[rng.integers(0, n, n)])
    lo, hi = np.percentile(stats, [2.5, 97.5])
    return float(stat(values)), float(lo), float(hi)


def fmt_cov(values: np.ndarray) -> str:
    p, lo, hi = boot_ci(values, np.mean)
    return f"{p:.1%} [{lo:.1%}, {hi:.1%}]"


def fmt_ratio(values: np.ndarray) -> str:
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return "n/a"
    p, lo, hi = boot_ci(values, np.median)
    return f"{p:.2f} [{lo:.2f}, {hi:.2f}]"


def main() -> None:
    df, audit = collect_records()
    variants = ["raw-marginal", "log-marginal", "log-Mondrian"]

    n_points = len(df) // len(variants)
    print(f"pooled test points: {n_points} "
          f"(datasets: {', '.join(GATE_DATASETS)}; SEERA excluded — sealed holdout)\n")

    # ---- marginal coverage, pooled and per dataset ----
    print("## Marginal coverage (nominal 90%)\n")
    print("| variant | pooled |", " | ".join(GATE_DATASETS), "|")
    print("|---|---|", "|".join(["---"] * len(GATE_DATASETS)), "|")
    for v in variants:
        sub = df[df.variant == v]
        cells = [fmt_cov(sub.hit.to_numpy())]
        for key in GATE_DATASETS:
            cells.append(fmt_cov(sub[sub.dataset == key].hit.to_numpy()))
        print(f"| {v} | " + " | ".join(cells) + " |")

    # ---- conditional coverage per training-derived size tercile ----
    print("\n## Conditional coverage by size tercile (pooled)\n")
    print("| variant | small | medium | large |")
    print("|---|---|---|---|")
    for v in variants:
        sub = df[df.variant == v]
        cells = [fmt_cov(sub[sub.tercile == t].hit.to_numpy())
                 for t in ("small", "medium", "large")]
        print(f"| {v} | " + " | ".join(cells) + " |")
    counts = df[df.variant == variants[0]].tercile.value_counts()
    print(f"\nn per tercile: small={counts.get('small', 0)}, "
          f"medium={counts.get('medium', 0)}, large={counts.get('large', 0)}")

    # ---- median interval width (upper/lower ratio) per tercile ----
    print("\n## Median interval width, upper/lower ratio (pooled)\n")
    print("| variant | small | medium | large | % lower<=0 |")
    print("|---|---|---|---|---|")
    for v in variants:
        sub = df[df.variant == v]
        cells = [fmt_ratio(sub[sub.tercile == t].ratio.to_numpy())
                 for t in ("small", "medium", "large")]
        nonpos = sub.lower_nonpos.mean()
        print(f"| {v} | " + " | ".join(cells) + f" | {nonpos:.1%} |")

    # ---- Mondrian fallback audit ----
    print("\n## Mondrian fallback audit (min_group_cal=15)\n")
    print(f"tercile calibration cells: {audit['cells']}, "
          f"self-calibrated: {audit['self_calibrated_cells']}, "
          f"fell back to marginal: {audit['fallback_cells']} "
          f"({audit['fallback_cells'] / audit['cells']:.0%})")

    # ---- sensitivity: does Mondrian differ at an exploratory lower threshold? ----
    # min_group_cal=8 is BELOW the finite-sample-meaningful size at alpha=0.10
    # (documented in conformal.py); this run is diagnostic only, to show what
    # per-group calibration would do if allowed to engage on tiny groups.
    df8, audit8 = collect_records(min_group_cal=8)
    sub8 = df8[df8.variant == "log-Mondrian"]
    print("\n## Sensitivity (diagnostic only): log-Mondrian with min_group_cal=8\n")
    print(f"fallback cells: {audit8['fallback_cells']}/{audit8['cells']} "
          f"({audit8['fallback_cells'] / audit8['cells']:.0%})")
    print("| tercile | coverage | median ratio |")
    print("|---|---|---|")
    for t in ("small", "medium", "large"):
        s = sub8[sub8.tercile == t]
        print(f"| {t} | {fmt_cov(s.hit.to_numpy())} | {fmt_ratio(s.ratio.to_numpy())} |")
    print(f"| marginal | {fmt_cov(sub8.hit.to_numpy())} | — |")


if __name__ == "__main__":
    main()
