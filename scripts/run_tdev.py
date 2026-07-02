"""Post-G0 R&D — validate the TDEV/schedule law on datasets with REAL durations.

NOT gate-carrying: the G0 verdict (NO-GO) stands and is not re-litigated here.

The product derives workforce from TDEV = c * PM^e with literature constants
(c=3.67, e=0.30) never validated on data. Several Track A datasets record the
REAL project duration — excluded as an effort feature (leakage: it is an
outcome) but perfectly legitimate as a TARGET here. Pre-declared questions:

> What (c, e) do the real durations support, per dataset and pooled? How far
> off is the literature formula on real schedules (MdAPE in weeks)? What
> residual sigma should the product attach to its weeks band?

Datasets and duration units (verified in EDA / against the source papers):
- desharnais: Length, months; effort person-hours -> PM = hours/152
- kitchenham: Actual.duration, DAYS (matches Estimated.completion.date minus
  Actual.start.date; median ratio 1.0) -> months = days/30.4375
- maxwell:    Duration, months; effort person-hours
- china:      Duration, months; effort person-hours
- cocomo81:   months (realized schedule), months; effort already person-months
SEERA is the SEALED holdout, opened once for the G0 verdict: it is explicitly
EXCLUDED from this experiment (no fit, no calibration, no measurement).

Honesty notes: per-dataset MdAPE of the fitted law is IN-SAMPLE (declared as
such); the pooled law is additionally scored LEAVE-ONE-DATASET-OUT — fit on
the other four, scored on the held-out dataset — which is the honest proxy for
"adopt one pooled law in the product and meet a new context".

Usage: python scripts/run_tdev.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from metis_benchmark.datasets.loaders import load
from metis_benchmark.evaluation import mdape
from metis_benchmark.tdev import (
    HOURS_PER_PERSON_MONTH,
    LITERATURE_C,
    LITERATURE_E,
    WEEKS_PER_MONTH,
    fit_tdev,
    predict_tdev_interval,
)

# Average calendar days per month (365.25 / 12): converts kitchenham's
# day-denominated durations to months.
DAYS_PER_MONTH = 30.4375

# dataset key -> (duration column, factor converting that column to months,
#                 factor converting recorded effort to person-months)
DATASETS: dict[str, tuple[str, float, float]] = {
    "desharnais": ("Length", 1.0, 1.0 / HOURS_PER_PERSON_MONTH),
    "kitchenham": ("Actual.duration", 1.0 / DAYS_PER_MONTH, 1.0 / HOURS_PER_PERSON_MONTH),
    "maxwell": ("Duration", 1.0, 1.0 / HOURS_PER_PERSON_MONTH),
    "china": ("Duration", 1.0, 1.0 / HOURS_PER_PERSON_MONTH),
    "cocomo81": ("months", 1.0, 1.0),  # effort already in person-months
}


def load_pairs(key: str) -> tuple[np.ndarray, np.ndarray]:
    """(PM, TDEV months) for one dataset, positive pairs only."""
    col, dur_to_months, eff_to_pm = DATASETS[key]
    df = load(key)
    dur = pd.to_numeric(df[col], errors="coerce") * dur_to_months
    pm = pd.to_numeric(df["effort"], errors="coerce") * eff_to_pm
    keep = dur.notna() & pm.notna() & (dur > 0) & (pm > 0)
    return pm[keep].to_numpy(dtype=float), dur[keep].to_numpy(dtype=float)


def mdape_weeks(pm: np.ndarray, tdev_months: np.ndarray, c: float, e: float) -> float:
    """MdAPE of the law's point prediction, computed on weeks."""
    pred_weeks, _, _ = predict_tdev_interval(pm, c, e, sigma=0.0)
    return mdape(tdev_months * WEEKS_PER_MONTH, pred_weeks)


def main() -> None:
    pairs = {key: load_pairs(key) for key in DATASETS}

    # ---- per-dataset and pooled fits ----
    print("## Fitted TDEV = c * PM^e vs literature (c=3.67, e=0.30)\n")
    print("| dataset | n | c [95% CI] | e [95% CI] | sigma | R^2 |"
          " MdAPE weeks, literature | MdAPE weeks, own fit (in-sample) |")
    print("|---|---|---|---|---|---|---|---|")
    for key, (pm, dur) in pairs.items():
        fit = fit_tdev(pm, dur, n_boot=2000, seed=0)
        lit = mdape_weeks(pm, dur, LITERATURE_C, LITERATURE_E)
        own = mdape_weeks(pm, dur, fit.c, fit.e)
        print(f"| {key} | {fit.n} | {fit.c:.2f} [{fit.c_ci[0]:.2f}, {fit.c_ci[1]:.2f}] "
              f"| {fit.e:.3f} [{fit.e_ci[0]:.3f}, {fit.e_ci[1]:.3f}] "
              f"| {fit.sigma:.3f} | {fit.r2:.3f} | {lit:.1%} | {own:.1%} |")

    pm_all = np.concatenate([p for p, _ in pairs.values()])
    dur_all = np.concatenate([d for _, d in pairs.values()])
    pooled = fit_tdev(pm_all, dur_all, n_boot=2000, seed=0)
    lit_all = mdape_weeks(pm_all, dur_all, LITERATURE_C, LITERATURE_E)
    own_all = mdape_weeks(pm_all, dur_all, pooled.c, pooled.e)
    print(f"| pooled | {pooled.n} | {pooled.c:.2f} [{pooled.c_ci[0]:.2f}, {pooled.c_ci[1]:.2f}] "
          f"| {pooled.e:.3f} [{pooled.e_ci[0]:.3f}, {pooled.e_ci[1]:.3f}] "
          f"| {pooled.sigma:.3f} | {pooled.r2:.3f} | {lit_all:.1%} | {own_all:.1%} |")

    # ---- pooled law scored leave-one-dataset-out (honest transfer measure) ----
    print("\n## Pooled law, leave-one-dataset-out MdAPE (weeks)\n")
    print("| held-out dataset | LODO c | LODO e | MdAPE weeks, LODO pooled |"
          " MdAPE weeks, literature |")
    print("|---|---|---|---|---|")
    for key in DATASETS:
        pm_rest = np.concatenate([p for k, (p, _) in pairs.items() if k != key])
        dur_rest = np.concatenate([d for k, (_, d) in pairs.items() if k != key])
        f = fit_tdev(pm_rest, dur_rest, n_boot=200, seed=0)
        pm_te, dur_te = pairs[key]
        print(f"| {key} | {f.c:.2f} | {f.e:.3f} "
              f"| {mdape_weeks(pm_te, dur_te, f.c, f.e):.1%} "
              f"| {mdape_weeks(pm_te, dur_te, LITERATURE_C, LITERATURE_E):.1%} |")

    # ---- what the recommended band looks like in practice ----
    print("\n## Pooled-law 90% band examples (weeks)\n")
    print("| PM | point | lower | upper |")
    print("|---|---|---|---|")
    for pm_ex in (1.0, 6.0, 12.0, 50.0):
        pt, lo, hi = predict_tdev_interval(pm_ex, pooled.c, pooled.e, pooled.sigma)
        print(f"| {pm_ex:.0f} | {float(pt):.1f} | {float(lo):.1f} | {float(hi):.1f} |")
    print(f"\npooled coefficients: c={pooled.c:.4f}, e={pooled.e:.4f}, "
          f"sigma={pooled.sigma:.4f} (log-months residual std)")

    # Residual sigma AROUND THE LITERATURE LAW: the honest band the product
    # would need if it kept (3.67, 0.30). Measured on the same pooled pairs;
    # ddof=0 because no parameter was estimated from this data.
    resid_lit = np.log(dur_all) - np.log(LITERATURE_C * pm_all**LITERATURE_E)
    print(f"literature-law residuals (pooled): sigma={resid_lit.std():.4f}, "
          f"mean log-bias={resid_lit.mean():+.4f} "
          f"(negative = literature over-predicts duration on average)")


if __name__ == "__main__":
    main()
