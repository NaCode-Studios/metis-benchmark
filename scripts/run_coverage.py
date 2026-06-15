"""Step 3 — conformalized interval coverage under rolling-origin CV.

The gate asks not only for point accuracy but for an interval that keeps its
promise: a nominal-90% interval must contain the true effort within 5 points of
90% on unseen projects. This script conformalizes the GBM quantiles (CQR) per
rolling-origin fold, pools the out-of-fold test rows, and measures empirical
coverage with a binomial 95% CI. It also sweeps nominal levels to draw the
calibration plot.

Coverage is measured on the rolling-origin folds (not the single split): on
12-29 single-split test rows the binomial CI on coverage is ~+-13 points, so
the "within 5 points" criterion is unmeasurable there; the pooled rolling-origin
test set makes it readable.

Usage: python scripts/run_coverage.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from metis_benchmark.conformal import ConformalizedQuantile
from metis_benchmark.track_a.gbm import LogGBMQuantile
from metis_benchmark.track_a.w3_experiment import load_dataset

# W3 verdict: SEERA holdout added; cocomo81 included (dateless -> ordered k-fold
# via its rolling_origin fallback in the harness is not valid, so coverage is
# measured on the dated datasets that support a calibration block).
GATE_DATASETS = ["desharnais", "kitchenham", "maxwell", "seera"]
# Nominal coverages to sweep; 0.90 is the gate level. Each maps to a lower/upper
# quantile pair (alpha/2, 1-alpha/2).
NOMINAL_LEVELS = [0.50, 0.70, 0.80, 0.90, 0.95]
# Every quantile any level needs, fit jointly by one GBM.
QUANTILES = sorted({round((1 - lv) / 2, 4) for lv in NOMINAL_LEVELS}
                   | {round(1 - (1 - lv) / 2, 4) for lv in NOMINAL_LEVELS})


def _pair(level: float) -> tuple[float, float]:
    a = round((1 - level) / 2, 4)
    return a, round(1 - a, 4)


def coverage_for_level(level: float) -> tuple[float, int]:
    """Pooled CQR coverage at one nominal level across the gate datasets.

    Uses the shared harness folds (encoding + train-median imputation), so the
    calibration block of each rolling-origin fold feeds CQR consistently.
    """
    lo_q, hi_q = _pair(level)
    captured: list[np.ndarray] = []
    for key in GATE_DATASETS:
        data = load_dataset(key, mode="rolling")
        # Re-derive the calibration block per fold: the harness pools train+cal
        # into Fold.train, so refit calibration as the most recent slice of it.
        for f in data.folds:
            n_cal = max(1, int(round(len(f.Xtr) * 0.2)))
            X_fit, X_cal, X_te = f.Xtr[:-n_cal], f.Xtr[-n_cal:], f.Xte
            y_fit, y_cal, y_te = f.ytr[:-n_cal], f.ytr[-n_cal:], f.yte
            model = LogGBMQuantile(quantiles=tuple(QUANTILES), seed=0).fit(X_fit, y_fit)
            qcal = model.predict_quantiles(X_cal)
            qte = model.predict_quantiles(X_te)
            cqr = ConformalizedQuantile(alpha=1 - level).calibrate(qcal[lo_q], qcal[hi_q], y_cal)
            lo, hi = cqr.interval(qte[lo_q], qte[hi_q])
            captured.append(((y_te >= lo) & (y_te <= hi)).astype(float))
    pooled = np.concatenate(captured)
    return float(pooled.mean()), len(pooled)


def binom_ci(p: float, n: int) -> tuple[float, float]:
    # Normal-approximation 95% binomial CI on the coverage proportion.
    se = (p * (1 - p) / n) ** 0.5
    return max(0.0, p - 1.96 * se), min(1.0, p + 1.96 * se)


def main() -> None:
    rows = []
    for level in NOMINAL_LEVELS:
        emp, n = coverage_for_level(level)
        lo, hi = binom_ci(emp, n)
        rows.append((level, emp, lo, hi, n))
        flag = "  <- GATE" if level == 0.90 else ""
        print(f"  nominal {level:.0%}  empirical {emp:.1%}  CI95 [{lo:.1%}, {hi:.1%}]  n={n}{flag}")

    # Calibration plot: nominal vs empirical with the ideal diagonal.
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nominal = [r[0] for r in rows]
    empirical = [r[1] for r in rows]
    yerr = [[r[1] - r[2] for r in rows], [r[3] - r[1] for r in rows]]
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "--", color="gray", label="ideal")
    ax.errorbar(nominal, empirical, yerr=yerr, marker="o", capsize=3, label="CQR (rolling-origin)")
    ax.axhspan(0.85, 0.95, color="green", alpha=0.08)  # gate tolerance band at 90%
    ax.set_xlabel("nominal coverage")
    ax.set_ylabel("empirical coverage (pooled, gate datasets)")
    ax.set_title("Track A — CQR calibration (rolling-origin)")
    ax.legend(loc="upper left")
    out = Path(__file__).resolve().parents[1] / "reports" / "results" / "calibration_track_a.png"
    fig.savefig(out, dpi=120, bbox_inches="tight")
    print(f"\ncalibration plot -> {out}")


if __name__ == "__main__":
    main()
