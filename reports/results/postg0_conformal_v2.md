# Post-G0 — Conformal in log space + Mondrian: conditional coverage measured

**Post-G0 R&D — non gate-carrying. Il verdetto G0 (NO-GO) resta valido e non è
oggetto di questo esperimento.**

Runner: `scripts/run_conformal_v2.py` (single run, seed 0, executed on the
committed data in `data/raw/`). All numbers below are from that run.

## Pre-declared question

The G0 interval (Step 3, `ConformalizedQuantile`) applies **one additive
correction in raw hours** after exponentiating the GBM's log-space quantiles.
For a log-normal target the same `q` hours dominates small projects and
vanishes on large ones. Marginal coverage at the verdict looked healthy
(94.6%), but marginal coverage can hide a conditional imbalance. Question:

> Does the raw method over-cover small projects and under-cover large ones?
> Does calibrating and shifting in **log space** equalize conditional coverage
> without losing the marginal one? Does **Mondrian** (per-size-tercile)
> calibration add anything at gate-sized calibration sets?

## Method

- **Datasets**: the Track A gate datasets — desharnais, kitchenham, maxwell,
  cocomo81. **SEERA is excluded**: it is the sealed holdout, opened exactly
  once for the G0 verdict, and is not used for any fit, calibration or
  measurement in this experiment.
- **Splits**: the protocol-assigned ones, unchanged — rolling-origin CV for
  the dated datasets, ordered 5-fold CV for the dateless cocomo81 — via the
  shared W3 harness (same encoding, same train-median imputation). The
  calibration block is the most recent 20% slice of each fold's training
  window, exactly as in `scripts/run_coverage.py`. Caveat: cocomo81's ordered
  k-fold has no temporal guarantee (protocol fallback, declared as always).
- **Base quantiles**: `LogGBMQuantile` p5/p95, seed 0, identical for every
  variant — the only difference between variants is the conformal step.
- **Variants** at nominal 90% (alpha = 0.10):
  1. `raw-marginal` — the G0 baseline: `ConformalizedQuantile(alpha=0.10)`.
  2. `log-marginal` — `ConformalizedQuantile(alpha=0.10, space="log")`:
     scores on log(y), shift applied in log space (a uniform multiplicative
     correction in hours).
  3. `log-Mondrian` — `MondrianConformalizedQuantile(alpha=0.10,
     space="log", min_group_cal=15)`: one correction per **training**
     log-size tercile, with an explicit fallback to the marginal correction
     when a tercile has fewer than 15 calibration points (at alpha = 0.10,
     n_cal = 15 is the smallest group where the finite-sample quantile is not
     simply the sample maximum).
- **Measured**: marginal coverage; conditional coverage per training-derived
  size tercile; median interval width as the upper/lower ratio (scale-free,
  poolable across effort units); 2000-rep percentile-bootstrap 95% CIs over
  test points (seed 0). Pooled test points: **207** (small=72, medium=58,
  large=77).

## Results

### Marginal coverage (nominal 90%)

| variant | pooled | desharnais | kitchenham | maxwell | cocomo81 |
|---|---|---|---|---|---|
| raw-marginal | 94.7% [91.3%, 97.6%] | 92.7% [82.9%, 100.0%] | 95.9% [90.4%, 100.0%] | 93.5% [83.9%, 100.0%] | 95.2% [88.7%, 100.0%] |
| log-marginal | 94.2% [90.8%, 97.1%] | 87.8% [78.0%, 97.6%] | 95.9% [90.4%, 100.0%] | 93.5% [83.9%, 100.0%] | 96.8% [91.9%, 100.0%] |
| log-Mondrian | 94.2% [90.8%, 97.1%] | 87.8% [78.0%, 97.6%] | 95.9% [90.4%, 100.0%] | 93.5% [83.9%, 100.0%] | 96.8% [91.9%, 100.0%] |

### Conditional coverage by size tercile (pooled)

| variant | small | medium | large |
|---|---|---|---|
| raw-marginal | 98.6% [95.8%, 100.0%] | 100.0% [100.0%, 100.0%] | **87.0% [79.2%, 93.5%]** |
| log-marginal | 90.3% [83.3%, 95.8%] | 98.3% [94.8%, 100.0%] | 94.8% [89.6%, 98.7%] |
| log-Mondrian | 90.3% [83.3%, 95.8%] | 98.3% [94.8%, 100.0%] | 94.8% [89.6%, 98.7%] |

### Median interval width, upper/lower ratio (pooled)

| variant | small | medium | large | % of test points with lower ≤ 0 |
|---|---|---|---|---|
| raw-marginal | 19.16 [8.21, 63.24] | 15.61 [10.32, 58.20] | 11.52 [9.06, 14.96] | **84.1%** |
| log-marginal | 53.64 [35.44, 93.80] | 92.48 [41.60, 125.99] | 52.03 [33.88, 57.35] | 0.0% |
| log-Mondrian | 53.64 [35.44, 93.80] | 92.48 [41.60, 125.99] | 52.03 [33.88, 57.35] | 0.0% |

The raw ratio is computed only on the 15.9% of test points whose lower bound
stayed positive, so the raw and log columns are **not** directly comparable:
on 84.1% of test points the raw additive shift pushes the lower bound to or
below zero hours — an uninformative (and physically meaningless) lower bound,
which is itself the strongest width finding of the run.

### Mondrian fallback audit (min_group_cal = 15)

Tercile calibration cells across all folds: 57. Self-calibrated: **0**. Fell
back to the marginal correction: **57 (100%)**. At gate calibration sizes
(10–26 points per fold, so 3–9 per tercile) the per-group finite-sample
guarantee can never engage; log-Mondrian therefore degenerates — by its own
explicit fallback rule — into log-marginal, which is why the two rows are
identical everywhere above.

Sensitivity (diagnostic only, below the meaningful finite-sample size):
`min_group_cal=8` lets 7/57 cells self-calibrate (fallback 88%) and moves
small-tercile coverage from 90.3% to 88.9% [81.9%, 95.8%], marginal to 93.7%
[90.3%, 96.6%] — no measurable improvement, more variance.

## Conclusion (honest)

1. **The pre-declared suspicion is confirmed.** The raw-hours correction
   over-covers small and medium projects (98.6% and 100.0% at nominal 90%)
   and under-covers large ones (87.0%, CI [79.2%, 93.5%]). The marginal
   94.7% is an average of two miscalibrations, not a calibrated interval at
   every scale.
2. **Log space equalizes without losing the marginal.** Conditional coverage
   becomes 90.3% / 98.3% / 94.8% (small/medium/large) — every tercile's CI
   now contains or exceeds the nominal 90% — while pooled marginal coverage
   stays at 94.2% [90.8%, 97.1%], statistically indistinguishable from the
   raw baseline's 94.7%.
3. **Log space also fixes the lower bound.** The raw method emits a
   non-positive lower bound on 84.1% of test points ("between −2 000 and
   9 000 hours" carries no lower information); the multiplicative shift keeps
   the lower bound positive by construction, at the cost of honest — wide —
   ratios (median upper/lower ≈ 52–92 pooled). The width is a property of the
   base p5/p95 on tiny cold-start data, not of the conformal step.
4. **Mondrian is not viable at gate calibration sizes.** With the documented
   threshold (15 points per tercile) the fallback engaged on 100% of cells;
   forcing it below the meaningful size (8) does not improve any measured
   quantity. Per-tercile calibration only becomes actionable when a
   calibration set reaches ≈ 45+ points (15 per tercile).

## Recommendation for the product

Adopt **log-marginal CQR** (`ConformalizedQuantile(space="log")`) as the
interval method in metis-core/metis-api: it repairs the large-project
under-coverage (87.0% → 94.8%), keeps the marginal promise (94.2%), and
guarantees a positive lower bound in hours. Keep `space="raw"` as the default
in this benchmark so `make reproduce-g0` still reproduces the frozen verdict.
Wire `MondrianConformalizedQuantile` only behind the calibration-size gate it
already enforces: with the product's local recalibration accumulating client
consuntivi, switch a client to per-tercile calibration once their calibration
history holds ≥ 15 points in every tercile — never before.
