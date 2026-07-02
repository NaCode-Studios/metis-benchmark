# Post-G0 — TDEV/schedule law validated on real recorded durations

**Post-G0 R&D — non gate-carrying. Il verdetto G0 (NO-GO) resta valido e non è
oggetto di questo esperimento.**

Runner: `scripts/run_tdev.py` (single run, seed 0, executed on the committed
data in `data/raw/`). All numbers below are from that run.

## Pre-declared question

The product derives workforce from the COCOMO-II schedule equation
TDEV(months) = c · PM^e with literature constants c = 3.67, e = 0.30 that were
never validated. Five Track A datasets record the REAL project duration —
correctly excluded as an effort *feature* (it is an outcome: leakage) but
perfectly legitimate as a *target*. Questions:

> What (c, e) do real durations support, per dataset and pooled, with what
> uncertainty? How far off is the literature formula on real schedules
> (MdAPE in weeks)? What residual sigma should the product attach to its
> weeks band?

## Method

- **Data** (duration units verified in EDA / against the source papers):
  desharnais `Length` (months), kitchenham `Actual.duration` (**days** — it
  matches `Estimated.completion.date − Actual.start.date` with median ratio
  1.0; converted at 30.4375 days/month), maxwell `Duration` (months), china
  `Duration` (months), cocomo81 `months` (months). Effort converted to
  person-months at the product's 152 h/PM (cocomo81 is already in PM).
  **SEERA is excluded** — sealed holdout, opened once for the verdict; it
  contributes nothing here (no fit, no calibration, no measurement).
- **Model**: OLS of log(TDEV) on log(PM) (`metis_benchmark.tdev.fit_tdev`) —
  the ML fit under multiplicative log-normal noise. Reported: c, e with
  2000-rep pairs-bootstrap 95% CIs, residual sigma of log-duration (ddof=2),
  log-space R².
- **Scoring**: MdAPE of the point prediction in weeks. Per-dataset own-fit
  MdAPE is **in-sample** (declared). The pooled law is additionally scored
  **leave-one-dataset-out** (fit on the other four, score on the held-out
  one) — the honest proxy for adopting one pooled law and meeting a new
  context.

## Results

### Fitted TDEV = c · PM^e vs literature (c=3.67, e=0.30)

| dataset | n | c [95% CI] | e [95% CI] | sigma | R² | MdAPE weeks, literature | MdAPE weeks, own fit (in-sample) |
|---|---|---|---|---|---|---|---|
| desharnais | 81 | 2.20 [1.33, 3.48] | 0.466 [0.332, 0.604] | 0.501 | 0.376 | 28.7% | 29.5% |
| kitchenham | 145 | 2.37 [1.87, 3.03] | 0.370 [0.279, 0.451] | 0.494 | 0.352 | 40.1% | 27.5% |
| maxwell | 62 | 3.28 [2.40, 4.45] | 0.427 [0.338, 0.516] | 0.408 | 0.542 | 30.7% | 25.3% |
| china | 499 | 2.88 [2.52, 3.27] | 0.342 [0.296, 0.389] | 0.610 | 0.332 | 38.0% | 36.2% |
| cocomo81 | 63 | 5.15 [4.34, 6.07] | 0.264 [0.231, 0.297] | 0.245 | 0.798 | 25.5% | 13.5% |
| **pooled** | **850** | **2.71 [2.48, 2.99]** | **0.377 [0.349, 0.405]** | **0.565** | **0.462** | **34.8%** | **33.2%** |

Residuals of the **literature** law on the pooled pairs: sigma = 0.574,
mean log-bias = −0.090 (the literature formula over-predicts duration by
~9% on average, in log space).

### Pooled law, leave-one-dataset-out MdAPE (weeks)

| held-out dataset | LODO c | LODO e | MdAPE weeks, LODO pooled | MdAPE weeks, literature |
|---|---|---|---|---|
| desharnais | 2.73 | 0.372 | 32.6% | 28.7% |
| kitchenham | 2.85 | 0.370 | 33.7% | 40.1% |
| maxwell | 2.73 | 0.364 | 35.1% | 30.7% |
| china | 2.60 | 0.402 | 37.0% | 38.0% |
| cocomo81 | 2.63 | 0.386 | 25.5% | 25.5% |

Mean LODO MdAPE: pooled law 32.8%, literature 32.6% — a statistical tie on
point accuracy out of context.

### Pooled-law 90% band examples (weeks, sigma = 0.565)

| PM | point | lower | upper |
|---|---|---|---|
| 1 | 11.8 | 4.6 | 29.8 |
| 6 | 23.1 | 9.1 | 58.5 |
| 12 | 30.0 | 11.8 | 75.9 |
| 50 | 51.4 | 20.3 | 130.1 |

## Conclusion (honest)

1. **The literature pair (3.67, 0.30) is outside the data's confidence
   intervals on BOTH parameters**: the pooled fit gives c = 2.71 [2.48, 2.99]
   (literature is too high) and e = 0.377 [0.349, 0.405] (literature is too
   flat). The two errors partially compensate in the mid-PM range, which is
   why the literature's point MdAPE (34.8% pooled) is only marginally worse
   than the fitted law's (33.2% in-sample; LODO is a tie, 32.8% vs 32.6%).
2. **Per-dataset laws disagree substantially** (c from 2.20 to 5.15, e from
   0.264 to 0.466; cocomo81, the only PM-native, waterfall-era dataset, is
   the outlier that pulls toward the literature values). One universal
   schedule law explains less than half the log-variance (pooled R² = 0.462).
3. **The dominant, robust finding is the residual scale, not the
   coefficients.** Whatever law is used, schedule has sigma ≈ 0.5–0.6 in
   log-months: a point TDEV misses by ~25–40% MdAPE, and a 90% band spans
   roughly ×/÷ 2.5. Shipping a point duration without that band would repeat
   in the schedule dimension exactly the mistake G0 exposed in the effort
   dimension.

## Recommendation for the product

Adopt the **pooled fitted coefficients c = 2.71, e = 0.377 with
sigma = 0.565** (`predict_tdev_interval(pm, 2.7123, 0.3770, 0.5648)`) in
place of the unvalidated literature pair, and ALWAYS surface the 90% weeks
band, never the point alone. Rationale: point accuracy is a tie, but only the
fitted law comes with measured parameters (inside their own CIs, unlike the
literature pair) and a measured sigma; the literature pair also carries a −9%
average log-bias on these data. If a conservative rollout prefers to keep
(3.67, 0.30) temporarily, it must at minimum attach the measured
sigma = 0.574 band — the unbanded point estimate is indefensible on this
evidence. Longer term, the per-dataset spread (conclusion 2) says the
schedule law is context-dependent: fold (c, e, sigma) into the same local
recalibration path Sprint 1 built for effort, updating them from client
consuntivi as duration actuals accumulate.
