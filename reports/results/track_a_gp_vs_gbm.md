# Track A — Step 2: GP vs gradient boosting (quantile), with CI95

> 2026-06-12. Both regressors: log-effort target, log1p features, same
> protocol split, **no First.estimate** (Q13). GBM = XGBoost pinball loss
> (q=0.5 point, q=0.9 reserved for Step 3). CI95 = 2000-resample bootstrap.
> Reproduce: `python scripts/run_track_a.py`.

## Engine candidates vs best statistical baseline (PRED(25))

| Dataset | n test | GP | GBM | best baseline | GP − base | GBM − base |
|---|---|---|---|---|---|---|
| desharnais | 15 | 33.3% | 33.3% | log-size 40.0% | −6.7% [−26.7, 13.3] | −13.3% [−46.7, 20.0] |
| kitchenham | 29 | 48.3% | **51.7%** | log-size 34.5% | +13.8% [0.0, 31.0] | **+17.2% [3.4, 34.5]** |
| maxwell | 12 | 50.0% | 50.0% | log-size 33.3% | +16.7% [−16.7, 50.0] | +16.7% [−16.7, 50.0] |
| **aggregate** | **56** | 44.6% | 44.6% | log-size 35.7% | +8.9% [−3.6, 21.4] | +8.9% [−5.4, 23.2] |

## Two findings

**1. Only one confirmable win so far — kitchenham with GBM.** GBM − log-size on
kitchenham is +17.2% with CI **[3.4, 34.5]**, the only paired comparison whose
95% interval clears zero. GP on the same dataset is +13.8% [0.0, 31.0] — the
lower bound sits exactly at 0, i.e. not confirmed. Maxwell and desharnais
remain indistinguishable (n=12, n=15). The aggregate still straddles 0.

**2. The 55% gate threshold is reachable — the signal exists.** In-sample
ceiling (a RandomForest fit and scored on all rows, an optimistic upper bound):
kitchenham 66%, maxwell 71%, desharnais 78% PRED(25). Unlike China (ceiling
~21%, Q11), these datasets *do* carry enough signal to clear 55%. The current
out-of-fold PRED(25) of ~45–52% versus that 66–78% ceiling is a generalization
gap, driven largely by tiny training folds (kitchenham ~116, maxwell ~50,
desharnais ~62 train rows). Closing it is a modeling/data-coverage problem, not
a dead end.

## Decision — which regressor carries the gate

**GBM carries the gate**, GP retained as the small-data alternative. Rationale:
- GBM matches or edges GP on every gate-carrying dataset and produces the only
  statistically confirmable beat-the-baseline result (kitchenham).
- GBM yields the p50/p90 quantiles natively (pinball loss), which Step 3's CQR
  conformalization consumes directly.
- The V20 thesis (GP for the smallest data) is not contradicted: GP stays as a
  fallback and the choice may flip per dataset as training folds grow.

No hyperparameters were tuned against any test block; both models use fixed,
conservative settings.

## Consequence for the gate

On point estimates the engine is still **below the 55% threshold** on every
gate-carrying dataset out-of-fold (kitchenham 51.7%, maxwell 50.0%, aggregate
44.6%). The gate is therefore **not currently passing**, but the path is open:
the ceiling shows 55% is reachable, and two levers remain — more temporal test
*and* training coverage via rolling-origin CV (Q14), and feature/model work to
close the generalization gap. This is the input the Step 6 verdict will weigh.
