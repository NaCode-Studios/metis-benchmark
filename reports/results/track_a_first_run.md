# Track A — first engine run (GP vs baselines) with bootstrap CI95

> 2026-06-12. Base Gaussian Process (RBF on log1p features, log-effort target),
> no conformal intervals and no gradient boosting yet. Out-of-fold predictions
> under the protocol-assigned split. CI95 = 2000-resample percentile bootstrap
> over the test rows. Reproduce: `python scripts/run_track_a.py`.
> Every metric is logged with its split and test count (the W2 logging rule).

## Per-dataset (PRED(25) and MdAPE with 95% CI)

| Dataset | Split | n test | feats | GP PRED(25) | best stat. baseline PRED(25) | engine − baseline (paired) | Expert PRED(25) |
|---|---|---|---|---|---|---|---|
| china | ordered 5-fold | 499 | 9 | 19.2% [15.6, 22.8] | log-size 22.0% [18.4, 25.9] | −2.8% [−6.2, 0.4] | — |
| desharnais | temporal | 15 | 7 | 33.3% [13.2, 60.0] | log-size 40.0% [20.0, 66.7] | −6.7% [−26.7, 13.3] | — |
| kitchenham | temporal | 29 | 2 | 48.3% [31.0, 65.5] | log-size 34.5% [17.2, 51.7] | +13.8% [0.0, 31.0] | 65.5% [48.3, 82.8] |
| maxwell | temporal | 12 | 23 | 50.0% [25.0, 75.0] | log-size 33.3% [8.3, 58.3] | +16.7% [−16.7, 50.0] | — |

MdAPE (GP): china 56.5% [53.7, 60.9], desharnais 29.9% [15.5, 58.7],
kitchenham 25.1% [13.2, 31.9], maxwell 25.3% [12.2, 62.4].

## Aggregate (gate-relevant view, Q7)

PRED(25) and the APEs behind MdAPE are unitless, so test rows pool across
datasets. Aggregate over the gate-carrying temporal datasets (China excluded
per Q11), n_test = 56:

| Model | PRED(25) | MdAPE |
|---|---|---|
| GP (engine) | 44.6% [32.1, 58.9] | 28.6% [21.1, 33.9] |
| log-size regression | 35.7% [23.2, 48.2] | 34.3% [27.2, 39.5] |

Engine − log-size, PRED(25): **+8.9% [−3.6, +21.4]**.

## Reading — honest per-dataset verdict within CI

- **china**: engine indistinguishable from baseline (−2.8% [−6.2, 0.4]); both
  at the dataset's ~21% ceiling — no signal in the features (Q11).
- **desharnais**: indistinguishable (−6.7% [−26.7, 13.3]); n=15, CIs span ~40
  points. Uninformative.
- **kitchenham**: point win (+13.8%) but the paired CI lower bound sits **at
  0.0** — not confirmed at 95%. n=29.
- **maxwell**: point win (+16.7%) but CI [−16.7, 50.0] — not confirmed. n=12.
- **aggregate**: point win (+8.9%) but CI still straddles 0 (−3.6 .. +21.4).

**Conclusion of Step 1 (counter to the expectation that the win would hold):**
the GP's advantage over the log-size baseline is **not statistically
confirmable** on the current data — not per dataset and not in aggregate. The
point estimates consistently favor the engine, but the single 80/20 temporal
split leaves test blocks of 12–29 rows, and even pooled (n=56) the 95% CI
includes zero. The expert on kitchenham (65.5%) remains clearly ahead of the
base GP.

This is a power problem, not (yet) a verdict. Two levers can resolve it,
neither of which touches the frozen thresholds:
1. **More temporal test coverage** via rolling-origin (expanding-window)
   temporal CV — still strictly past→future, never random — which multiplies
   the pooled test rows. This is a split-methodology change to a FROZEN
   protocol, so it is a founder decision (tracked as Q14).
2. **A larger true effect**: the gradient-boosting regressor (Step 2) may open
   a gap wide enough to clear the CI at the current n. Worth running regardless.
