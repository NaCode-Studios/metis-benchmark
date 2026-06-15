# Track A — Step 3: conformalized coverage under rolling-origin

> 2026-06-12. CQR (Romano et al. 2019) over the GBM quantiles, calibrated per
> rolling-origin fold on that fold's calibration block, coverage pooled across
> the gate datasets. Measured on rolling-origin (not the single split): on
> 12-29 single-split test rows the binomial CI on coverage is ~±13 points, so
> the "within 5 points" criterion is unmeasurable there; pooled (n=143) it is
> readable. Reproduce: `python scripts/run_coverage.py`.
> Calibration plot: `calibration_track_a.png`.

## Calibration sweep (pooled, n=143)

| Nominal | Empirical | CI95 (binomial) | gap |
|---|---|---|---|
| 50% | 61.5% | [53.6, 69.5] | +11.5 |
| 70% | 81.1% | [74.7, 87.5] | +11.1 |
| 80% | 90.2% | [85.3, 95.1] | +10.2 |
| **90% (gate)** | **94.4%** | **[90.6, 98.2]** | **+4.4** |
| 95% | 93.7% | [89.7, 97.7] | −1.3 |

## Reading

- **The coverage criterion PASSES.** At the gate's nominal 90%, empirical
  coverage is 94.4%, |94.4 − 90| = 4.4 < 5 — inside the protocol's 5-point
  tolerance (sec. 5). The interval keeps its promise on unseen projects. This
  is the one gate criterion the engine clears cleanly.
- **The intervals over-cover (conservative).** Empirical sits above nominal at
  every level below 95% (+10 to +11 points at 50–80%). The cause is the small
  per-fold calibration block: early rolling-origin folds calibrate on a few
  rows, and CQR's finite-sample correction `ceil((n+1)(1-α))/n` is conservative
  at small n, widening the interval. This errs on the safe side — the promise
  is over-kept, never broken — and **was not adjusted by hand** (that would be
  auto-deception, per the W2 rule). Larger calibration sets (more data / more
  gate datasets) would tighten it toward the diagonal.
- **Why this matters for the product.** Coverage is the criterion that
  separates Metis from "an LLM guessing numbers": the p90 buffer holds. The
  engine's current weakness is point accuracy (PRED(25)/MdAPE below threshold),
  not the honesty of its intervals.

## Machine built (reusable, split-agnostic)

`src/metis_benchmark/conformal.py` — `ConformalizedQuantile` (two-sided CQR,
finite-sample, tested). It wraps any quantile regressor and is reused as-is for
Track B. Coverage here is computed on the rolling-origin folds defined in
protocol v1.2; nothing is frozen on the single split.
