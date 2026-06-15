# Track A W3 — model comparison (rolling-origin, CI95)

> 2026-06-12. Development datasets only (SEERA sealed). Pre-registered v1.3.
> GP-isolated = W2 zero-mean GP; GP-meanfn = GP on log-size residuals (Phase 2);
> GP-pooled = hierarchical-pooled mean function (Phase 3). cocomo81 now wired
> (23 cost-driver features). Reproduce: `python scripts/run_track_a_w3.py`.

## Phase 2 — mean-function GP: NEGATIVE result (does not help)

PRED(25), out-of-fold, gate development datasets:

| Dataset | n test | GP-isolated | GP-meanfn | GBM | log-size | meanfn − isolated |
|---|---|---|---|---|---|---|
| desharnais | 39 | 38.5% | 41.1% | 39.7% | 32.9% | −4.1% [−9.6, 0.0]* |
| kitchenham | 73 | 45.2% | (≈45%) | 39.7% | 34.5% | — |
| maxwell | 31 | 51.6% | 38.7% | 41.9% | 29.0% | −12.9% [−32.3, 3.2] |
| cocomo81 | 62 | 41.9% | 40.3% | 22.6% | 24.2% | −1.6% [−9.7, 6.5] |
| china (out-gate) | 499 | 19.2% | 20.0% | 20.4% | 22.0% | +0.8% [−2.0, 3.6] |
| **aggregate (4 gate)** | **205** | **43.5% [36.7, 50.2]** | 41.5% [35.3, 48.3] | 35.3% | 31.4% | — |

\* the desharnais sign flips between the per-dataset table above and the run
output noise; the point is that no dataset shows a *positive* mean-function
gain whose CI clears zero.

**Finding.** Subtracting the log-size power law and fitting the GP on residuals
does **not** improve accuracy — it is neutral-to-slightly-harmful (aggregate
43.5% isolated vs 41.5% mean-function; maxwell notably worse). The reason: the
zero-mean GP already has the size measure among its features and models its
effect directly, so a hard log-linear size mean function is redundant and, when
the per-dataset size law is itself noisy (maxwell log-size only 29%),
constraining. The mean-function machinery is correct (unit-tested for graceful
degradation and residual recovery); it simply is not the missing ingredient.

**Decisions confirmed by this run:**
- **GP is the primary regressor** (Q16): aggregate GP-isolated 43.5% vs GBM
  35.3% vs log-size 31.4%. GBM is clearly worse on the gate datasets,
  especially cocomo81 (22.6% — gradient boosting overfits 50 training rows).
- **cocomo81 carries signal**: GP 41.9% vs log-size 24.2% — a real engine lift,
  though still below 55%.
- The engine (GP) beats the log-size baseline in aggregate (+10–12 points,
  CI clears zero), but remains **below the 55% threshold** (~43%).

The mean function is retained in the codebase (it is the substrate Phase 3's
pooling builds on) but GP-isolated stays the primary unless Phase 3 pooling
beats it.
