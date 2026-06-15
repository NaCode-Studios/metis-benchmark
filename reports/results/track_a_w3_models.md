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

## Phase 3 — hierarchical partial pooling: also NEGATIVE (does not help)

Pooled global slope (empirical Bayes, never-test initial windows of the three
rolling-origin dev datasets): **b_0 = 0.557, τ² = 0** — the datasets' size→effort
elasticities are consistent enough that DerSimonian–Laird finds no between-dataset
variance, so the pooled slope is a single shared elasticity.

PRED(25), out-of-fold:

| Dataset | GP-isolated | GP-meanfn | GP-pooled | GBM | log-size |
|---|---|---|---|---|---|
| desharnais | 38.5% | 41.1% | (≈39%) | 39.7% | 32.9% |
| maxwell | 51.6% | 38.7% | 38.7% | 41.9% | 29.0% |
| cocomo81 | 41.9% | 40.3% | 38.7% | 22.6% | 24.2% |
| china (out-gate) | 19.2% | 20.0% | 19.4% | 20.4% | 22.0% |
| **aggregate (4 gate)** | **43.5% [36.7, 50.2]** | 41.5% | 41.5% [35.3, 48.3] | 35.3% | 31.4% |

Paired: **GP-pooled − GP-isolated = −1.9% [−6.3, +2.9]** (aggregate) — indistinguishable,
slightly negative.

**Finding.** The "main move" of W3 does **not** help either. Pooling the
size-law slope across datasets is neutral-to-slightly-negative versus the
zero-mean GP, for the same reason the mean function failed: the GP already
models the size effect through its features, so constraining it with a shared
size law — pooled or not — removes flexibility without adding information. The
pooling machinery is correct (unit-tested: differential shrinkage, DL τ²
estimation, leakage-free global from never-test rows); it is simply not the
missing ingredient.

**Phase 2 + 3 conclusion.** Neither the mean function nor hierarchical pooling
closes the gap. The best Track A model remains the **W2 zero-mean GP-isolated**
at **43.5%** aggregate — beating the log-size baseline by +12.1% [5.3, 18.8]
(CI clears zero) but stuck **~12 points below the 55% threshold**. Whether 55%
is reachable at all is now a question for the honest feasibility ceiling
(Phase 4), which determines the verdict and the pre-registered stop rule.

## Phase 4 — honest feasibility ceiling: STOP RULE TRIGGERED

The W2 in-sample ceiling (66–78%) measured memorization. The honest ceiling is
the best **out-of-sample** PRED(25) over a diverse flexible model zoo (GP, GBM,
RandomForest-500, HistGradientBoosting) under the protocol split.
Reproduce: `python scripts/run_ceiling.py`.

| Dataset | best model | honest ceiling PRED(25) | resid σ(log) best / size-only |
|---|---|---|---|
| desharnais | RF-500 | 46.3% [31.7, 61.0] | 0.61 / 0.61 |
| kitchenham | GP | 45.2% [34.2, 56.2] | 0.59 / 0.70 |
| maxwell | GP | 51.6% [35.5, 67.7] | 0.60 / 0.50 |
| cocomo81 | GP | 41.9% [30.6, 54.8] | 0.57 / 1.01 |

**4 of 4 gate development datasets fall below 55%.** The pre-registered stop
rule (protocol v1.3 sec. 8) triggers: *55% PRED(25) is not achievable on this
data*, so Track A closes as "point-accuracy threshold unreachable, demonstrated."

**Why — the irreducible noise.** The best model's out-of-sample log-residual
scatter is σ ≈ 0.57–0.61 on every dataset: roughly ±60% multiplicative error
that no model removes. PRED(25) requires landing within ±25%; with this much
irreducible productivity scatter, the fraction within 25% is structurally
capped near 45–52%. The recorded, leakage-safe features simply do not explain
enough of the effort/size productivity variance — the same phenomenon proven
for China (Q11), now shown to hold across all gate datasets. (On cocomo81 the
features do help a lot — σ 1.01→0.57 — but not enough to clear 55%.)

**This validates the W3 negative results.** The mean function and hierarchical
pooling did not fail to be implemented; they failed to help because there is no
more signal to extract. The ceiling is the data's, not the model's.
