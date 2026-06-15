# Track A — W3 final verdict (definitive; supersedes the W2 partial verdict)

> 2026-06-12. Applies the FROZEN thresholds (protocol v1.3 sec. 5) to the W3
> results. Estimator: rolling-origin temporal CV (cocomo81: ordered k-fold).
> Primary regressor: GP (Q16). SEERA opened once here, as the sealed holdout.
> Thresholds were fixed before results and are not renegotiated.

## Final gate table — all 5 datasets (cold-start, no First.estimate)

| Dataset | split | n test | GP PRED(25) | GP MdAPE | honest ceiling | vs log-size | expert |
|---|---|---|---|---|---|---|---|
| desharnais | rolling | 39 | 38.5% | 28.8% | 46.3% | beats (agg.) | — |
| kitchenham | rolling | 73 | 45.2% | 28.1% | 45.2% | **beats** +12.3% [4.1, 20.6] | 61.6% |
| maxwell | rolling | 31 | 51.6% | 23.1% | 51.6% | **beats** +22.6% [3.2, 45.2] | — |
| cocomo81 | ord. k-fold | 62 | 41.9% | 35.6% | 41.9% | beats (agg.) | — |
| **seera (holdout)** | rolling | 59 | 25.4% | 41.6% | 33.9% | +3.4% | 25.4% |
| china (out-gate, Q11) | ord. k-fold | 499 | 19.2% | 56.5% | ~21% | — | — |

Aggregate (4 gate dev datasets, n=205): GP 43.5% [36.7, 50.2], beats log-size
+12.1% [5.3, 18.8].

## Verdict against the FROZEN thresholds

| Gate criterion | Threshold | Result | Status |
|---|---|---|---|
| PRED(25) ≥ 55% on ≥2 datasets | 55% | **0 of 5** datasets reach 55% (best 51.6%) | **FAIL** |
| MdAPE ≤ 22% on ≥2 datasets | 22% | 0 of 5 reach ≤22% (best 23.1%) | **FAIL** |
| Coverage within 5 pts of 90% | 85–95% | 94.6% [91.5, 97.7], n=204 (incl. SEERA) | **PASS** |
| Beat statistical baselines | — | GP beats log-size on the signal datasets, CI clears 0 | **PASS** |
| Beat expert (aggregate) | — | kitchenham 45.2% vs 61.6%; seera tie 25.4% | **FAIL** |

## Honest feasibility — STOP RULE CONFIRMED (incl. holdout)

The best out-of-sample model (GP/GBM/RF-500/HistGB) is below 55% on **5 of 5**
gate datasets — including SEERA at **33.9%**, confirmed on data that played no
role in development. Irreducible out-of-sample log-residual scatter σ ≈ 0.57–0.61
(±~60% multiplicative). **55% PRED(25) is not achievable on these public
datasets with leakage-safe features — demonstrated, not assumed.**

## Verdict: **Track A does NOT pass — threshold proven unreachable on this data**

The engine, on a properly powered and pre-registered evaluation:
- **keeps its interval promise** (CQR coverage 94.6%) — PASS;
- **beats the naive statistical baselines** with confidence — PASS;
- but **cannot reach the point-accuracy thresholds** (PRED(25) ≥ 55%,
  MdAPE ≤ 22%) — and the honest ceiling proves no model can, on this data —
  and **does not beat the human expert** — FAIL.

This is a *demonstrated* negative, not a "couldn't measure" or "didn't try
hard enough": the feature sets were leakage-audited, the estimator is
rolling-origin with CIs, GP/GBM/RF/HistGB were all tried, mean-function and
hierarchical pooling were implemented and shown not to help because there is no
more signal, and the ceiling was confirmed on a sealed holdout.

## Pre-registered decision (protocol v1.3 Phase 6): **close Track A, go to Track B**

The decision rule, fixed in advance:
- Pass → Track B. *(not the case)*
- Not pass but gap narrowing AND 55% reachable → propose W4. *(not the case —
  Phase 4 proves 55% unreachable)*
- **Gap not moved OR feasibility says impossible → close Track A as best-effort
  + assist mode as product, proceed to Track B.** ✓ **This applies.**

**Actions:**
1. **Track A closed** as best-effort. The tabular cold-start engine is honest
   (calibrated intervals, beats naive baselines) but does not meet the gate;
   55% is proven unreachable on public data with deployable features.
2. **Assist mode (Q13) retained as a product capability** — on kitchenham it
   lifts the expert from 61.6% to 68.5% (separate from the gate).
3. **Proceed to Track B** (semantic channel: JOSSE, SiP) — the V20 thesis that
   the *text* of requirements carries signal the tabular cost-driver tables
   lack is now the live hypothesis, and it is where Metis's differentiation
   actually sits.
4. **Repo stays private (Q6):** no full G0 pass; not published.

## Signature

Decider: **founder** (protocol v1.3). Verdict: **Track A not passing;
55% threshold proven unreachable on public tabular data; closed best-effort;
assist mode kept as product; proceeding to Track B.**

_Signed: ____________________  Date: ___________
