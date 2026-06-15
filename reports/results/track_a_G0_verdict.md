# Track A — G0 verdict (partial: tabular channel only)

> 2026-06-12. Applies the FROZEN thresholds (protocol v1.2 sec. 5) to the
> rolling-origin results. This is the **Track A** verdict only; the full G0
> also requires Track B (semantic channel), not yet run. Thresholds were fixed
> before results and are not renegotiated here.
> Estimator: rolling-origin temporal CV (v1.2, pre-registered). Numbers read
> off the gate-selected regressor; GP shown alongside as the doctrine-correct
> regressor (Q16). Single-split numbers are in the companion reports.

## Gate-carrying datasets, rolling-origin (cold-start, no First.estimate)

| Dataset | n test | PRED(25) (gate-sel.) | GP | MdAPE (GP) | vs log-size baseline | Expert |
|---|---|---|---|---|---|---|
| desharnais | 39 | 33.3% | 38.5% | 28.8% | indistinguishable | — |
| kitchenham | 73 | 45.2% | 45.2% | 28.1% | **beats** +12.3% [4.1, 20.6] | 61.6% |
| maxwell | 31 | 41.9% | 51.6% | 23.1% | **beats** (GP) +22.6% [3.2, 45.2] | — |
| **aggregate** | 143 | 41.3% [33.6, 49.0] | ~45% | 28.8% | +6.3% [−2.1, 15.4] | — |
| china (out-gate, Q11) | 499 | 19.2% | — | 56.5% | indistinguishable | — |

## Verdict against the FROZEN thresholds

| Gate criterion | Threshold | Result | Status |
|---|---|---|---|
| PRED(25) ≥ 55% on ≥2 datasets | 55% | best is maxwell 51.6% (GP); **no dataset ≥ 55%** | **FAIL** |
| MdAPE ≤ 22% on ≥2 datasets | 22% | best is maxwell 23.1%; **no dataset ≤ 22%** | **FAIL** |
| Coverage within 5 pts of 90% | 85–95% | 94.4% [90.6, 98.2] | **PASS** |
| Beat statistical baselines | — | beats log-size on kitchenham & maxwell (GP), CI clears 0 | **PASS** |
| Beat expert (aggregate per track) | — | cold-start 45.2% vs expert 61.6% (kitchenham) | **FAIL** |

## Verdict: **PARTIAL — Track A does NOT pass (point accuracy below threshold)**

The engine, on a properly powered estimator:
- **keeps its interval promise** (coverage 94.4%, the criterion that separates
  Metis from an LLM guessing numbers) — PASS;
- **beats the naive statistical baselines** on the two signal-bearing datasets
  with confidence intervals that clear zero — PASS;
- but **does not reach the point-accuracy thresholds** (PRED(25) ≥ 55%,
  MdAPE ≤ 22%) on any gate dataset, and **does not beat the human expert**
  cold-start — FAIL.

This FAIL now *means something*: under the underpowered single split it was
"cannot measure"; under rolling-origin it is a real, interpretable shortfall.
Crucially it is **not a dead end** — the in-sample ceiling (kitchenham 66%,
maxwell 71%, desharnais 78%, Q15) shows 55% is reachable on these features. The
~15–25 point gap between out-of-fold (~45–52%) and ceiling (66–78%) is
generalization, driven by tiny training folds. Channel fusion of GP+GBM was
tested and does not close it (max 54.8%, Q-Step-5); the remedy is feature
engineering and more gate datasets (cocomo81, seera), not more model plumbing.

## Decisions carried by this verdict

- **Repo stays private (Q6).** Publication triggers on a *full* G0 pass
  (Track A + Track B). Track A does not pass, and Track B is unrun, so the
  benchmark is **not** published at the end of W2.
- **Doctrine-correct regressor is GP (Q16)**, vindicated by rolling-origin; the
  next pre-registered iteration should make GP the primary, not GBM.
- **First.estimate stays in assist mode only (Q13)**; its 68.5% is a product
  number, absent from this verdict by construction.
- **Next iteration to clear the gate:** feature engineering on the existing
  datasets and adding cocomo81 + seera as gate datasets (more training rows,
  more pooled test), targeting the 66–78% ceiling.

## Signature

Decider: **founder** (per protocol v1.2). Verdict: **Track A partial — not
passing on point accuracy; coverage and baseline criteria met; path to 55%
open.** Full G0 deferred until Track B (semantic channel) is validated.

_Signed: ____________________  Date: ___________
