# Metis — Full G0 verdict (Phase 0 / public backtest)

> 2026-06-18. Both tracks complete. Applies the FROZEN thresholds (protocol
> v1.x/v2.0) to the rolling-origin / group-by-project results, with bootstrap
> CIs and pre-registered stop rules. Decider: founder.

## Result by track

| Track | Channel | Gate datasets | Best engine PRED(25) | Honest ceiling | Coverage | Verdict |
|---|---|---|---|---|---|---|
| A | Tabular (project) | desharnais, kitchenham, maxwell, cocomo81, seera | 41–52% (GP) | 33.9–51.6%, all <55% | 94.6% PASS | **not passing** |
| B | Semantic (task text) | JOSSE, SiP | 14–20% | 15.0% / 19.6%, both <55% | (moot) | **not passing** |

**Neither track meets the gate.** On both, the honest feasibility ceiling — the
best out-of-sample accuracy any model class reaches under the protocol split —
is below the 55% PRED(25) threshold, so the stop rule fires on each. The
result is a **documented no-go on public data**: it is demonstrated, not
assumed, and not a measurement artifact.

## What was actually proven

1. **Tabular cost-driver features cap at ~50% PRED(25)** (irreducible log-residual
   σ≈0.6). Confirmed across 5 datasets incl. a sealed holdout; mean-function and
   hierarchical pooling did not help because there is no more signal.
2. **The semantic channel does not predict real logged effort** cross-project:
   it ties or loses to the text-blind median on JOSSE/SiP; the ceiling is
   ~15–20%. Text carries a *small* signal on story points (Deep-SE +2.7%) but
   that does not transfer to hours/seconds.
3. **The engine is honest where it can be**: it beats naive statistical
   baselines on Track A with confidence, and its conformal intervals keep their
   90% promise (coverage 94.6%). The failure is accuracy, not calibration.
4. **The human expert beats every model** on the datasets that record one
   (kitchenham 61.6%, JOSSE 45%, SiP 41.6%) — experts use team/codebase context
   absent from the public features and text.

## Honest reading for the product

The public-data gate was built to be hard and to be a real test, not a sales
demo (the V20 "blind test"). It returns a clear signal: **public, leakage-safe,
cross-organization data does not contain enough to hit the precision bar Metis
set.** This is informative, not fatal — it tells where the value is and is not:

- **Where the gate fails:** generic cross-organization estimation from public
  attributes or requirement text alone. No model — ours or a flexible zoo —
  clears 55% on this data.
- **Where value remains (evidence-based):**
  - **Calibrated intervals** are real and distribution-free (coverage holds).
    "Here is a range that holds 90% of the time" is deliverable today.
  - **Assist mode** lifts an existing expert estimate (kitchenham 61.6% → 68.5%)
    — Metis as an expert *amplifier*, not a cold-start replacement.
  - **Proprietary, in-organization data** is the untested hypothesis: the gate
    used cross-project cold start (hardest case) on *public* logged time. A
    single client's own history — same team, same logging conventions, same
    codebase — is exactly the regime the public datasets cannot proxy, and is
    the V20 cold-start Stage 3. This is the Phase-1 pilot's job to test.

## Pre-registered decision

- **Full G0: NO-GO on the public-data gate as specified.** Neither track passes;
  55% is proven unreachable on public data with deployable, leakage-safe inputs.
- **Repo stays private (Q6):** publication required a genuine pass; there is none.
  The benchmark and its negative result are kept as an internal, reproducible
  asset (and an honest one — it would be publishable as a *methods/negative-result*
  artifact, a separate decision).
- **Product direction on the evidence:** pivot the near-term offer to (a)
  calibrated-interval estimation and (b) expert-assist refinement, and make the
  Phase-1 pilot a test of **proprietary single-client data** (the one regime the
  public gate could not evaluate) before committing to the cold-start
  estimation claim. Do not ship the "AI estimates effort cold-start from a spec"
  claim on the strength of public data — it is not supported.

## Signature

Decider: **founder** (protocol v1.x/v2.0). Verdict: **G0 NO-GO on the public
gate; tabular and semantic channels both proven below the 55% threshold on
public data; engine intervals are honest; expert-assist and
proprietary-data-pilot are the evidence-based next steps.**

_Signed: ____________________  Date: ___________
