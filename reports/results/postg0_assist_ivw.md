# Post-G0 — inverse-variance assist blend, at adequate power

> **Post-G0 R&D — non gate-carrying. Il verdetto G0 (NO-GO) resta valido e non
> è oggetto di questo esperimento.**
> All pre-registered reports and verdicts are untouched. SEERA is a sealed
> holdout opened exactly once for the G0 verdict: it is **excluded** from every
> number below. Like `run_assist.py`, everything here consumes recorded expert
> estimates and therefore lives permanently OUTSIDE the gate.
> Reproduce: `PYTHONPATH=src python scripts/run_assist_ivw.py`.

## Question

G0 withdrew the assist claim because its only paired test (kitchenham, n=73)
was under-powered (McNemar p=0.2266, power ≈ 0.16): absence of proof, not
proof of absence. Meanwhile the product blends model p50 and expert estimate
with a **hand-picked fixed weight**. Both problems are addressed at once:
weight the blend by measured information (inverse variance), test it on every
dataset that records an expert estimate, pooled, and report the power
explicitly. The question: **does the IVW blend beat the expert alone — and
with what power?**

## Pre-declared design (frozen before any result was computed)

- **Datasets & splits** (the gate's own splits, unchanged; no SEERA):
  - *kitchenham* — rolling-origin on date; model = log-space GP on the
    cold features (`Adjusted.function.points`, `Client.code`); expert =
    `First.estimate` (n=145 → 73 tested rows).
  - *sip* — rolling-origin on date; model = k-NN Nadaraya-Watson over the
    cached bge-small embeddings, (k, τ) test-blind per fold; expert =
    `HoursEstimate` (6,149 tested rows).
  - *josse* — grouped k-fold by project; same k-NN NW engine; expert = the
    annotated subset (`expert_estimated_effort > 0`, 4,329 tested rows).
- **Sigma measurement (out-of-sample, deployable):** per fold the model fits
  on the TRAIN block only; `σ_m` and `σ_e` are SDs of log residuals on the
  CALIBRATION block (the block the gate reserves for conformal calibration —
  here it calibrates the weight), on rows with an expert estimate. The fold's
  test rows are blended with `w = σ_m⁻²/(σ_m⁻² + σ_e⁻²)` from those cal-block
  sigmas — a weight a deployed system could actually have computed in time.
- **Arms:** (1) expert alone; (2) fixed-0.5 log blend — the product's current
  mode; (3) IVW blend. Blends are log-space (`assist.blend_log_space`).
- **Primary test:** exact McNemar on PRED(25) hits, IVW vs expert alone, per
  dataset and pooled. **Pooling method (declared here, before the results):**
  the paired hit vectors are concatenated across datasets — equivalently, the
  discordant counts b and c are summed over datasets and the exact binomial is
  applied to the pooled discordants. Each project contributes at most one
  pair; under H0 a discordant pair is a fair coin whatever its dataset;
  PRED(25) is unit-free, so pooling across effort units is legitimate.
- **Power:** minimal detectable effect via `mcnemar_mde`, achieved power at
  the observed discordant split (inverting `mcnemar_discordant_for_power`),
  required N via `mcnemar_sample_size` at the planning effect π₁=0.65.

## Results (run of 2026-07-02; two consecutive runs bit-identical)

Measured log-error scales and the weights they produce (w = weight on the
model):

| dataset | σ_m (cal, median over folds) | σ_e (cal, median) | w(model) per fold | σ_m / σ_e (test, descriptive) |
|---|---:|---:|---:|---:|
| kitchenham | 0.658 | 0.379 | 0.13–0.36 | 0.607 / 0.327 |
| sip | 1.497 | 0.952 | 0.25–0.36 | 1.610 / 0.981 |
| josse | 1.411 | 1.003 | 0.31–0.36 | 1.551 / 0.990 |

The expert is the more precise estimator on **every** dataset (σ_e < σ_m
throughout), so IVW correctly pushes the model weight down to ~0.13–0.36 —
the information-honest replacement of the hand-picked 0.5.

Arms (out-of-fold test rows with an expert estimate; PRED(25) and MdAPE with
bootstrap 95% CI):

| dataset | arm | PRED(25) | MdAPE |
|---|---|---|---|
| kitchenham (n=73) | expert alone | 61.6% [50.7, 72.6] | 16.7% [9.7, 24.6] |
| | blend fixed-0.5 | 53.4% [41.1, 64.4] | 22.8% [15.4, 28.6] |
| | blend IVW | 63.0% [52.1, 74.0] | 21.7% [17.9, 24.4] |
| sip (n=6149) | expert alone | 41.6% [40.3, 42.8] | 37.0% [34.1, 40.0] |
| | blend fixed-0.5 | 23.1% [22.0, 24.1] | 55.2% [53.7, 57.0] |
| | blend IVW | 28.8% [27.7, 30.0] | 48.7% [47.6, 50.0] |
| josse (n=4329) | expert alone | 45.0% [43.4, 46.4] | 33.3% [33.3, 33.3] |
| | blend fixed-0.5 | 21.3% [20.1, 22.6] | 56.9% [55.3, 58.2] |
| | blend IVW | 25.5% [24.1, 26.8] | 47.9% [46.4, 49.3] |

Primary paired test (IVW vs expert alone; b = IVW hits where expert misses,
c = the reverse):

| comparison | b | c | discordant | exact p | dPRED25 [95% CI] |
|---|---:|---:|---:|---:|---|
| kitchenham | 8 | 7 | 15 | 1.0000 | +1.4% [−8.2, +11.0] |
| sip | 411 | 1197 | 1608 | <0.0001 | −12.8% [−14.0, −11.5] |
| josse | 367 | 1212 | 1579 | <0.0001 | −19.5% [−21.3, −17.7] |
| **POOLED (n=10,551)** | **786** | **2416** | **3202** | **<0.0001** | — |

Secondary (the product's current fixed-0.5 vs expert, pooled): b=911, c=3079,
p<0.0001 — worse still.

## Power analysis (the point of this experiment)

| test | n | discordant | MDE (π₁, 80% power) | achieved power at observed π₁ |
|---|---:|---:|---:|---:|
| kitchenham alone | 73 | 15 | ≥0.834 | 0.04 |
| sip | 6149 | 1608 | ≥0.535 | 1.00 |
| josse | 4329 | 1579 | ≥0.536 | 1.00 |
| **pooled** | **10551** | **3202** | **≥0.525** | **1.00** |

The pooled study needs only 85 discordant pairs to detect a moderate edge
(π₁=0.65) at 80% power; it observed 3,202. **This is no longer an
under-powered test.** Kitchenham alone remains as under-powered as G0 said
(MDE π₁≥0.834), which is why it reads as a wash while the pooled test is
decisive.

## Post-hoc diagnostics (explanatory; not part of the pre-declared comparison)

The IVW optimality proof assumes unbiased estimators with independent errors.
Both assumptions fail here, measured on the test rows:

| dataset | mean log-residual (model) | mean log-residual (expert) | ρ(res_m, res_e) | correlation-aware w* |
|---|---:|---:|---:|---:|
| kitchenham | +0.125 | −0.108 | +0.476 | +0.044 |
| sip | +0.142 | +0.070 | +0.400 | +0.145 |
| josse | +0.769 | −0.264 | +0.336 | +0.197 |

`w* = (σ_e² − ρσ_mσ_e)/(σ_m² + σ_e² − 2ρσ_mσ_e)` is the MSE-optimal model
weight once error correlation is admitted: 0.04–0.20, well below the
independence-assuming IVW's 0.25–0.36. The model's errors are correlated with
the expert's (both miss on the same hard projects), so the diversification
benefit the IVW banks on is largely fictitious; on josse the model also
carries a +0.77 log bias (≈2.2× median underestimate) that the blend inherits
in proportion to w.

## Honest answer

1. **The IVW blend does not beat the expert alone — it loses, decisively.**
   Pooled McNemar p<0.0001 in the expert's favour, at power ≈ 1.00. This is a
   real negative, not absence of evidence: with these models and these
   datasets, mixing the model's output into a recorded expert estimate makes
   the estimate worse.
2. **IVW strictly dominates the product's current fixed-0.5** on all three
   datasets (e.g. sip 28.8% vs 23.1%; kitchenham 63.0% vs 53.4%). If a blend
   must exist, information weighting is the right way to set its weight —
   but the honest default weight is even lower than IVW suggests (see w*).
3. **Scope caveat:** on sip/josse the "model" is the Track B engine that
   already failed to beat the text-blind floor at G0 — this experiment
   measures blending *that* model. The IVW machinery is model-agnostic: if a
   future model measures σ_m < σ_e on a client's calibration data, the same
   formula will shift weight toward it automatically. Note also that
   `run_assist.py`'s expert-as-*feature* mode (GP input, not output blend) is
   a different mechanism and remains not-demonstrated rather than harmful.

## Product recommendation

- **Do not adopt blend-IVW as the assist default.** When a client provides an
  expert estimate, the default should pass it through (w=0 on the model
  output) until local calibration data proves σ_m < σ_e for that client.
- **Replace the hand-picked fixed weight with the measured IVW weight** for
  clients who explicitly want a blend: it is strictly less harmful than 0.5
  everywhere measured, and it is computable from the same local consuntivi
  the Sprint-1 recalibration already collects (σ_e and σ_m on the client's
  own history — exactly the cal-block procedure used here).
- The blend applies to the hours dimension; cost and workforce inherit the
  chosen p50 multiplicatively, as everywhere in Metis.
