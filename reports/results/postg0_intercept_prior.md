# Post-G0 — hierarchical intercept: the epoch bias as a measured prior

> **Post-G0 R&D — non gate-carrying. Il verdetto G0 (NO-GO) resta valido e non
> è oggetto di questo esperimento.**
> All pre-registered reports and verdicts (reports/REPORT.md, protocol.md, the
> G0 result files) are untouched. SEERA is a sealed holdout opened exactly once
> for the G0 verdict: it is **excluded** from every number below (no fit, no
> calibration, no measurement).
> Reproduce: `PYTHONPATH=src python scripts/run_intercept_prior.py`.

## Question

The 6-8x bias G0 observed on real projects is, in log space, an **intercept
offset**: the gate refits the log-productivity intercept per dataset ("it
carries the unit"), which silently assumes local actuals exist. A cold-start
organization has none. So instead of discovering the offset a posteriori, we
measure how much the intercept **varies between organizations/epochs**:
`tau_a` = between-dataset SD of mean log-productivity. `e^(±2·tau_a)` is then
the multiplicative scale range an honest cold-start estimate must declare
before any local recalibration.

## Method (declared before the results were computed)

- **Contributors** — the tabular datasets whose size is in (adjusted) function
  points and whose effort is convertible to person-hours, so every intercept
  lives in one unit system (log person-hours per FP): desharnais, kitchenham,
  maxwell, china, albrecht (effort shipped in **kilo** person-hours, converted
  x1000). In log space a unit mismatch is an additive intercept offset
  indistinguishable from organization bias, hence:
- **Exclusions** — `seera` (sealed holdout, see header); `cocomo81` (size in
  KLOC, effort in person-months: any KLOC→FP gearing constant is
  language-dependent folklore and would masquerade as epoch bias).
- **Estimator** — per dataset `a_d = mean log(effort/size)` with
  `se_d = sd/√n`; DerSimonian–Laird across datasets splits the observed spread
  of the `a_d` into within-noise and the between component `tau_a²`
  (`fit_intercept_prior` in `track_a/pooling.py`, additive — gate defaults
  unchanged). 95% CI for `tau_a` by Q-profile (Viechtbauer 2007); with k=5
  datasets a bootstrap over datasets would be noise.
- **Sensitivity** — intercepts re-taken at the pooled slope `b_0` (i.e.
  `a_d = mean(log effort − b_0·log size)`) to check `tau_a` is not a size-mix
  artifact of pinning the elasticity at 1.

## Results (run of 2026-07-02, data in `data/raw`, venv `.venv`)

Per-dataset intercepts (primary, elasticity pinned at 1):

| dataset | n | a_d | se_d | e^a_d (person-hours per FP) |
|---|---:|---:|---:|---:|
| desharnais | 81 | 2.721 | 0.066 | 15.20 |
| kitchenham | 145 | 1.782 | 0.063 | 5.94 |
| maxwell | 62 | 2.459 | 0.077 | 11.70 |
| china | 499 | 2.041 | 0.043 | 7.69 |
| albrecht | 24 | 3.138 | 0.140 | 23.05 |

The five organizations disagree on raw productivity by a factor ~3.9 between
extremes (5.9 vs 23.1 h/FP) — with within-dataset SEs an order of magnitude
smaller, so the spread is real heterogeneity, not sampling noise.

Random-effects summary:

| quantity | primary (slope=1) | sensitivity (slope=b_0=0.871) |
|---|---:|---:|
| pooled intercept a_0 | 2.417 (11.21 h/FP) | 3.159 |
| **tau_a** | **0.440** | 0.449 |
| tau_a 95% CI (Q-profile) | [0.306, 1.543] | [0.324, 1.633] |
| e^(±tau_a) (68% range) | x1.55 / x0.64 | x1.57 / x0.64 |
| **e^(±2·tau_a) (95% range)** | **x2.41 / x0.41** | x2.45 / x0.41 |
| CI-upper 95% range e^(±2·tau_hi) | x21.88 | x26.20 |

The sensitivity run moves `tau_a` by 0.009: the between-dataset spread is
**not** a size-mix artifact.

## Sanity check: does the observed 6-8x bias fit the prior?

| observed bias | log offset | vs 2·tau_a = 0.88 | vs 2·tau_hi = 3.09 |
|---|---:|---|---|
| 6x | 1.79 | **OUTSIDE** the point prior | inside the CI-upper prior |
| 8x | 2.08 | **OUTSIDE** the point prior | inside the CI-upper prior |

Honest reading: the point prior **under-covers** the bias actually observed on
the user's real projects. That is informative, not a contradiction. The five
contributors are all 1979–2003 enterprise organizations; `tau_a` measures the
spread *among them*, and a 2025 solo/AI-assisted context sits outside the
corpus's epoch span (the 6-8x offset is ~4·tau_a — an out-of-population draw,
exactly the "epoch offset" reading of the G0 failure). With k=5 the CI is wide
enough to contain the observed bias, so the data cannot *reject* that 6-8x is
ordinary between-organization variation — but the point estimate says the
corpus's internal spread does not predict it.

## Product reading (the assumption the API should state)

- **Cold start, no local recalibration:** the expected p50 scale factor is
  within **±2.4x at 95%** (`e^(±2·tau_a)`, tau_a = 0.440) *if the client's
  organization resembles the benchmark population* — and the measured 6-8x on
  out-of-population projects shows that clause can fail by another ~2-3x.
  The honest phrasing is therefore: "**at least ±2.4x**, measured across five
  historical organizations; modern out-of-corpus contexts have shown up to
  8x" (tau_a CI upper bound: ±21.9x — the small-k uncertainty is itself part
  of the honest message).
- **With local recalibration** (Sprint 1: local intercept refit on client
  actuals) the tau_a term collapses to the within-organization noise — which
  is why the recalibration + storico consuntivi path, not a better global
  model, is the mitigation for the epoch bias.
- These numbers quantify the *hours* dimension scale risk; cost and workforce
  inherit it multiplicatively.

## Limitations

- k=5 datasets: tau_a is estimable but its CI spans a factor of 5; every
  downstream use should carry the CI, not just the point.
- All contributors are function-point, waterfall-era enterprise projects; the
  prior is a **lower bound** for organizations outside that population (the
  sanity check above demonstrates this concretely).
- Elasticity pinned at 1 in the primary (mean log-productivity); the pooled
  slope sensitivity (b_0 = 0.871) shows the conclusion is insensitive.
