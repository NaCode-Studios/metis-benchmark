# Evaluation protocol — Metis benchmark (Gate G0)

> Status: **v1.0 — READY TO FREEZE** (2026-06-12). Pending only: G0 signer
> designation and two unit confirmations (sec. 7) — neither affects thresholds.
> Once frozen, thresholds and rules in this document do not change after seeing
> results; any amendment must be recorded here with date and motivation.
> Note: the only experiments run before this version were the *baseline*
> smoke tests (scripts/smoke_baselines.py), which set the bar — no engine
> model has seen any test block.

## 1. Question under test

Can a statistical engine (similarity + regression with conformalized quantile
intervals) estimate software effort better than the standard baselines — and
better than the human expert, where the dataset records one — on public data?

Validation runs on two tracks:

- **Track A — project level** (tabular channel): PROMISE (Desharnais, COCOMO81,
  China, Kitchenham, Maxwell, Albrecht) + SEERA.
- **Track B — task level with text** (semantic channel): Deep-SE, JOSSE, SiP;
  TAWOS as retrieval scale only (not part of the gate).

## 2. Split policy

- **Temporal split only**: oldest records train, middle block calibrates the
  conformal intervals, newest records test. Never random.
- Default fractions: 60% train / 20% calibration / 20% test.
- Date granularity per dataset (see data_dictionary.md): day-level —
  kitchenham, sip; year-level ordering — desharnais, maxwell, seera.
- **Fallback split** (no usable date): cocomo81, china, albrecht, deepse,
  josse → ordered 5-fold cross-validation on a fixed seed, declared in the
  report with an explicit caveat. For deepse, the numeric suffix of the issue
  key gives per-project creation ordering and is used as the temporal proxy.

## 3. Metrics

| Metric | Definition | Implementation |
|--------|------------|----------------|
| PRED(25) | share of estimates within 25% of the actual | `evaluation.pred_at` |
| MdAPE | median absolute percentage error | `evaluation.mdape` |
| Coverage | share of actuals inside the conformal p50–p90 interval | `evaluation.empirical_coverage` |

Target variable is modeled in log space: z = log(effort).

## 4. Mandatory baselines

The engine must beat **all three** wherever they are computable:

1. **Median by category** (`baselines.MedianByCategory`)
2. **Linear regression of log(effort) on log(size)** (`baselines.LogSizeRegression`)
3. **Human expert estimate** recorded in the dataset (`baselines.expert_estimates`)
   — available on **four** datasets: kitchenham and seera (Track A), josse
   (~19% of tasks) and sip (Track B).

Smoke-test bar (test blocks, scripts/smoke_baselines.py): statistical
baselines land at PRED(25) 16–41%; the expert reaches **65.5% / 18.5%** on
kitchenham — above the gate thresholds — and 38.5% / 40.8% on sip. On
expert-annotated datasets the binding bar is therefore the expert, not the
statistical baselines.

## 5. Gate G0 — pass criteria

| Criterion | Threshold |
|-----------|-----------|
| PRED(25) | ≥ 55% on at least **2 datasets of each track** |
| MdAPE | ≤ 22% on at least **2 datasets of each track** |
| Empirical coverage | within **5 points** of the nominal 90% |
| Baselines | engine beats all three, including the expert where recorded (kitchenham, seera, josse, sip) |

Albrecht (24 rows) does not count toward the "2 datasets per track" rule —
kept for completeness only. TAWOS is outside the gate by design.

Thresholds are deliberately more conservative than the pilot KPIs
(PRED(25) ≥ 60%, MdAPE ≤ 20%): public datasets are more heterogeneous than a
single client's history, which benefits from shrinkage.

**Outcomes.** Pass → Phase 1 (core engine). Fail → iterate on feature
engineering and model choice (GP vs gradient boosting) before writing a single
line of client code; if iteration is not enough, documented stop.

## 6. Anti-leakage rules

- No feature derived from information unavailable at estimation time.
  Known leakage fields excluded from features: SEERA `% project gain (loss)`
  and `Actual duration`; kitchenham `Actual.duration` and
  `Estimated.completion.date`; china `Duration`, `N_effort` and the
  productivity ratios (`PDR_*`, `NPDR_*`, `NPDU_*`), which are derived from
  actual effort.
- Conformal calibration set never overlaps training and never precedes it
  chronologically.
- Hyperparameter selection on train via temporal CV inside the train block;
  the test block is touched once, at the end.
- **TAWOS subset declaration**: retrieval/reranking experiments use the 20
  TAWOS projects with the most story-point-annotated issues, capped at
  100,000 issues total, ordered by issue creation date. TAWOS is used for
  retrieval scale only and contributes nothing to the gate.

## 7. Open items

- [x] Verify download sources and licenses per dataset (registry.py, 2026-06-12)
- [x] Data dictionary per dataset (reports/data_dictionary.md)
- [x] List of datasets requiring the non-temporal fallback split (sec. 2)
- [x] TAWOS subset declaration (sec. 6)
- [ ] Confirm Albrecht effort unit against the literature and SEERA effort
  unit against "SEERA dataset attribute formulas.pdf" (affects reporting
  conversion only, not thresholds)
- [ ] Who signs G0
