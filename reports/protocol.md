# Evaluation protocol — Metis benchmark (Gate G0)

> Status: **DRAFT — to be frozen at the end of week 1, before any experiment runs.**
> Once frozen, thresholds and rules in this document do not change after seeing
> results; any amendment must be recorded here with date and motivation.

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
- Datasets without a usable date field: fallback split documented per dataset
  in the data dictionary, flagged explicitly in the report. <!-- W1: list them here -->

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
   — JOSSE (~19% of tasks), SiP (all tasks).

## 5. Gate G0 — pass criteria

| Criterion | Threshold |
|-----------|-----------|
| PRED(25) | ≥ 55% on at least **2 datasets of each track** |
| MdAPE | ≤ 22% on at least **2 datasets of each track** |
| Empirical coverage | within **5 points** of the nominal 90% |
| Baselines | engine beats all three, including the expert on JOSSE/SiP |

Thresholds are deliberately more conservative than the pilot KPIs
(PRED(25) ≥ 60%, MdAPE ≤ 20%): public datasets are more heterogeneous than a
single client's history, which benefits from shrinkage.

**Outcomes.** Pass → Phase 1 (core engine). Fail → iterate on feature
engineering and model choice (GP vs gradient boosting) before writing a single
line of client code; if iteration is not enough, documented stop.

## 6. Anti-leakage rules

- No feature derived from information unavailable at estimation time.
- Conformal calibration set never overlaps training and never precedes it
  chronologically.
- Hyperparameter selection on train via temporal CV inside the train block;
  the test block is touched once, at the end.
- TAWOS subset (projects and issue range) is declared here before any
  retrieval experiment: <!-- W1: declare subset -->

## 7. Open items to close before freezing (week 1)

- [ ] Verify download sources and licenses per dataset (registry.py `verify` entries)
- [ ] Data dictionary per dataset: effort unit, size unit, date field, category field
- [ ] List of datasets requiring the non-temporal fallback split
- [ ] TAWOS subset declaration
- [ ] Who signs G0
