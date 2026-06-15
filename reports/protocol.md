# Evaluation protocol — Metis benchmark (Gate G0)

> Status: **v1.2 — FROZEN** (2026-06-12). Thresholds and rules below do not
> change after seeing results; any amendment must be recorded here with date
> and motivation, and may only tighten the gate, never loosen it.
> Note: the only experiments run before the v1.1 freeze were the *baseline*
> smoke tests, which set the bar. v1.2 amends only the *estimator* (the split
> methodology), never the thresholds — see the amendment log (sec. 8).
>
> **G0 decider: the founder.** The thresholds in sec. 5 are fixed at this
> freeze and are not revisable downward after results are seen. The decider's
> role at G0 is to apply them, not to renegotiate them.

## 1. Question under test

Can a statistical engine (similarity + regression with conformalized quantile
intervals) estimate software effort better than the standard baselines — and
better than the human expert, where the dataset records one — on public data?

Validation runs on two tracks:

- **Track A — project level** (tabular channel): PROMISE (Desharnais, COCOMO81,
  China, Kitchenham, Maxwell, Albrecht) + SEERA.
- **Track B — task level with text** (semantic channel): **gate-valid: JOSSE
  and SiP** (real effort in seconds / hours). Deep-SE is **secondary
  evidence** for the semantic retrieval, not a gate dataset (see sec. 5).
  TAWOS as retrieval scale only (not part of the gate).

## 2. Split policy

Split method is assigned per dataset by feasibility, verified 2026-06-12
(scripts feeding data_dictionary.md). Never a plain random split.

- **Rolling-origin temporal CV** (v1.2, supersedes the single temporal split
  for the gate verdict) — expanding-window cross-validation, strictly
  past→future, never random. Used where a usable date exists (≥5 distinct
  values): desharnais, kitchenham, maxwell, seera (Track A), sip (Track B).
  Full parameters frozen in sec. 8. The single 60/20/20 temporal split is
  retained and reported alongside for transparency, but the gate verdict reads
  the rolling-origin result.
- **Group-by-project split** — train on a set of projects, test on held-out
  projects (grouped k-fold, no project spans train and test). Used for the
  dateless Track B datasets that carry a project field: deepse (14 projects),
  josse (371 projects). This is deliberately the cold-start scenario — a new
  client/project with no in-project history — which is the hardest and most
  product-relevant case for Metis, so it is the right fallback here rather
  than a weaker proxy. The deepse issue-key suffix may be used as a secondary
  within-project temporal robustness check, not as the gate split.
- **Ordered k-fold CV** (fixed seed) — last-resort fallback for dateless
  Track A datasets with no project grouping: cocomo81, china, albrecht.
  Declared in the report with an explicit caveat that no temporal guarantee
  applies.

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
| Statistical baselines | engine beats median-by-category and log-size regression wherever computable |
| Expert baseline | engine beats the expert **in aggregate per track**, not per single dataset (see below) |

**Gate-valid datasets** (count toward the "2 datasets per track" rule):
- Track A: desharnais, cocomo81, china, kitchenham, maxwell, seera.
- Track B: **JOSSE and SiP only**.

Excluded from the gate count:
- **Deep-SE** — its effort label is story points, which measures agreement
  with each team's own pointing convention, not accuracy in hours. It serves
  as secondary evidence that semantic retrieval works, reported alongside but
  never as a gate threshold (decision 2026-06-12).
- **Albrecht** (24 rows) — too small; kept for completeness.
- **TAWOS** — retrieval scale only, by design.

**Beating the expert — aggregate, not per-dataset (decision 2026-06-12).**
The expert comparison is judged on the pooled expert-annotated test rows of a
track, not dataset by dataset. Rationale: on small datasets (e.g. kitchenham,
~29 test rows) the expert's PRED(25) of 65.5% carries a wide confidence
interval, so a per-dataset hard gate would let noise drive a go/no-go call.
The G0 minute still reports the per-dataset expert comparison next to the
aggregate, for full transparency.

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

## 7. Decisions log

Frozen at v1.1 (2026-06-12):
- [x] Sources and licenses per dataset verified (registry.py)
- [x] Data dictionary per dataset (reports/data_dictionary.md)
- [x] Split method assigned per dataset by feasibility (sec. 2)
- [x] TAWOS subset declaration (sec. 6)
- [x] G0 decider = founder; thresholds fixed, not revisable downward (header)
- [x] "Beat the expert" judged in aggregate per track, not per dataset (sec. 5)
- [x] Deep-SE excluded from the gate count (story-point label) (sec. 5)

Amended at v1.2 (2026-06-12):
- [x] Rolling-origin temporal CV as the gate estimator, pre-registered (sec. 8)
- [x] Gate regressor = GBM, GP fallback by test-blind train-internal validation

Outside the freeze (reporting only, do not affect thresholds):
- [ ] Confirm Albrecht effort unit against the literature
- [ ] Confirm SEERA effort unit against "SEERA dataset attribute formulas.pdf"

These two units cancel inside PRED(25)/MdAPE (both are ratios), so they are
needed only for the euro conversion in the final report, not for the gate.

## 8. Amendment log

### v1.2 (2026-06-12) — rolling-origin temporal CV for the gate verdict

**This section is pre-registered: it is committed BEFORE the rolling-origin
code is written or any rolling-origin number is seen. The commit timestamp is
the pre-registration.** Every degree of freedom is fixed here, in advance.

**Outcome-independent motivation.** The single 60/20/20 temporal split leaves
test blocks of 12–29 rows on the Track A temporal datasets (n=62–145). At that
size the bootstrap CI on PRED(25) spans ~30–40 points and the binomial CI on
coverage is ~±13 points — the gate criteria ("within 5 points of nominal",
beat-the-baseline) are literally not measurable. Rolling-origin (expanding
window) CV is the literature-standard temporal evaluation for small samples:
it multiplies both test and training coverage while remaining strictly
past→future. This change improves the *estimator*; it does not touch the
*thresholds* (sec. 5), which are immovable. The justification holds identically
whether the result lands at 53% or 62% — it is adopted because single-split is
underpowered, not because of any observed number. **Rolling-origin does not
guarantee a PASS:** if the engine stays below 55% under the correct estimator,
that is the honest gate result, and the response is to iterate or declare
Track A partial — the difference is that "fail" now means something, where
before it only meant "cannot measure".

**Frozen parameters (rolling-origin temporal CV).**
- Scope: the temporal datasets (desharnais, kitchenham, maxwell, seera, sip).
  Dateless datasets keep their v1.1 split (China ordered k-fold; deepse/josse
  group-by-project).
- Initial training window: the earliest **50%** of the chronologically ordered
  rows.
- Folds: the remaining 50% is cut into **5** consecutive equal test blocks
  (expanding window — each fold trains on every row before its test block).
  The last fold absorbs any remainder.
- Minimum test rows per fold: **5**. If 5 folds cannot each hold ≥5 rows,
  reduce the fold count to `floor(back_rows / 5)`, with a floor of **3** folds.
- Calibration block (for CQR): the most recent **20%** of each fold's training
  window, immediately preceding that fold's test block (temporally valid).
- Aggregation: **pooled** — out-of-fold test predictions are concatenated
  across folds, and the metric (plus its 2000-resample percentile bootstrap CI,
  seed 0) is computed once on the pool. Per-fold metrics are not averaged
  (a 6-row fold has a meaningless per-fold PRED).
- Hyperparameters: **fixed**, the conservative settings already in
  `track_a/gp.py` and `track_a/gbm.py`. No tuning of window, fold count, or any
  model hyperparameter against any test fold.

**Gate regressor selection (pre-registered).** GBM is the primary gate
regressor (v1.2 Step 2 decision: it gives the only confirmable beat-the-baseline
and the native p50/p90 quantiles CQR needs). Per dataset, the gate falls back
to **GP** when GBM underperforms the log-size baseline on **train-internal
temporal validation** — the last 20% of that dataset's training rows, scored
before any test fold is touched — which flags GBM overfitting on tiny samples
(the cold-start doctrine: GP for <1000 projects). The criterion is test-blind
and applied mechanically. Both GP and GBM out-of-fold results are reported for
every dataset regardless; the verdict reads the registered regressor.

**Reporting.** The public report shows single-split and rolling-origin results
side by side; the single-split numbers stay in the benchmark permanently. The
gate verdict is read off the rolling-origin result. Thresholds unchanged:
PRED(25) ≥ 55%, MdAPE ≤ 22%, coverage within 5 points of 90%.
