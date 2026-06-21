# Evaluation protocol — Metis benchmark (Gate G0)

> Status: **v1.3 — FROZEN** (2026-06-12). Thresholds and rules below do not
> change after seeing results; any amendment must be recorded here with date
> and motivation, and may only tighten the gate, never loosen it.
> Note: the only experiments run before the v1.1 freeze were the *baseline*
> smoke tests, which set the bar. v1.2 amended the *estimator* (split
> methodology); v1.3 (W3) pre-registers the feature sets, the modeling
> approach (mean-function GP + hierarchical pooling), the SEERA holdout, and
> the stop rule — never the thresholds. See the amendment log (sec. 8).
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
| Coverage | share of actuals inside the conformal interval at the nominal level (p5–p95 for 90%) | `evaluation.empirical_coverage` |

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
and the native quantile predictions CQR needs — e.g. p5/p95 for the 90% interval). Per dataset, the gate falls back
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

### v1.3 (2026-06-12, W3) — feature sets, mean-function GP, hierarchical pooling, SEERA holdout, stop rule

**Pre-registered: committed BEFORE any W3 code. The commit timestamp is the
pre-registration.** v1.3 fixes the modeling approach and the new gate datasets;
it does not touch the thresholds (sec. 5), which remain PRED(25) ≥ 55%,
MdAPE ≤ 22% on ≥2 datasets, coverage within 5 points of 90%.

**Outcome-independent motivation.** W2 showed the engine below 55% out-of-fold
(~45–52%) while the in-sample ceiling is 66–78% (Q15): the shortfall is
generalization, driven by tiny per-dataset training folds and a zero-mean GP
that ignores the strong size→effort power law. The fixes below are standard,
literature-grounded ways to reduce that gap — a size mean function, partial
pooling across datasets (V20 p.9), and two more gate datasets — adopted because
they address generalization, not because of any observed number. They are
committed before being coded. They do not guarantee a PASS: the stop rule below
defines, in advance, the honest threshold under which Track A closes as
"55% unreachable, demonstrated".

**Gate datasets (Track A).** desharnais, kitchenham, maxwell, **cocomo81**,
**seera**. China stays out-gate (Q11, no signal). Albrecht stays out (24 rows).
Splits: rolling-origin for the dated datasets (desharnais, kitchenham, maxwell,
seera); ordered k-fold CV for cocomo81 (dateless).

**SEERA = final holdout.** SEERA's *data values* are not used to build features,
select models, tune hyperparameters, or calibrate anything. It is opened once,
at the final verdict (Phase 5). Only SEERA's *schema and attribute formulas*
(metadata) were read — by an adversarial leakage audit — to choose
leakage-safe columns; no effort value was seen. Development iterates on the
four non-holdout datasets only.

**Primary regressor: GP** (Q16), vindicated by rolling-origin in W2. GBM is
reported alongside. (The W2 pre-registered GBM-primary selection is superseded
here, before W3 results exist.)

**Feature sets (leakage-safe, fixed here).** Chosen by a 3-auditor adversarial
leakage audit reconciled conservatively (any credible leakage flag → exclude).
- **cocomo81** (23): the scale factors and effort multipliers
  `prec, flex, resl, team, pmat, rely, data, cplx, ruse, docu, time, stor,
  pvol, acap, pcap, pcon, apex, plex, ltex, tool, site, sced` plus `kloc`
  (size). Excluded as outcome/leakage: `effort` (target), `defects`, `months`.
  Ordinal ratings encoded vl<l<n<h<vh<xh → 1..6.
- **seera** (51): the organizational, contractual, sizing, planned-team,
  requirements-level and product-requirement attributes that are knowable
  before a project starts (full list transcribed into `track_a/features.py`).
  Excluded as leakage (17 unanimous + 2 conservative): every attribute whose
  SEERA formula references realized work or schedule — `Actual duration`,
  `% project gain (loss)`, `Users stability`, `Requirment stability`,
  `Requirements flexibility`, `Programmers capability`, `Analysts capability`,
  `Team continuity`, `Team cohesion`, `Team contracts`, `Schedule quality`,
  `Outsourcing impact`, `Process reengineering`, `Requirement accuracy level`,
  `Technical documentation`, `Comments within the code`, `User manual` — plus
  IDs, `Actual effort` (target), `Estimated effort` (expert baseline), and the
  non-primary size columns. size = `Object points`, expert = `Estimated effort`,
  date = `Year of project`.

**Encoding and imputation (fixed).** COCOMO ordinal ratings → integer codes as
above. SEERA Likert attributes coerced to numeric; `N/A`/`?` → missing.
Missing values imputed with the **training-fold median** (fit on train, applied
to test — never pooled across the split), so imputation leaks nothing.

**Mean-function GP (Phase 2).** The GP no longer models effort from zero mean.
The log-size power law `log(effort) = log(a) + b·log(size)` (the existing
`LogSizeRegression`) is fit on the training fold and subtracted; the GP models
only the residual in log space over the local features; predictions re-add the
baseline. With no residual signal this degrades gracefully to the log-size
baseline (asserted by a unit test on china).

**Hierarchical partial pooling (Phase 3).** A multilevel model partial-pools the
power-law coefficients `(a, b)` across datasets: each dataset has its own
`(a_d, b_d)` shrunk toward a global `(a_0, b_0)`. Differing effort units
(person-hours vs person-months) are absorbed by the per-dataset intercept
`a_d`; everything is in log space. Shrinkage strength `k` is estimated by
empirical Bayes from the between-dataset variance (V20 p.9 formula
`μ = (n·y + k·μ_sector)/(n + k)`), never by hand or arbitrary thresholds.
**Leakage constraint:** the global coefficients and `k` are estimated from
training rows only — no test-fold row of any dataset (the one under evaluation
or the others) enters the pooling. Rolling-origin order (past→future) is
preserved.

**Honest feasibility ceiling + STOP RULE (Phase 4).** The optimistic in-sample
RandomForest ceiling (Q15) is replaced by an *honest achievable* ceiling per
dataset: nested CV with the best model class plus an estimate of irreducible
noise (the effort/size productivity variance unexplained by the recorded
fields, as computed for china in Q11). **Pre-registered stop rule:** if this
honest ceiling is **below 55% PRED(25) on more than one gate dataset**, then 55%
is not statistically achievable on this data — Track A closes as "threshold
unreachable, demonstrated", and that is the Track A outcome (best-effort engine
+ assist mode kept as product; proceed to Track B). The decision is on the
numbers, not on optimism.

**Decisions log addition (v1.3):**
- [x] SEERA blocked as final holdout; schema-only metadata used for features
- [x] cocomo81 + seera wired as gate datasets; feature sets fixed by audit
- [x] GP primary (supersedes v1.2 GBM-primary, pre-W3-results)
- [x] Mean-function GP and hierarchical pooling pre-registered
- [x] Honest-ceiling stop rule pre-registered (Phase 4)

### v1.3 addendum (2026-06-12, pre-registered before pooling code) — exact pooling procedure

Refines the Phase-3 hierarchical pooling of v1.3 with the precise,
leakage-free estimation procedure, fixed before any pooled number is produced.

- **What is pooled: the slope only.** The power law is
  `log(effort) = a_d + b_d·log(size)`. The slope `b_d` (size→effort
  elasticity) is dimensionless and comparable across datasets, so it is
  partial-pooled toward a global `b_0`. The intercept `a_d` carries the unit
  (person-hours vs person-months) and stays **per-dataset, free** — never
  pooled — refit on each fold's training rows given the pooled slope.
- **Empirical-Bayes shrinkage.** Per-dataset slope `b_d` and its standard
  error (within variance `σ²_d`) come from an OLS log-log fit on training rows.
  The global `b_0` and between-dataset variance `τ²` use the DerSimonian–Laird
  random-effects estimator. The pooled slope is the precision-weighted shrink
  `b_d^pool = (b_d/σ²_d + b_0/τ²)/(1/σ²_d + 1/τ²)` — the V20
  `μ = (n·y + k·μ_sector)/(n+k)` form with `k = σ²_d/τ²`, estimated from data,
  not by hand.
- **Leakage-free global.** `b_0` and `τ²` are estimated only from rows that are
  **never test rows anywhere**: the initial training window (earliest 50%) of
  each rolling-origin dataset, which the expanding window always keeps in
  training. The dateless datasets (cocomo81, china) test every row across
  folds, so they **receive** shrinkage but **do not contribute** to the global.
  No test-fold row of any dataset enters the global. The global is fixed across
  folds.
- **At the verdict**, SEERA's initial window joins the global contributors;
  this is part of running the frozen pipeline once, not a development decision.

### v2.0 (2026-06-12) — Track B pre-registration (semantic channel)

**Pre-registered: committed BEFORE any Track B code. The commit timestamp is
the pre-registration.** Thresholds unchanged (sec. 5): PRED(25) ≥ 55%,
MdAPE ≤ 22% on ≥2 datasets, coverage within 5 points of 90%.

**Outcome-independent motivation.** Track A proved that project-level tabular
cost-driver features cap PRED(25) near ~50% (irreducible σ≈0.6). The V20 thesis
is that the *text* of requirements carries similarity signal those tables lack:
semantically close tasks should have similar effort (V20 Fig 2). Track B tests
exactly that, on task-level data with real effort and recorded human estimates.
This is Metis's actual differentiation; it is validated here on its own merits.
A negative is still possible and the stop rule below defines it in advance.

**Gate-valid datasets (exactly two — both must clear the per-dataset bar):**
- **JOSSE** — JIRA tasks (Apache/JBoss/Spring), effort = `actual_effort` in
  seconds (JIRA timespent), ~19% carry an expert estimate. Split:
  **group-by-project** (371 projects) — cross-project cold-start, the hard and
  product-relevant case (a new client retrieves only from other projects).
- **SiP** — 10k task estimates with `HoursActual` and a developer estimate
  (`HoursEstimate`, the expert) for every task. Split: **rolling-origin**
  temporal CV (has dates 2004–2014).

**Secondary (not gate):** Deep-SE (story points, group-by-project, 14 projects)
— evidence the retrieval works on text, reported alongside, never a gate number
(Q8). TAWOS stays retrieval-scale-only, out of this validation.

**Caveat (recorded up front):** JOSSE and SiP effort are *logged* time, which
is noisier than project ledgers. Higher irreducible noise is expected; the
honest ceiling (below) will quantify whether 55% is reachable on this data.

**Semantic engine (the candidate).**
- Embeddings: **BAAI/bge-small-en-v1.5** (384-d, local, normalized) over the
  task text (title + description; `corpus` for JOSSE). Deterministic, no
  training → no embedding leakage. (The V20 "BGE via ONNX" model family; ONNX
  is a deployment detail, irrelevant to validation.)
- Retrieval + prediction: k-NN by cosine similarity over the training index,
  Nadaraya–Watson kernel-weighted mean in log space —
  `ŷ = exp(Σ wᵢ·log(yᵢ) / Σ wᵢ)`, `wᵢ = exp((sᵢ − 1)/τ)` (V20). `k` and `τ`
  are selected per dataset by **test-blind train-internal validation** from a
  fixed grid (k ∈ {5,10,20,50}, τ ∈ {0.05,0.1,0.2}), never against the test.
- Reranking (Phase B4, pre-registered enhancement): a cross-encoder
  (`cross-encoder/ms-marco-MiniLM-L-6-v2`) re-scores the top-50 cosine
  candidates before weighting; reported as a separate variant.

**Baselines (mandatory, same discipline as Track A).** Global median
(text-blind floor — the engine must beat it to prove text carries signal);
median-by-category where present (SiP `Category`; JOSSE project); and the
**human expert** (JOSSE ~19%, SiP all) — the binding bar, judged in aggregate
per track (Q7). An embedding→GBM regressor is also reported, to compare learned
regression on embeddings against k-NN retrieval.

**Leakage discipline.**
- Group-by-project (JOSSE, Deep-SE): the retrieval index excludes every
  held-out project entirely — a test task retrieves only from other projects.
- Rolling-origin (SiP): the index holds past tasks only; future never leaks.
- k/τ (and any reranker threshold) selected on train-internal validation only.
- Large datasets are embedded in full (JOSSE ~23k, SiP ~12k); no subsampling
  of gate datasets.

**Honest ceiling + STOP RULE (same as W3).** The honest ceiling per gate
dataset = best out-of-sample PRED(25) over a zoo {k-NN-NW (best k/τ),
embedding→GBM, embedding→RF, TF-IDF k-NN (lexical reference)}. If the ceiling
is below 55% on both gate datasets, 55% is unreachable on this data and Track B
closes as "threshold unreachable, demonstrated". Decision on the numbers.

**Full G0.** Track A is closed not-passing (W3). If Track B passes its gate,
the full G0 is a partial/conditional pass on the semantic channel; if Track B
also fails, the G0 is a documented no-go on public data, and the product
decision (assist mode, within-project vs cold-start, Phase-1 pilot to get
proprietary data) is taken on the evidence. Repo publication still requires a
genuine pass (Q6).

**Decisions log addition (v2.0):**
- [x] Track B datasets/splits fixed: JOSSE (group-by-project), SiP (rolling-origin)
- [x] Embedding model fixed: bge-small-en-v1.5, local, normalized
- [x] Predictor: k-NN Nadaraya–Watson (log space), k/τ test-blind selected
- [x] Honest-ceiling stop rule pre-registered for Track B
