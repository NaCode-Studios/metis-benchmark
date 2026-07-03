# Can a model estimate software effort from public data? A pre-registered blind test

**A reproducible benchmark of statistical effort estimation on open datasets — and an honest negative result.**

NaCode Studios · 2026 · [metis-benchmark](https://github.com/NaCode-Studios/metis-benchmark)

---

## TL;DR

We built the test we would want any "AI estimates your software effort" vendor
to pass — and ran it on ourselves, with the thresholds, splits, features and
stop rule **pre-registered (committed to git) before each gate experiment**.
Across **nine public datasets** and two channels (tabular project attributes and
the natural-language text of requirements), **no model class clears the
precision bar** of PRED(25) ≥ 55% out-of-sample. The honest feasibility ceiling
— the best out-of-sample accuracy any flexible model reaches under a
leakage-free, time-ordered evaluation — has a central estimate of **~42–52% on
the tabular development datasets (33.9% on the sealed SEERA holdout)** and
**~15–20% on requirement text**, in every case below 55% (the small tabular
sets carry wide confidence intervals whose upper bounds reach into the 50s–60s).
Where a dataset records a *strong* human estimate (Kitchenham 61.6%, JOSSE 45%,
SiP 41.6%) the expert beats every model; on the SEERA holdout the recorded
expert is itself weak (25.4%) and the model ties it.

Two things survive the negative:
1. **The intervals are honest.** Conformalized prediction keeps its promise —
   a nominal-90% interval covers the truth **94.6%** of the time on unseen
   projects, distribution-free (in fact slightly conservative: a kept, if loose,
   guarantee).
2. **The method is the contribution.** Pre-registration, an adversarial leakage
   audit, rolling-origin evaluation, and a pre-committed stop rule turn "we
   couldn't get a good number" into "55% is below the honest ceiling on every
   gate dataset" — a difference that matters.

This is not a product pitch. It is the blind test, reported straight.

---

## 1. Why this benchmark exists

Software effort estimation is one of the most expensive unsolved problems in the
industry. Large IT projects run **+45% over budget on average** (McKinsey–Oxford,
5,400+ projects); only ~30% of software projects fully succeed on time, budget
and scope (Standish CHAOS). A wave of AI tooling now claims to quantify effort,
cost and team size from a specification or a backlog.

The claim is testable. The effort-estimation research community has published
datasets with **real recorded effort** for decades. So before building a
product on the premise that a model can estimate effort, we asked the falsifiable
version of the question:

> On public data with real effort, can a statistical engine — similarity search
> plus regression with calibrated intervals — beat the standard baselines and
> the human expert, out-of-sample, without leakage?

We pre-registered the thresholds, the splits, the features, and a stop rule, and
committed each before running the corresponding *gate* experiment. The git
history is the pre-registration. (The 55%/22% bar was fixed after a round of
baseline and expert *smoke tests* that showed where the bar realistically sat —
e.g. the expert reaching 65.5% on Kitchenham — but before any gate-model run.)

## 2. What "passing" means (frozen before results)

| Criterion | Threshold |
|---|---|
| **PRED(25)** — share of estimates within 25% of actual | ≥ 55% on ≥ 2 datasets per channel |
| **MdAPE** — median absolute percentage error | ≤ 22% on ≥ 2 datasets per channel |
| **Coverage** — actuals inside the nominal-90% interval | within 5 points of 90% |
| **Baselines** | beat median-by-category and log-size regression |
| **Expert** | beat the recorded human estimate, in aggregate |

PRED(25) and MdAPE are the standard metrics of the effort-estimation literature,
so results are directly comparable with published papers. The 55%/22% bar is
deliberately *more lenient* than the pilot KPIs we set for proprietary data —
public datasets are more heterogeneous than one organization's own history.

## 3. The data

Two channels, nine datasets, all public and openly licensed.

**Channel A — tabular, project level** (function points, cost drivers):
PROMISE Desharnais (81), COCOMO81 (63), China (499), Kitchenham (145),
Maxwell (62), Albrecht (24), and SEERA (120). Gate datasets exclude Albrecht
(too small) and China (no signal — see below).

**Channel B — semantic, task level** (the requirement text): JOSSE (23,186 JIRA
tasks, effort in logged seconds; Alhamed & Storer 2022) and SiP (12,299 task
estimates with developer hours; Jones & Cullum 2019) are the gate datasets;
Deep-SE (21,064 issues with story points; Choetkiertikul et al. 2019) is
secondary evidence (story points are not hours). TAWOS is reserved for
retrieval scale only.

SEERA was sealed as a **final holdout**: its data was never used to choose
features, models, or hyperparameters — only its schema was read, to audit for
leakage. It was opened exactly once, for the verdict.

## 4. Method — the parts that make a negative trustworthy

A negative result is only worth publishing if the evaluation is sound. Four
choices do the work:

**Time-ordered evaluation, never random.** Random train/test splits let a model
peek at the future and inflate every metric. We use **rolling-origin
(expanding-window) cross-validation** on dated datasets and **leave-projects-out
(grouped) k-fold** on the task datasets — the latter is deliberately the
*cold-start* case: a new project retrieves only from *other* projects' history,
exactly the situation a vendor faces with a new client.

**An adversarial leakage audit.** Many dataset attributes secretly encode the
outcome. SEERA's `Requirements stability` is computed from "modifications during
testing and deployment"; `Team continuity` counts developers who *left the
project*; `Programmers' capability` is built from realized work-accuracy. None of
these are knowable before a project starts. An **independent three-way automated
audit** classified all 76 SEERA attributes against their published formulas and
we kept only those that survive unanimously — **51 of 76** (the audited feature
sets are fixed in [`reports/protocol.md`](protocol.md) §8). A feature that
exists only in a finished-project dataset, not in a fresh specification, never
enters a gate model.

**Conformalized intervals (CQR).** We do not promise point estimates; we promise
ranges that hold. Conformalized Quantile Regression (Romano, Patterson, Candès
2019) wraps any model and calibrates a distribution-free, finite-sample interval.
We verify the *empirical* coverage, not the assumed one.

**A pre-committed stop rule.** Before seeing results we fixed: if the honest
feasibility ceiling — the best out-of-sample PRED(25) over a flexible model zoo
(Gaussian Process, gradient boosting, random forest, k-NN, TF-IDF) — is below
55% on more than one gate dataset, then 55% is *unreachable on this data* and the
channel closes as "threshold unreachable, demonstrated." This is what converts a
disappointing number into a real finding.

Bootstrap 95% confidence intervals accompany every metric: the tabular test
blocks are small (12–73 projects), and a single PRED(25) point on 29 projects is
noise without a band around it.

## 5. Results — Channel A (tabular)

Best engine = a log-space Gaussian Process (the column below); gradient
boosting, a size mean function, and hierarchical partial pooling across datasets
were all implemented and **did not help** (the GP already captures size through
its features).

| Dataset | n test | GP engine PRED(25) | Honest ceiling [CI95] | vs log-size baseline |
|---|---|---|---|---|
| Desharnais | 41 | 36.6% | 46.3% [31.7, 61.0] | beats (aggregate) |
| Kitchenham | 73 | 45.2% | 45.2% [34.2, 56.2] | **+12.3% [4.1, 20.6]** |
| Maxwell | 31 | 51.6% | 51.6% [35.5, 67.7] | **+22.6% [3.2, 45.2]** |
| COCOMO81 | 62 | 41.9% | 41.9% [30.6, 54.8] | beats (aggregate) |
| SEERA *(holdout)* | 59 | 25.4% | 33.9% | +3.4% |
| China *(no signal)* | 499 | 19.2% | ~21% | no lift (19.2% vs 22.0%) |

Aggregate over the four gate development datasets (n=207): **43.5% [36.7, 50.2]**,
beating the log-size baseline by **+12.1% [5.3, 18.8]** — a real, confident lift
over naive baselines. But **no dataset's central estimate reaches 55%**; the
honest ceiling's point estimate is below 55% on all five (including the sealed
holdout), which triggered the pre-registered stop rule. The small tabular sets
carry wide CIs (Maxwell's ceiling upper bound is 67.7%), so we do not claim 55%
is *impossible* on any single small dataset — we claim its central estimate is
below 55% everywhere, on data whose intrinsic noise floor is high: the best
model's out-of-sample residual scatter is **σ ≈ 0.6 in log space (±~60%
multiplicative)** on every dataset.

**Coverage holds:** a nominal-90% conformal interval covers
**94.6% [91.5, 97.7]** of unseen projects — in fact it *over*-covers at lower
nominal levels (the intervals are conservative), so the 90% promise is kept but
loose. The engine's *honesty* holds even where its *accuracy* does not.

The human expert, recorded on Kitchenham, scores **61.6%** — well above any
model. When the expert's own estimate is given to the model as an input (an
"assist" mode, deliberately barred from the gate because it is unavailable
cold-start), the model *refines* it from 61.6% to **68.5%**.

## 6. Results — Channel B (semantic)

The hypothesis: semantically similar requirements have similar effort, so
embedding the text and retrieving nearest neighbours should beat blind baselines.
We embed with a local sentence model (BGE-small), retrieve by cosine similarity,
and predict a kernel-weighted mean of neighbours' efforts.

| Method | SiP | JOSSE |
|---|---|---|
| Semantic k-NN (embeddings) | 17.9% | 14.4% |
| Learned regressor on embeddings (GBM) | 19.5% | 15.0% |
| Learned regressor on embeddings (RF) | 18.8% | 14.8% |
| TF-IDF k-NN (lexical) | 19.6% | 14.2% |
| **Honest ceiling** | **19.6%** | **15.0%** |
| Text-blind median (floor) | 18.0% | 17.2% |
| Human expert | 41.6% | 45.0% |

On real logged effort, **no text-based method beats the text-blind median**: a
statistical tie on SiP, and on JOSSE the *entire* ceiling (15.0%) is **below**
the floor (17.2%) — the requirement text is worse than guessing the median.
Cross-encoder reranking made it worse still (the standard `ms-marco` reranker is
a query-document model, wrong for symmetric task pairs). The expert (41–45%)
dominates everything.

The one positive: on **Deep-SE story points** the semantic channel beats the
floor by **+2.7% [2.0, 3.4]** — text carries *some* signal. But story points are
a team's own relative scale, not hours, and the signal does not transfer to
logged time.

**The likely mechanism:** logged time (JIRA timespent, developer hours) is
dominated by who did the work, interruptions, and logging habits — noise the
requirement text does not contain, especially across organizations (cold start).
Two identically-worded tasks take very different logged time on different teams.
We frame this as a result *about our four method classes on this data*, not a
proof that text can never help: a symmetric cross-encoder fine-tuned on effort
pairs, or within-organization data, could change the picture.

## 7. What the benchmark proves — and what it does not

**Proven, on public data:**
1. Generic, cross-organization effort estimation from public tabular attributes
   has a central honest ceiling of **~42–52% PRED(25)** (33.9% on the sealed
   holdout) — driven by a high intrinsic noise floor (σ≈0.6), not by weak
   modeling: richer models and a flexible zoo do not move it.
2. **No text method we tested — four classes** (semantic k-NN, two learned
   regressors on embeddings, lexical TF-IDF) — beats the text-blind median on
   logged effort (ceiling 15–20%). A symmetric cross-encoder fine-tuned on
   effort pairs is untried; the standard query-document reranker does not help.
3. **Calibrated intervals are achievable and honest** (94.6% coverage,
   distribution-free, slightly conservative).
4. **Where a strong expert estimate is recorded, the human beats every model**
   (Kitchenham 61.6%, JOSSE 45%, SiP 41.6%) — using team and codebase context
   the public data lacks. On the SEERA holdout the expert is itself weak (25.4%)
   and the model ties it.

**Not tested (the honest gap):** every gate split was *cross-organization cold
start* on *public* data. A single organization's **own history** — same team,
same logging conventions, same codebase — is precisely the regime public datasets
cannot proxy, and precisely where an estimation tool would be deployed. This
benchmark does not measure that case; it is the obvious next experiment, on
proprietary data, and we make no claim about it here.

## 8. Reproduce it

```bash
git clone https://github.com/NaCode-Studios/metis-benchmark
cd metis-benchmark
python -m venv .venv && source .venv/bin/activate
pip install -e ".[models,semantic,plots,dev]" -c constraints-g0.txt
python scripts/download.py            # public datasets (licenses in registry.py)
python scripts/run_track_a_rolling.py # tabular channel
python scripts/run_ceiling.py         # tabular honest ceiling + stop rule
python scripts/embed_track_b.py       # cache embeddings (slow, once)
python scripts/run_track_b.py         # semantic channel
python scripts/run_ceiling_b.py       # semantic honest ceiling + stop rule
make test                             # full test suite (all green)
```

Or, equivalently, `make reproduce-g0` after installing — it chains the same
scripts in the same order.

**Pinned environment.** `constraints-g0.txt` freezes the exact package set the
G0 numbers were produced with (`pip freeze` of the benchmark venv,
CPython 3.14.4). Bootstrap seeds are fixed, but library upgrades can still
move third-decimal behavior (GP optimizer, XGBoost splits), so exact
reproduction requires the constraints file. Key versions:
`numpy==2.4.6`, `scikit-learn==1.9.0`, `xgboost==3.2.0`, `scipy==1.17.1`
(plus `pandas==3.0.3`, `MAPIE==1.4.1`).

Every number in this report is produced by these scripts. The evaluation
protocol, its amendments, and the pre-registration timestamps are in
[`reports/protocol.md`](protocol.md); the full per-phase results are in
[`reports/results/`](results/).

## 9. Sources

- McKinsey & Co. × University of Oxford — *Delivering large-scale IT projects on
  time, on budget, and on value* (5,400+ projects: +45% over budget, −56% value).
- Standish Group, CHAOS Report 2023 — project outcomes.
- PROMISE Repository (Zenodo mirrors, CC-BY): Desharnais, COCOMO81, China,
  Kitchenham, Maxwell, Albrecht.
- SEERA — *Software engineERing in SudAn* cost-estimation dataset (PROMISE/ACM 2020).
- JOSSE — Alhamed & Storer (2022), DOI 10.5281/zenodo.7022735.
- SiP — Jones & Cullum (2019), github.com/Derek-Jones/SiP_dataset.
- Deep-SE — Choetkiertikul et al. (2019), IEEE TSE; CSVs via SOLAR-group.
- Romano, Patterson, Candès (2019) — *Conformalized Quantile Regression*.

## 10. Errata

**2026-07-02 — desharnais row count and GP point estimate (§5).** Two fully
diagnosed causes, no verdict change (every value stays far below the 55% bar):

1. *Bug fix.* The "gate (selected)" series in `track_a/experiment.py` predicted
   from the raw (non-imputed) test matrix instead of the train-median-imputed
   one; on datasets with missing features it silently diverged from the
   selected regressor. Fixed and locked by `tests/test_experiment.py`. Impact
   on the rolling run: PRED(25) unchanged everywhere; gate MdAPE 29.6% → 28.5%
   (desharnais), 28.5% → 28.3% (aggregate).
2. *Row policy propagation.* Since v1.3, rows with missing features are
   imputed, not dropped; desharnais keeps its 4 rows with −1-coded missing
   experience fields (77 → 81 rows). The §5 table previously carried the
   pre-v1.3 desharnais snapshot: **n test 39 → 41, GP 38.5% → 36.6%**.
   Re-applying the old drop policy reproduces the old numbers exactly.

The aggregate was re-verified on 2026-07-02: **43.5% [36.7, 50.2]** and
**+12.1% [5.3, 18.8]** over log-size are unchanged at reported precision; the
pooled row count is corrected from 205 to **207**. Kitchenham, Maxwell,
COCOMO81 and China reproduce identically. The sealed SEERA holdout was not
re-opened. Full old→new detail in
[`results/track_a_rolling_origin.md`](results/track_a_rolling_origin.md) (ERRATA section).

---

*This report is an internal NaCode Studios artifact, written to be publishable.
Publishing the benchmark is a deliberate decision: a rigorous, honest negative
is a credibility asset, but it is the founder's call to make.*
