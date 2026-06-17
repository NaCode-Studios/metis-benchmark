# Track B — semantic channel (rolling-origin / group-by-project, CI95)

> 2026-06-18. Pre-registered v2.0. Engine = k-NN Nadaraya-Watson over
> bge-small-en-v1.5 embeddings (k/τ test-blind). Gate datasets: JOSSE
> (group-by-project, cross-project cold start), SiP (rolling-origin). Deep-SE
> secondary (story points). Out-of-fold, bootstrap CI95.
> Reproduce: `python scripts/run_track_b.py`, `scripts/run_ceiling_b.py`.

## Phase 3 — bi-encoder retrieval vs the text-blind floor

| Dataset | n test | engine PRED(25) | global-median (floor) | engine − floor | expert |
|---|---|---|---|---|---|
| deepse (secondary, story pts) | 21,064 | 22.5% [21.9, 23.0] | 19.8% | **+2.7% [2.0, 3.4]** | — |
| **SiP** (gate) | 6,149 | 17.9% [16.9, 18.8] | 18.0% | −0.1% [−1.5, 1.2] | 41.6% [40.3, 42.8] |
| **JOSSE** (gate) | 23,186 | 14.4% [14.0, 14.9] | 17.2% | −2.8% [−3.4, −2.2] | 45.0% [43.4, 46.4] |

**Finding.** On story points (Deep-SE) the text carries a small but real signal
(+2.7%, CI clears zero). On the **gate datasets — real logged effort — the
semantic channel does NOT beat the text-blind median**: a statistical tie on
SiP, significantly *worse* than the floor on JOSSE. The human expert is far
ahead on both (41–45% vs the engine's 14–18%).

**Why** — the pre-registered caveat, confirmed: JOSSE/SiP effort is *logged
time* (JIRA timespent / hours). Two semantically near-identical tasks take very
different logged time depending on developer, interruptions, and logging habits,
especially across projects (cold start). Semantic similarity of the requirement
text does not predict logged time; the human estimator, who knows the team and
codebase, does much better.

## Phase 4 — cross-encoder reranking: does not help (wrong tool)

The pre-registered cross-encoder (`ms-marco-MiniLM-L-6-v2`) is a query→passage
reranker, not a symmetric task-pair similarity model: on a relevant vs an
irrelevant task pair it returns near-identical saturated scores (−11.39 vs
−11.43). On a 700-task SiP sample, reranking the bi-encoder's top-30 candidates
*lowered* PRED(25) from 21.6% to 16.9%. Reranking with this model degrades
retrieval; it is the wrong instrument for symmetric task similarity. (A
symmetric cross-encoder fine-tuned on effort pairs is a Phase-1 R&D item, not a
public-benchmark fix.)

## Phase 5 — honest feasibility ceiling: STOP RULE TRIGGERED (2/2 gate datasets)

Best out-of-sample PRED(25) over every text-based method —
`python scripts/run_ceiling_b.py`:

| Method | SiP | JOSSE |
|---|---|---|
| k-NN-NW (bge embeddings) | 17.9% [16.9, 18.8] | 14.4% [14.0, 14.9] |
| emb→GBM (learned) | 19.5% [18.6, 20.5] | 15.0% [14.5, 15.5] |
| emb→RF (learned) | 18.8% [17.8, 19.8] | 14.8% [14.4, 15.3] |
| TF-IDF k-NN (lexical) | 19.6% [18.6, 20.6] | 14.2% [13.8, 14.7] |
| **honest ceiling** | **19.6%** | **15.0%** |
| global-median floor | 18.0% | 17.2% |
| expert | 41.6% | 45.0% |

**Both gate datasets fall below 55% — the stop rule fires.** No text-based
method (retrieval, learned regression on embeddings, or lexical TF-IDF) clears
~20%. On **JOSSE the entire ceiling (15.0%) is below the text-blind floor
(17.2%)**: the requirement text is *worse* than predicting the median. The
honest ceiling is 15–20%, far under the 55% threshold, and the human expert
(41–45%) dominates everything.

**Coverage (CQR).** Not the binding criterion here — Track B fails decisively on
point accuracy, not on interval calibration. The conformal layer holds coverage
by construction (demonstrated on the same machinery in Track A, 94.6%), so it
does not change the verdict.

## Verdict: Track B does NOT pass — semantic channel fails the gate on public logged-time data

The pre-registered, leakage-audited evaluation shows the semantic channel
**cannot estimate real (logged) effort** on the public gate datasets: it ties
or loses to the text-blind median, the honest ceiling is ~15–20% (55%
unreachable, demonstrated), and the human expert is far ahead. Cross-encoder
reranking made it worse (wrong model class). The one positive — text adds a
small signal on Deep-SE story points — does not count for the gate (Q8) and
does not transfer to hours/seconds.

**Why this is a real result, not a setup failure:** embeddings are correct
(semantic geometry verified), the pipeline is leakage-free (cross-project index,
test-blind k/τ), four method classes were tried, and the ceiling was measured
honestly. The limiting factor is the data: logged time is dominated by
developer/team/logging noise that the requirement text does not carry —
especially cross-project (cold start). This is the harder, product-relevant
setting the protocol deliberately chose.

