# Post-G0 — TAWOS: do a team's own story points predict its own outcomes?

> **Post-G0 R&D — NOT gate-carrying.** TAWOS is declared in the registry as
> retrieval-scale-only and stays outside the gate; the G0 verdict is untouched.
> Reproduce: `python scripts/extract_tawos.py` then
> `PYTHONPATH=src python scripts/run_tawos_within_project.py`.
> Dataset: TAWOS (MSR 2022), DOI 10.5522/04/21308124, Apache-2.0, research use only.

## Read the label before the numbers

TAWOS's `Total_Effort_Minutes` is derived from an issue's in-progress status
transitions. Its median is about **160 hours per issue**, which is a *lead time*, not
hours of work. The real effort label is JIRA's worklog field `Timespent`, and only
900 issues carry both it and a story point.

So the large table below measures the **estimate-to-outcome** relationship — does a
team's own point scale, calibrated on its own history, predict how long an issue
actually takes to get through? — and is **not comparable to a G0 PRED(25)**. The small
table is the correctly-labelled check.

This is still the product's question in the regime that matters: one team, its own
estimates, its own history, a strictly temporal split, and a baseline that is the team
calibrating itself rather than a model.

## Design

Per project, rolling-origin on the estimation date (the frozen v1.2 parameters), folds
with fewer than 30 training rows dropped, out-of-fold rows pooled. Story points are not
hours, so the baseline converts them through the **train block's own median
outcome-per-point** — the conversion a client would actually make. The estimate-blind
floor is the train block's median outcome.

Admissible context: issue type, priority, reporter, assignee, sprint. Status,
resolution and every post-estimation timestamp are ex-post and never enter.

## Outcome = in-progress lead time (20,864 out-of-fold rows, 28 projects with >= 200 paired issues)

| arm | PRED(25) [CI95] | MdAPE [CI95] | ΔPRED(25) vs the rate [CI95] | McNemar p |
|---|---|---|---|---|
| story points x the team's own rate (the bar) | 12.5% [12.0%, 12.9%] | 82.7% [82.1%, 83.4%] | — | — |
| the team's median outcome (estimate-blind floor) | 10.3% [9.9%, 10.7%] | 86.3% [85.6%, 87.0%] | -2.2% [-2.7%, -1.7%] | 3.267e-15 |
| GBM on log(SP) + context | 11.8% [11.4%, 12.3%] | 86.5% [85.8%, 87.2%] | -0.6% [-1.2%, -0.0%] | 0.03576 |
| GBM on context alone, no story point | 10.7% [10.2%, 11.1%] | 89.2% [88.6%, 89.9%] | -1.8% [-2.4%, -1.2%] | 2.424e-09 |

## Outcome = logged work time (403 rows, 6 projects with >= 60 paired issues)

| arm | PRED(25) [CI95] | MdAPE [CI95] | ΔPRED(25) vs the rate [CI95] | McNemar p |
|---|---|---|---|---|
| story points x the team's own rate (the bar) | 16.4% [12.9%, 19.9%] | 65.7% [56.2%, 70.0%] | — | — |
| the team's median outcome (estimate-blind floor) | 19.6% [15.6%, 23.6%] | 71.9% [63.8%, 77.0%] | +3.2% [-1.7%, +8.4%] | 0.2414 |
| GBM on log(SP) + context | 13.9% [10.7%, 17.4%] | 75.7% [71.2%, 83.4%] | -2.5% [-7.4%, +2.5%] | 0.3682 |
| GBM on context alone, no story point | 16.4% [12.9%, 20.1%] | 80.7% [71.1%, 86.4%] | +0.0% [-4.7%, +5.2%] | 1 |

## Reading

**On lead time the team's own point scale does carry signal — and the model still does
not beat it.** Converting story points through the block's own median rate beats the
estimate-blind floor by +2.2% [+1.7%, +2.7%]
(McNemar p = 3.3e-15), so the points are not noise. But a gradient-boosted
model given those same points plus the issue's context lands
-0.6% [-1.2%, -0.0%] against the plain
conversion: no better, and the CI excludes any improvement worth selling. Dropping the
story point costs another point and a half, which is the same control result the other
two corpora give — the existing estimate is carrying the accuracy, not the model.

**On the correctly-labelled effort subset the points stop working entirely.** Against
logged work time the team's conversion scores 16.4% and the estimate-blind
floor scores 19.6% — the floor is *ahead*, though at n =
403 the interval straddles zero and the honest reading is a tie. That is G0's
Track B finding arriving from a different direction: story points measure agreement with
a team's own pointing convention, and that convention does not convert into hours.

**What the three corpora say together.** SiP (one company, 12,299 estimates), Apache
JIRA (~2,100 estimates across 180 teams) and TAWOS (43,108 estimates across 28 projects)
were asked the same question with the same shape of experiment, and answered the same
way: **the estimator that already exists is the one to beat, and none of the model
classes tried beats it.** What differs is only what a correction is worth — nothing on
SiP's unbiased estimator, a great deal on Apache's biased one, which is exactly why the
product's local recalibration must adapt rather than assume.

**The label limits what may be carried forward.** A lead time absorbs queueing, review
latency and holidays, none of which an effort estimate is accountable for. Nothing here
is comparable to a G0 PRED(25), and none of it is a Metis accuracy claim.
