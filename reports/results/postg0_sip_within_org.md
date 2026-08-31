# Post-G0 — SiP as a within-organisation corpus: is the expert's error learnable?

> **Post-G0 R&D — NOT gate-carrying.** The G0 verdict (NO-GO on the public gate)
> stands and is not revisited. This experiment consumes recorded expert estimates,
> which `protocol.md` §5 places permanently outside the gate, and it never reads the
> sealed SEERA holdout.
> Reproduce: `PYTHONPATH=src python scripts/run_sip_within_org.py`.

## Question

`reports/REPORT.md` §7 names one honest gap: every G0 split was a *cross-organisation*
cold start, and the regime a product is deployed in — one organisation's own history,
one team, one set of logging conventions — was never measured. SiP is that regime in
public form: **one company, 12,299 estimate events over ten years (2004-2014), 22
developers, 20 project codes, an expert estimate on 100% of rows.** The gate used it as
a text dataset only.

The question is deliberately not G0's. G0 asked whether a model can estimate effort from
scratch; the answer was no. This asks whether the organisation's **own estimation error**
is conditionally learnable from context available at estimation time:

> target = log(actual / expert estimate), features = leakage-safe task context,
> split = rolling-origin on the estimation date, baseline = the expert alone.

## Design (all four arms scored on the same out-of-fold rows)

| arm | prediction |
|---|---|
| `expert` | `HoursEstimate`, untouched — the bar |
| `shift` | `HoursEstimate x exp(mean train log-ratio)` — one global factor, i.e. exactly what metis-api's local recalibration does today |
| `corrected` | `HoursEstimate x exp(f(x))`, `f` a gradient-boosted model of the log-ratio fitted on the train block only |
| `model_only` | a GBM on `log(effort)` from the same features **without** the expert anchor — the control |

Split: rolling-origin on `EstimateOn`, the frozen v1.2 parameters the gate used for SiP
(initial train fraction 0.5, 5 folds, min 5 test rows per fold). Every fold trains
strictly on its past; categorical level sets come from the train block, so an unseen
level at test time becomes a missing value rather than a silently reused code.

### Leakage split, declared before fitting

Admissible (knowable when the estimate is made):

- `HoursEstimate` — the estimate itself — the anchor being corrected
- `Priority` — assigned when the task is raised, before it is estimated
- `Category` — task class (Development / Management / Operational), set at raise time
- `SubCategory` — finer task class (Bug, Enhancement, Progress Meeting, ...), set at raise time
- `ProjectCode` — which project the task belongs to — known upfront
- `ProjectBreakdownCode` — work-package within the project — known upfront
- `RaisedByID` — who raised the task — known upfront
- `AssignedToID` — who the task is assigned to — known at estimation time
- `AuthorisedByID` — who authorised it — known upfront (missing on 65% of rows, encoded as such)

Inadmissible:

- `HoursActual` — the target
- `DeveloperID` — who ACTUALLY did the work — an assignment can change after the estimate
- `DeveloperHoursActual` — derived from the actual
- `TaskPerformance` — estimate minus actual, by construction the answer
- `DeveloperPerformance` — derived from realised work, the SEERA 'Programmers capability' trap
- `StatusCode` — the task's FINAL status — unknown while it is being estimated
- `StartedOn` — post-estimation timestamp
- `CompletedOn` — post-estimation timestamp
- `Summary` — admissible in principle, but this module is the TABULAR channel; the text channel was measured at G0 and is not re-litigated here

`DeveloperPerformance` and `TaskPerformance` are the SiP equivalents of the SEERA
attributes the G0 audit threw out: both are computed from the realised work.

## Results — pooled out-of-fold (n = 6,149)

| arm | PRED(25) [CI95] | MdAPE [CI95] | ΔPRED(25) vs expert [CI95] | McNemar p |
|---|---|---|---|---|
| expert alone (the bar) | 41.6% [40.3%, 42.8%] | 37.0% [34.1%, 40.0%] | — | — |
| expert x global factor (mean) | 40.7% [39.5%, 42.0%] | 39.0% [37.3%, 40.4%] | -0.8% [-1.2%, -0.5%] | 2.627e-05 (b=49, c=101) |
| expert x global factor (median) | 41.6% [40.3%, 42.8%] | 37.0% [34.1%, 40.0%] | +0.0% [+0.0%, +0.0%] | 1 (b=0, c=0) |
| expert x conditional f(x) | 29.0% [27.9%, 30.2%] | 52.5% [50.7%, 54.3%] | -12.6% [-13.9%, -11.2%] | 3.021e-72 (b=561, c=1333) |
| model alone, no anchor | 13.0% [12.2%, 13.9%] | 86.7% [84.4%, 88.2%] | -28.5% [-30.0%, -27.1%] | 1.184e-271 (b=469, c=2223) |

Achieved power: 1,894 discordant pairs out of 6,149 test rows
(p_disc = 0.308); the minimal detectable effect at 80% power is
pi1 = 0.533. For contrast, the G0 assist test that
forced the assist claim to be withdrawn had ~11 discordant pairs and power ~0.16.

## Sensitivities

**Excluding exact ties.** On 26.3% of rows the actual
equals the estimate to the digit — a logging convention (the estimate copied forward at
close) as much as a measurement, and it flatters every arm that stays near the anchor.
Dropped (n = 4,533):

| arm | PRED(25) [CI95] | MdAPE [CI95] | ΔPRED(25) vs expert [CI95] | McNemar p |
|---|---|---|---|---|
| expert alone (the bar) | 20.7% [19.5%, 21.9%] | 55.6% [52.4%, 56.2%] | — | — |
| expert x global factor (mean) | 19.6% [18.5%, 20.7%] | 56.3% [53.8%, 58.0%] | -1.1% [-1.7%, -0.6%] | 2.627e-05 (b=49, c=101) |
| expert x global factor (median) | 20.7% [19.5%, 21.9%] | 55.6% [52.4%, 56.2%] | +0.0% [+0.0%, +0.0%] | 1 (b=0, c=0) |
| expert x conditional f(x) | 20.8% [19.6%, 21.9%] | 61.8% [59.6%, 63.4%] | +0.0% [-1.4%, +1.4%] | 1 (b=561, c=560) |
| model alone, no anchor | 13.2% [12.2%, 14.2%] | 81.7% [80.5%, 83.7%] | -7.5% [-9.0%, -6.0%] | 8.94e-22 (b=469, c=811) |

**Tasks of 4 hours and up.** The median SiP task is 3 hours and estimates are recorded
as integers, so at the small end a 25% band is narrower than the recording granularity
(n = 2,717):

| arm | PRED(25) [CI95] | MdAPE [CI95] | ΔPRED(25) vs expert [CI95] | McNemar p |
|---|---|---|---|---|
| expert alone (the bar) | 29.9% [28.2%, 31.7%] | 50.0% [47.4%, 52.1%] | — | — |
| expert x global factor (mean) | 29.8% [28.1%, 31.6%] | 50.0% [47.8%, 52.4%] | -0.1% [-0.7%, +0.5%] | 0.8072 (b=32, c=35) |
| expert x global factor (median) | 29.9% [28.2%, 31.7%] | 50.0% [47.4%, 52.1%] | +0.0% [+0.0%, +0.0%] | 1 (b=0, c=0) |
| expert x conditional f(x) | 21.3% [19.8%, 22.9%] | 69.4% [66.6%, 71.7%] | -8.6% [-10.5%, -6.7%] | 1.07e-17 (b=259, c=493) |
| model alone, no anchor | 11.6% [10.5%, 12.9%] | 83.0% [81.2%, 85.1%] | -18.3% [-20.4%, -16.3%] | 5.009e-64 (b=208, c=706) |

## Reading: NO CORRECTION BEATS THE EXPERT

Neither arm improves on the organisation's own estimator. Three findings sit
underneath that headline, and the second and third are the ones worth carrying forward.

**1. The correction does not beat the expert, and on the full set it is far worse.**
Conditional correction lands -12.6% [-13.9%, -11.2%]
against the expert. Once the exact ties are removed the gap closes to
+0.0% [-1.4%, +1.4%] — a clean null at
n = 4,533 with 1,121
discordant pairs, so this is a *measured* tie, not an underpowered one. The size
structure that is plainly visible in the raw data (small tasks overrun, large tasks come
in under) is real in-sample and does not survive out-of-sample transfer across the
timeline. Conditioning on task context does not recover the organisation's own error.

**2. A single global recalibration factor makes an unbiased estimator worse.**
The shift arm is -0.8% [-1.2%, -0.5%]
(McNemar p = 2.6e-05) — small, but significantly negative, and
the factor here was fitted on thousands of past rows rather than the handful a deployment
would have. This company's median log-ratio is 0.000: its estimator is already unbiased
in the median, and multiplying it by a mean-driven constant only moves it off target.
The product consequence is direct. `metis-api` applies exactly this transform,
unconditionally, from three recorded actuals. On an organisation like this one it would
subtract accuracy. A correction fitted on n = 3 and applied at full strength is the
failure mode this arm demonstrates at n = thousands; the mitigation is to shrink the
local delta toward zero rather than to trust it whole.

**3. The expert's SiP baseline is substantially a logging convention.**
On 26.3% of test rows the actual equals the estimate to the digit. The expert's
headline 41.6% PRED(25) — the number G0 reports, reproduced here exactly — falls
to 20.7% once those rows are dropped. This does not overturn anything: the model
still loses on both readings, and "the human beats every model" survives. But the
*margin* is smaller than the headline suggests, and any future comparison against a SiP
expert baseline should quote both numbers.

**What the control says.** The no-anchor model reaches
13.0%, in line with the G0 semantic channel. The
expert anchor is carrying essentially all of the accuracy in every arm that has one —
consistent with the post-G0 IVW finding that any blend with the current model makes the
expert alone worse.

**Consequence for the pilot.** The within-organisation hypothesis is not refuted in
general: this is one company, at task granularity (median 3 h), with a task-tracker's
context and no project-level features. But the cheapest version of the hypothesis — that
an organisation's estimation bias is a stable, learnable function of the context a
tracker records — is now measured, once, at real power, and it does not hold. A pilot
should be sold on the calibrated interval, which G0 did demonstrate, and not on an
expected point-accuracy lift.

### What this does and does not establish

- It is **one organisation**, not a sample of organisations. A result here says the
  effect exists on this company's decade of history; it does not say it generalises to
  the next client. That is still what a pilot is for.
- The unit is the *estimate event*, and tasks can be re-estimated. That is the
  protocol's declared unit for SiP and it is unchanged here.
- Task granularity is small (median 3 h). The 4-hour stratum above is the honest read
  for anything the product would price.
- Nothing here touches G0's two closed claims: cross-organisation tabular attributes
  and requirement text remain measured, and remain below the bar.
