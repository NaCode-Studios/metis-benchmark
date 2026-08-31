# Post-G0 — Apache JIRA: the SiP question, asked of many teams

> **Post-G0 R&D — NOT gate-carrying.** Mined after the gate closed, consumes recorded
> expert estimates (permanently outside the gate per `protocol.md` §5), never reads the
> sealed SEERA holdout. The G0 verdict stands untouched.
> Reproduce: `python scripts/mine_apache_jira.py` then
> `PYTHONPATH=src python scripts/run_apache_jira_within_org.py`.

## Why

`postg0_sip_within_org.md` measured, on one company's decade of history, that correcting
its own estimates from its own task context does not beat leaving them alone. The fair
objection is that SiP is *one* organisation: one logging culture, one estimation habit.

Apache's public JIRA answers the objection cheaply. Time tracking there is optional, so
only **2,094 issues across 180 projects** carry both an
original estimate and logged time — small, but they are many independent teams rather
than one, and both halves of the pair are recorded rather than reconstructed.

Same four arms, same rolling-origin split, same declared leakage rule as the SiP run:
the ex-post half (time spent, resolution date, final status) never enters the features;
project, issue type, priority, reporter, assignee and component count all exist when the
issue is filed.

## The corpus is nothing like SiP

That is the point. SiP's estimator is unbiased in the median (log-ratio median 0.000);
Apache's contributors **over-estimate heavily and consistently** — the pooled log-ratio
mean is −1.05, i.e. issues take roughly a third of the time they were given. So this is
the regime `metis-api`'s local recalibration was actually built for, and the one SiP
could not exercise.

## Results — pooled out-of-fold (n = 1,047)

| arm | PRED(25) [CI95] | MdAPE [CI95] | ΔPRED(25) vs estimate [CI95] | McNemar p |
|---|---|---|---|---|
| estimate alone (the bar) | 14.0% [11.9%, 16.0%] | 300.0% [200.0%, 500.0%] | — | — |
| estimate x global factor (mean) | 6.9% [5.4%, 8.4%] | 116.6% [97.2%, 167.6%] | -7.2% [-9.8%, -4.5%] | 3.028e-07 (b=69, c=144) |
| estimate x global factor (median) | 14.4% [12.3%, 16.4%] | 275.0% [200.0%, 400.0%] | +0.4% [-0.9%, +1.6%] | 0.6587 (b=25, c=21) |
| estimate x conditional f(x) | 9.3% [7.5%, 11.0%] | 108.1% [96.4%, 149.5%] | -4.8% [-7.0%, -2.7%] | 6.125e-05 (b=51, c=101) |
| model alone, no anchor | 13.1% [11.2%, 15.2%] | 82.1% [78.9%, 86.5%] | -1.0% [-3.7%, +1.9%] | 0.5495 (b=108, c=118) |

Achieved power: 152 discordant pairs out of 1,047 test rows (p_disc = 0.145);
minimal detectable effect at 80% power is
pi1 = 0.613.

## Sensitivities

**Excluding exact ties** (6.3% of test rows record the
actual as exactly the estimate — the same logging convention SiP shows, and it flatters
every arm that stays near the anchor). n = 981:

| arm | PRED(25) [CI95] | MdAPE [CI95] | ΔPRED(25) vs estimate [CI95] | McNemar p |
|---|---|---|---|---|
| estimate alone (the bar) | 8.3% [6.5%, 10.0%] | 500.0% [300.0%, 500.0%] | — | — |
| estimate x global factor (mean) | 7.3% [5.7%, 9.0%] | 172.6% [116.6%, 233.9%] | -0.9% [-3.3%, +1.5%] | 0.5095 (b=69, c=78) |
| estimate x global factor (median) | 8.7% [6.9%, 10.5%] | 400.0% [299.9%, 500.0%] | +0.4% [-0.9%, +1.8%] | 0.6587 (b=25, c=21) |
| estimate x conditional f(x) | 7.4% [5.8%, 9.2%] | 150.7% [109.6%, 184.9%] | -0.8% [-2.9%, +1.1%] | 0.5047 (b=51, c=59) |
| model alone, no anchor | 13.0% [11.0%, 15.1%] | 82.3% [78.5%, 87.3%] | +4.8% [+2.2%, +7.3%] | 0.0003721 (b=108, c=61) |

**Issues estimated at 4 hours or more** (n = 635):

| arm | PRED(25) [CI95] | MdAPE [CI95] | ΔPRED(25) vs estimate [CI95] | McNemar p |
|---|---|---|---|---|
| estimate alone (the bar) | 7.4% [5.5%, 9.4%] | 2300.0% [1900.0%, 3100.0%] | — | — |
| estimate x global factor (mean) | 3.3% [2.0%, 4.7%] | 1059.6% [808.6%, 1319.0%] | -4.1% [-6.6%, -1.7%] | 0.001858 (b=20, c=46) |
| estimate x global factor (median) | 9.4% [7.2%, 11.8%] | 2300.0% [1700.0%, 2669.2%] | +2.0% [+0.8%, +3.5%] | 0.004425 (b=16, c=3) |
| estimate x conditional f(x) | 5.2% [3.5%, 6.9%] | 537.5% [458.6%, 703.2%] | -2.2% [-4.9%, +0.5%] | 0.1302 (b=30, c=44) |
| model alone, no anchor | 11.7% [9.3%, 14.3%] | 85.1% [81.2%, 88.9%] | +4.3% [+0.9%, +7.7%] | 0.01773 (b=74, c=47) |

## Reading: NO CORRECTION BEATS THE RECORDED ESTIMATE

Set against SiP, the pair says something neither says alone. On an organisation whose
estimator is already unbiased, a global recalibration factor *subtracts* accuracy
(SiP: −0.8%, p = 2.6e-05). On one that is heavily and consistently biased, the same
factor is the correction that matters. Neither corpus rewards conditioning on task
context beyond that.

That is precisely the behaviour `metis_api.recalibration` was rebuilt around: shrink the
local factor toward zero when the recorded ratios look like scatter, keep it whole when
they agree, and never assume which regime a client is in.

### Limits, stated plainly

- **Small.** 1,047 scored rows pooled, and the per-project counts are far too thin for
  per-project verdicts. Read the pooled paired test, not the strata.
- **Optional time tracking is not random.** The issues that carry both fields are the
  ones somebody chose to track, which is a self-selected slice of Apache's work.
- **An "original estimate" in an open-source tracker is often a formality** — the modal
  values are round (1 day, 1 week) — so this measures correcting a *weak* estimate, and
  should not be read as a claim about a professional estimating process.
