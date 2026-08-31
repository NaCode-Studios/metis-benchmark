"""Post-G0 R&D — the SiP question asked of many teams instead of one company.

`postg0_sip_within_org.md` found that correcting an organisation's own estimate from
its own task context does not beat leaving it alone. The obvious objection is that SiP
is a single company with a single logging culture. This runs the identical four-arm
experiment on ~2,100 Apache JIRA issues across ~180 independent projects, each with a
recorded original estimate and recorded time spent.

    python scripts/mine_apache_jira.py            # fetch the corpus (once)
    PYTHONPATH=src python scripts/run_apache_jira_within_org.py

NOT gate-carrying: mined after G0 closed, consumes recorded expert estimates, and never
touches SEERA. The G0 verdict is untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metis_benchmark.datasets.loaders import load  # noqa: E402
from metis_benchmark.evaluation import mcnemar_mde  # noqa: E402
from metis_benchmark.within_org import APACHE_SPEC, run_within_org, score_frame  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "reports" / "results" / "postg0_apache_jira.md"
ARMS = ("expert", "shift", "shift_med", "corrected", "model_only")
LABEL = {
    "expert": "estimate alone (the bar)",
    "shift": "estimate x global factor (mean)",
    "shift_med": "estimate x global factor (median)",
    "corrected": "estimate x conditional f(x)",
    "model_only": "model alone, no anchor",
}


def _arm_table(result) -> str:
    lines = [
        "| arm | PRED(25) [CI95] | MdAPE [CI95] | ΔPRED(25) vs estimate [CI95] | McNemar p |",
        "|---|---|---|---|---|",
    ]
    for name in ARMS:
        a = result.arms[name]
        d = result.delta_pred25.get(name)
        m = result.mcnemar.get(name)
        delta = f"{d.point:+.1%} [{d.low:+.1%}, {d.high:+.1%}]" if d else "—"
        pval = f"{m['p']:.4g} (b={m['b']}, c={m['c']})" if m else "—"
        lines.append(
            f"| {LABEL[name]} | {a.pred25.point:.1%} [{a.pred25.low:.1%}, {a.pred25.high:.1%}] "
            f"| {a.mdape.point:.1%} [{a.mdape.low:.1%}, {a.mdape.high:.1%}] | {delta} | {pval} |"
        )
    return "\n".join(lines)


def main() -> None:
    df = load("apache_jira")
    result = run_within_org(df, spec=APACHE_SPEC)
    pooled = result.frame
    assert pooled is not None

    no_ties = score_frame(pooled.loc[~pooled["exact_tie"]].reset_index(drop=True))
    big_rows = pooled.loc[pooled["expert"] >= 4.0].reset_index(drop=True)
    big = score_frame(big_rows) if len(big_rows) > 50 else None

    n = result.arms["expert"].n
    disc = result.mcnemar["corrected"]["discordant"]
    p_disc = disc / n if n else 0.0
    mde = mcnemar_mde(n, p_disc) if p_disc > 0 else None

    d_shift = result.delta_pred25["shift"]
    d_corr = result.delta_pred25["corrected"]
    shift_wins = d_shift.low > 0
    corr_wins = d_corr.low > 0
    if shift_wins and not corr_wins:
        head = "THE GLOBAL FACTOR PAYS, CONDITIONING DOES NOT"
    elif corr_wins and shift_wins:
        head = "BOTH CORRECTIONS PAY"
    elif corr_wins:
        head = "CONDITIONING PAYS"
    else:
        head = "NO CORRECTION BEATS THE RECORDED ESTIMATE"

    text = f"""# Post-G0 — Apache JIRA: the SiP question, asked of many teams

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
only **{len(df):,} issues across {df['project'].nunique()} projects** carry both an
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

## Results — pooled out-of-fold (n = {n:,})

{_arm_table(result)}

Achieved power: {disc:,} discordant pairs out of {n:,} test rows (p_disc = {p_disc:.3f});
minimal detectable effect at 80% power is
pi1 = {f"{mde:.3f}" if mde else "not reachable"}.

## Sensitivities

**Excluding exact ties** ({100 * pooled["exact_tie"].mean():.1f}% of test rows record the
actual as exactly the estimate — the same logging convention SiP shows, and it flatters
every arm that stays near the anchor). n = {no_ties.arms["expert"].n:,}:

{_arm_table(no_ties)}

**Issues estimated at 4 hours or more** (n = {big.arms["expert"].n:,}):

{_arm_table(big) if big else "_too few rows in this stratum to score._"}

## Reading: {head}

Set against SiP, the pair says something neither says alone. On an organisation whose
estimator is already unbiased, a global recalibration factor *subtracts* accuracy
(SiP: −0.8%, p = 2.6e-05). On one that is heavily and consistently biased, the same
factor is the correction that matters. Neither corpus rewards conditioning on task
context beyond that.

That is precisely the behaviour `metis_api.recalibration` was rebuilt around: shrink the
local factor toward zero when the recorded ratios look like scatter, keep it whole when
they agree, and never assume which regime a client is in.

### Limits, stated plainly

- **Small.** {n:,} scored rows pooled, and the per-project counts are far too thin for
  per-project verdicts. Read the pooled paired test, not the strata.
- **Optional time tracking is not random.** The issues that carry both fields are the
  ones somebody chose to track, which is a self-selected slice of Apache's work.
- **An "original estimate" in an open-source tracker is often a formality** — the modal
  values are round (1 day, 1 week) — so this measures correcting a *weak* estimate, and
  should not be read as a claim about a professional estimating process.
"""
    OUT.write_text(text)
    print(f"wrote {OUT}")
    for name in ARMS:
        print(f"  {name:12} PRED25 {result.arms[name].pred25}")


if __name__ == "__main__":
    main()
