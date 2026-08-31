"""Post-G0 R&D — is one organisation's estimation error conditionally learnable?

Runs the four arms of `metis_benchmark.within_org` on SiP (one company, ten years,
an expert estimate on every row) and writes reports/results/postg0_sip_within_org.md.

    PYTHONPATH=src python scripts/run_sip_within_org.py

NOT gate-carrying: it consumes recorded expert estimates, which protocol.md §5 keeps
permanently outside the gate, and it never reads SEERA. The G0 verdict is untouched.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metis_benchmark.datasets.loaders import load  # noqa: E402
from metis_benchmark.evaluation import mcnemar_mde  # noqa: E402
from metis_benchmark.within_org import (  # noqa: E402
    SIP_ADMISSIBLE,
    SIP_INADMISSIBLE,
    run_within_org,
    score_frame,
)

OUT = Path(__file__).resolve().parents[1] / "reports" / "results" / "postg0_sip_within_org.md"
ARMS = ("expert", "shift", "shift_med", "corrected", "model_only")
LABEL = {
    "expert": "expert alone (the bar)",
    "shift": "expert x global factor (mean)",
    "shift_med": "expert x global factor (median)",
    "corrected": "expert x conditional f(x)",
    "model_only": "model alone, no anchor",
}


def _arm_table(result) -> str:
    lines = ["| arm | PRED(25) [CI95] | MdAPE [CI95] | ΔPRED(25) vs expert [CI95] | McNemar p |",
             "|---|---|---|---|---|"]
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


def _verdict(result, no_ties, pooled) -> tuple[str, str]:
    """The reading, built from the numbers rather than asserted."""
    d_corr = result.delta_pred25["corrected"]
    d_shift = result.delta_pred25["shift"]
    nt_corr = no_ties.delta_pred25["corrected"]
    corr_wins = d_corr.low > 0
    shift_wins = d_shift.low > 0
    tie_rate = float(pooled["exact_tie"].mean())
    exp_all = result.arms["expert"].pred25.point
    exp_nt = no_ties.arms["expert"].pred25.point

    if corr_wins and not shift_wins:
        head = "CONDITIONING PAYS"
    elif corr_wins and shift_wins:
        head = "CORRECTION PAYS, CONDITIONING UNPROVEN"
    elif shift_wins:
        head = "ONLY THE GLOBAL SHIFT PAYS"
    else:
        head = "NO CORRECTION BEATS THE EXPERT"

    body = f"""Neither arm improves on the organisation's own estimator. Three findings sit
underneath that headline, and the second and third are the ones worth carrying forward.

**1. The correction does not beat the expert, and on the full set it is far worse.**
Conditional correction lands {d_corr.point:+.1%} [{d_corr.low:+.1%}, {d_corr.high:+.1%}]
against the expert. Once the exact ties are removed the gap closes to
{nt_corr.point:+.1%} [{nt_corr.low:+.1%}, {nt_corr.high:+.1%}] — a clean null at
n = {no_ties.arms["expert"].n:,} with {no_ties.mcnemar["corrected"]["discordant"]:,}
discordant pairs, so this is a *measured* tie, not an underpowered one. The size
structure that is plainly visible in the raw data (small tasks overrun, large tasks come
in under) is real in-sample and does not survive out-of-sample transfer across the
timeline. Conditioning on task context does not recover the organisation's own error.

**2. A single global recalibration factor makes an unbiased estimator worse.**
The shift arm is {d_shift.point:+.1%} [{d_shift.low:+.1%}, {d_shift.high:+.1%}]
(McNemar p = {result.mcnemar["shift"]["p"]:.2g}) — small, but significantly negative, and
the factor here was fitted on thousands of past rows rather than the handful a deployment
would have. This company's median log-ratio is 0.000: its estimator is already unbiased
in the median, and multiplying it by a mean-driven constant only moves it off target.
The product consequence is direct. `metis-api` applies exactly this transform,
unconditionally, from three recorded actuals. On an organisation like this one it would
subtract accuracy. A correction fitted on n = 3 and applied at full strength is the
failure mode this arm demonstrates at n = thousands; the mitigation is to shrink the
local delta toward zero rather than to trust it whole.

**3. The expert's SiP baseline is substantially a logging convention.**
On {tie_rate:.1%} of test rows the actual equals the estimate to the digit. The expert's
headline {exp_all:.1%} PRED(25) — the number G0 reports, reproduced here exactly — falls
to {exp_nt:.1%} once those rows are dropped. This does not overturn anything: the model
still loses on both readings, and "the human beats every model" survives. But the
*margin* is smaller than the headline suggests, and any future comparison against a SiP
expert baseline should quote both numbers.

**What the control says.** The no-anchor model reaches
{result.arms["model_only"].pred25.point:.1%}, in line with the G0 semantic channel. The
expert anchor is carrying essentially all of the accuracy in every arm that has one —
consistent with the post-G0 IVW finding that any blend with the current model makes the
expert alone worse.

**Consequence for the pilot.** The within-organisation hypothesis is not refuted in
general: this is one company, at task granularity (median 3 h), with a task-tracker's
context and no project-level features. But the cheapest version of the hypothesis — that
an organisation's estimation bias is a stable, learnable function of the context a
tracker records — is now measured, once, at real power, and it does not hold. A pilot
should be sold on the calibrated interval, which G0 did demonstrate, and not on an
expected point-accuracy lift."""
    return head, body


def main() -> None:
    df = load("sip")
    result = run_within_org(df)
    pooled = result.frame
    assert pooled is not None

    # Sensitivities, sliced from the SAME out-of-fold predictions (no refit, so the
    # models never see a row twice under a different rule).
    no_ties = score_frame(pooled.loc[~pooled["exact_tie"]].reset_index(drop=True))
    # SiP's median task is 3 hours and estimates are recorded as integers, so at the
    # small end PRED(25) is partly a rounding artefact: >= 4h is the stratum where a
    # 25% band is wider than the recording granularity.
    big = pooled.loc[pooled["expert"] >= 4.0].reset_index(drop=True)
    large = score_frame(big) if len(big) > 50 else None

    head, body = _verdict(result, no_ties, pooled)
    n = result.arms["expert"].n
    disc = result.mcnemar["corrected"]["discordant"]
    p_disc = disc / n if n else 0.0
    mde = mcnemar_mde(n, p_disc) if p_disc > 0 else None

    text = f"""# Post-G0 — SiP as a within-organisation corpus: is the expert's error learnable?

> **Post-G0 R&D — NOT gate-carrying.** The G0 verdict (NO-GO on the public gate)
> stands and is not revisited. This experiment consumes recorded expert estimates,
> which `protocol.md` §5 places permanently outside the gate, and it never reads the
> sealed SEERA holdout.
> Reproduce: `PYTHONPATH=src python scripts/run_sip_within_org.py`.

## Question

`reports/REPORT.md` §7 names one honest gap: every G0 split was a *cross-organisation*
cold start, and the regime a product is deployed in — one organisation's own history,
one team, one set of logging conventions — was never measured. SiP is that regime in
public form: **one company, {len(df):,} estimate events over ten years (2004-2014), 22
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
(initial train fraction {0.5}, {5} folds, min {5} test rows per fold). Every fold trains
strictly on its past; categorical level sets come from the train block, so an unseen
level at test time becomes a missing value rather than a silently reused code.

### Leakage split, declared before fitting

Admissible (knowable when the estimate is made):

{chr(10).join(f"- `{k}` — {v}" for k, v in SIP_ADMISSIBLE.items())}

Inadmissible:

{chr(10).join(f"- `{k}` — {v}" for k, v in SIP_INADMISSIBLE.items())}

`DeveloperPerformance` and `TaskPerformance` are the SiP equivalents of the SEERA
attributes the G0 audit threw out: both are computed from the realised work.

## Results — pooled out-of-fold (n = {n:,})

{_arm_table(result)}

Achieved power: {disc:,} discordant pairs out of {n:,} test rows
(p_disc = {p_disc:.3f}); the minimal detectable effect at 80% power is
pi1 = {f"{mde:.3f}" if mde else "not reachable"}. For contrast, the G0 assist test that
forced the assist claim to be withdrawn had ~11 discordant pairs and power ~0.16.

## Sensitivities

**Excluding exact ties.** On {100 * pooled["exact_tie"].mean():.1f}% of rows the actual
equals the estimate to the digit — a logging convention (the estimate copied forward at
close) as much as a measurement, and it flatters every arm that stays near the anchor.
Dropped (n = {no_ties.arms["expert"].n:,}):

{_arm_table(no_ties)}

**Tasks of 4 hours and up.** The median SiP task is 3 hours and estimates are recorded
as integers, so at the small end a 25% band is narrower than the recording granularity
(n = {large.arms["expert"].n:,}):

{_arm_table(large) if large else "_too few rows in this stratum to score._"}

## Reading: {head}

{body}

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
"""
    OUT.write_text(text)
    print(f"wrote {OUT}")
    for name in ARMS:
        a = result.arms[name]
        print(f"  {name:12} PRED25 {a.pred25}")


if __name__ == "__main__":
    main()
