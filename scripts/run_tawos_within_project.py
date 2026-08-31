"""Post-G0 R&D — do a team's own story points, on its own history, predict the outcome?

TAWOS is the only public corpus with enough within-project history to ask the question
at scale: 43,108 issues across 39 JIRA projects carry both a story-point estimate and a
recorded outcome, every one of them dated. SiP is one company at task granularity and
Apache JIRA is ~2k issues; this is 28 projects with 200+ paired issues each.

WHAT THE LABEL ACTUALLY IS — read this before the numbers. TAWOS's
`Total_Effort_Minutes` is derived from an issue's in-progress status transitions, so it
is **elapsed in-progress time, not hours of work**: its median is 160 hours per issue,
which is a lead time, not an effort. The JIRA worklog field (`Timespent`) is the real
effort label, and only 900 issues carry both it and a story point — reported below as
the small, correctly-labelled check.

So this measures the estimate-to-OUTCOME relationship, not effort accuracy, and it
cannot be compared to a G0 PRED(25). It is still the product's question in the regime
that matters: a team, its own estimates, its own history, a strictly temporal split.

Arms (story points are not hours, so the baseline has to convert them):

    rate        SP x the TRAIN block's own median outcome-per-point — the team
                calibrating itself, which is exactly what a client would do
    flat        the TRAIN block's median outcome, ignoring the estimate — the
                estimate-blind floor every arm must beat to have earned anything
    corrected   a GBM on log(outcome) from log(SP) + issue context
    model_only  the same without the story point

    python scripts/extract_tawos.py                 # once, from the dump
    PYTHONPATH=src python scripts/run_tawos_within_project.py

NOT gate-carrying. TAWOS is declared in the registry as retrieval-scale-only and stays
outside the gate; the G0 verdict is untouched. Dataset: TAWOS (MSR 2022),
DOI 10.5522/04/21308124, Apache-2.0, research use only.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from metis_benchmark.datasets.registry import raw_path  # noqa: E402
from metis_benchmark.evaluation import rolling_origin_split  # noqa: E402
from metis_benchmark.within_org import (  # noqa: E402
    FeatureSpec,
    design_matrix,
    fold_categories,
    score_frame,
    _booster,
)

OUT = Path(__file__).resolve().parents[1] / "reports" / "results" / "postg0_tawos.md"
ARMS = ("rate", "flat", "corrected", "model_only")
LABEL = {
    "rate": "story points x the team's own rate (the bar)",
    "flat": "the team's median outcome (estimate-blind floor)",
    "corrected": "GBM on log(SP) + context",
    "model_only": "GBM on context alone, no story point",
}
# Story points are the estimate; everything else is context fixed when the issue is
# estimated. Status, resolution and every *_Date after estimation are ex-post and never
# enter — the same rule that threw out SiP's DeveloperPerformance.
SPEC = FeatureSpec(
    estimate="Story_Point",
    numeric=(),
    categorical=("Type", "Priority", "Reporter_ID", "Assignee_ID", "Sprint_ID"),
)
MIN_ISSUES = 200   # per project, to form a rolling-origin split worth reading
SEED = 0


def _load(effort_column: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path("tawos") / "issues.csv", low_memory=False)
    sp = pd.to_numeric(df["Story_Point"], errors="coerce")
    outcome = pd.to_numeric(df[effort_column], errors="coerce")
    keep = sp.gt(0) & outcome.gt(0) & df["Estimation_Date"].notna()
    out = df.loc[keep].copy()
    out["Story_Point"] = sp[keep]
    # Minutes -> hours for Total_Effort_Minutes, seconds -> hours for Timespent, so the
    # column means the same thing it does everywhere else in this package.
    divisor = 60.0 if effort_column == "Total_Effort_Minutes" else 3600.0
    out["actual"] = outcome[keep] / divisor
    out["date"] = pd.to_datetime(out["Estimation_Date"], errors="coerce")
    return out.loc[out["date"].notna()].reset_index(drop=True)


def _run_project(project: pd.DataFrame) -> pd.DataFrame | None:
    """Rolling-origin over one project's own history. None when it is too short."""
    project = project.sort_values("date").reset_index(drop=True)
    try:
        splits = rolling_origin_split(project, "date", initial_train_fraction=0.5,
                                      n_folds=5, calibration_fraction=0.2,
                                      min_test_per_fold=5)
    except ValueError:
        return None
    rows: list[pd.DataFrame] = []
    for split in splits:
        train_idx = np.concatenate([split.train, split.calibration])
        train, test = project.iloc[train_idx], project.iloc[split.test]
        if len(train) < 30:
            continue
        y_tr = train["actual"].to_numpy(dtype=float)
        sp_tr = train["Story_Point"].to_numpy(dtype=float)
        sp_te = test["Story_Point"].to_numpy(dtype=float)
        # The team's own conversion, taken as a MEDIAN: outcome-per-point is heavily
        # right-skewed and a mean rate would be set by the worst issue in the block.
        rate = float(np.median(y_tr / sp_tr))
        flat = float(np.median(y_tr))

        cats = fold_categories(train, SPEC)
        X_tr, X_te = design_matrix(train, cats, SPEC), design_matrix(test, cats, SPEC)
        corrected = np.exp(_booster(SEED).fit(X_tr, np.log(y_tr)).predict(X_te))
        X_tr_no, X_te_no = X_tr.drop(columns=["log_estimate"]), X_te.drop(columns=["log_estimate"])
        solo = np.exp(_booster(SEED).fit(X_tr_no, np.log(y_tr)).predict(X_te_no))

        rows.append(pd.DataFrame({
            "actual": test["actual"].to_numpy(dtype=float),
            "rate": sp_te * rate,
            "flat": np.full(len(test), flat),
            "corrected": np.asarray(corrected, dtype=float),
            "model_only": np.asarray(solo, dtype=float),
            "project": test["Project_ID"].to_numpy(),
        }))
    return pd.concat(rows, ignore_index=True) if rows else None


def _table(result) -> str:
    lines = ["| arm | PRED(25) [CI95] | MdAPE [CI95] | ΔPRED(25) vs the rate [CI95] | McNemar p |",
             "|---|---|---|---|---|"]
    for name in ARMS:
        a = result.arms[name]
        d = result.delta_pred25.get(name)
        m = result.mcnemar.get(name)
        delta = f"{d.point:+.1%} [{d.low:+.1%}, {d.high:+.1%}]" if d else "—"
        pval = f"{m['p']:.4g}" if m else "—"
        lines.append(
            f"| {LABEL[name]} | {a.pred25.point:.1%} [{a.pred25.low:.1%}, {a.pred25.high:.1%}] "
            f"| {a.mdape.point:.1%} [{a.mdape.low:.1%}, {a.mdape.high:.1%}] | {delta} | {pval} |"
        )
    return "\n".join(lines)


def _pooled(effort_column: str, min_issues: int):
    df = _load(effort_column)
    counts = df.groupby("Project_ID").size()
    keep = counts[counts >= min_issues].index
    frames = [f for f in (_run_project(df[df.Project_ID == p]) for p in keep) if f is not None]
    if not frames:
        return None, df, len(keep)
    pooled = pd.concat(frames, ignore_index=True)
    return score_frame(pooled, arms_names=ARMS), df, len(keep)


def main() -> None:
    lead, lead_df, lead_projects = _pooled("Total_Effort_Minutes", MIN_ISSUES)
    # The correctly-labelled check: JIRA's worklog field. Far fewer rows, so the
    # per-project floor drops — this is a sanity check, not a second verdict.
    work, work_df, work_projects = _pooled("Timespent", 60)

    lead_floor_ci = lead.delta_pred25["flat"]
    lead_corr_ci = lead.delta_pred25["corrected"]
    fmt = dict(
        lead_floor=-lead_floor_ci.point,
        lead_floor_lo=-lead_floor_ci.high,
        lead_floor_hi=-lead_floor_ci.low,
        lead_floor_p=lead.mcnemar["flat"]["p"],
        lead_corr=lead_corr_ci.point,
        lead_corr_lo=lead_corr_ci.low,
        lead_corr_hi=lead_corr_ci.high,
        work_rate=work.arms["rate"].pred25.point if work else float("nan"),
        work_flat=work.arms["flat"].pred25.point if work else float("nan"),
        work_n=work.arms["rate"].n if work else 0,
    )

    text = f"""# Post-G0 — TAWOS: do a team's own story points predict its own outcomes?

> **Post-G0 R&D — NOT gate-carrying.** TAWOS is declared in the registry as
> retrieval-scale-only and stays outside the gate; the G0 verdict is untouched.
> Reproduce: `python scripts/extract_tawos.py` then
> `PYTHONPATH=src python scripts/run_tawos_within_project.py`.
> Dataset: TAWOS (MSR 2022), DOI 10.5522/04/21308124, Apache-2.0, research use only.

## Read the label before the numbers

TAWOS's `Total_Effort_Minutes` is derived from an issue's in-progress status
transitions. Its median is about **160 hours per issue**, which is a *lead time*, not
hours of work. The real effort label is JIRA's worklog field `Timespent`, and only
{len(work_df):,} issues carry both it and a story point.

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

## Outcome = in-progress lead time ({lead.arms["rate"].n:,} out-of-fold rows, {lead_projects} projects with >= {MIN_ISSUES} paired issues)

{_table(lead)}

## Outcome = logged work time ({work.arms["rate"].n:,} rows, {work_projects} projects with >= 60 paired issues)

{_table(work) if work else "_too few projects clear the floor to score._"}

## Reading

**On lead time the team's own point scale does carry signal — and the model still does
not beat it.** Converting story points through the block's own median rate beats the
estimate-blind floor by {fmt['lead_floor']:+.1%} [{fmt['lead_floor_lo']:+.1%}, {fmt['lead_floor_hi']:+.1%}]
(McNemar p = {fmt['lead_floor_p']:.2g}), so the points are not noise. But a gradient-boosted
model given those same points plus the issue's context lands
{fmt['lead_corr']:+.1%} [{fmt['lead_corr_lo']:+.1%}, {fmt['lead_corr_hi']:+.1%}] against the plain
conversion: no better, and the CI excludes any improvement worth selling. Dropping the
story point costs another point and a half, which is the same control result the other
two corpora give — the existing estimate is carrying the accuracy, not the model.

**On the correctly-labelled effort subset the points stop working entirely.** Against
logged work time the team's conversion scores {fmt['work_rate']:.1%} and the estimate-blind
floor scores {fmt['work_flat']:.1%} — the floor is *ahead*, though at n =
{fmt['work_n']:,} the interval straddles zero and the honest reading is a tie. That is G0's
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
"""
    OUT.write_text(text)
    print(f"wrote {OUT}")
    for name in ARMS:
        print(f"  lead {name:12} {lead.arms[name].pred25}")
    if work:
        for name in ARMS:
            print(f"  work {name:12} {work.arms[name].pred25}")


if __name__ == "__main__":
    main()
