"""Within-organisation estimate correction — the regime the public gate could not test.

WHY THIS MODULE EXISTS. G0 closed two claims with evidence: cross-organisation
tabular attributes cap at ~42-52% PRED(25), and requirement text does not predict
logged effort cross-project (ceiling 15-20%). Both verdicts stand. What neither
answers is the regime the product is actually deployed in: **one organisation, its
own history, its own estimator**. `reports/REPORT.md` §7 names this as the honest
gap and defers it to a proprietary pilot.

It does not have to be deferred. **SiP is a single organisation**: 12,299 estimate
events logged by one company over ten years (2004-2014), 22 developers, 20 project
codes, and a developer estimate recorded on 100% of rows. The gate used it as a
*text* dataset only. Used as a tabular within-organisation dataset it is the closest
public proxy that exists for the Stage-3 pilot question, and it is already downloaded.

THE QUESTION IS DELIBERATELY NOT G0's. G0 asked whether a model can estimate effort
from scratch; the answer is no, and nothing here revisits it. This asks the question
the product's surviving claims rest on:

    Is the organisation's own estimation error CONDITIONALLY learnable from
    leakage-safe task context — does correcting the expert beat the expert?

Four arms answer it on the same test rows, so every comparison is paired:

    expert      the recorded HoursEstimate, untouched — the bar to beat
    shift       expert x exp(MEAN train log-ratio) — a single global recalibration
                factor, i.e. exactly what metis-api's local recalibration computes
    shift_med   expert x exp(MEDIAN train log-ratio) — the same idea with a robust
                centre. The two diverge exactly when the log-ratios are heavy-tailed,
                which is the case a product cannot assume away: on Apache JIRA the
                mean log-ratio is -1.05 and the median -0.35, so the mean-based factor
                drags every well-estimated issue off target to chase a tail
    corrected   expert x exp(f(x)), f fitted on the train block only — the hypothesis
    model_only  a GBM on log(effort) from the same features WITHOUT the expert anchor,
                the control that shows how much of the result the anchor carries

`shift` is the one that makes the result actionable: if `corrected` beats `expert`
but `shift` does not, conditioning is what pays, and the product's unconditional
recalibration factor is leaving the value on the table. If `shift` already captures
it, the simpler thing is the right thing.

STATUS: post-G0 R&D, NOT gate-carrying. The G0 verdict (NO-GO on the public gate) is
untouched by anything in this module: it consumes recorded expert estimates, which
`protocol.md` §5 places permanently outside the gate, and SEERA is not read at all.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from metis_benchmark.evaluation import (
    CI,
    bootstrap_metric_ci,
    mcnemar_exact,
    mdape,
    paired_bootstrap_diff,
    pred_at,
    pred_hits,
    rolling_origin_split,
)

# --- the leakage split, declared before any model is fitted ------------------
# Same doctrine as the SEERA audit (protocol.md §8): a feature qualifies only if
# it is knowable at the moment the estimate is made. SiP's columns split cleanly,
# and the reason is recorded per column so the decision can be argued with.

SIP_ADMISSIBLE: dict[str, str] = {
    "HoursEstimate": "the estimate itself — the anchor being corrected",
    "Priority": "assigned when the task is raised, before it is estimated",
    "Category": "task class (Development / Management / Operational), set at raise time",
    "SubCategory": "finer task class (Bug, Enhancement, Progress Meeting, ...), set at raise time",
    "ProjectCode": "which project the task belongs to — known upfront",
    "ProjectBreakdownCode": "work-package within the project — known upfront",
    "RaisedByID": "who raised the task — known upfront",
    "AssignedToID": "who the task is assigned to — known at estimation time",
    "AuthorisedByID": "who authorised it — known upfront (missing on 65% of rows, encoded as such)",
}

SIP_INADMISSIBLE: dict[str, str] = {
    "HoursActual": "the target",
    "DeveloperID": "who ACTUALLY did the work — an assignment can change after the estimate",
    "DeveloperHoursActual": "derived from the actual",
    "TaskPerformance": "estimate minus actual, by construction the answer",
    "DeveloperPerformance": "derived from realised work, the SEERA 'Programmers capability' trap",
    "StatusCode": "the task's FINAL status — unknown while it is being estimated",
    "StartedOn": "post-estimation timestamp",
    "CompletedOn": "post-estimation timestamp",
    "Summary": "admissible in principle, but this module is the TABULAR channel; "
    "the text channel was measured at G0 and is not re-litigated here",
}

@dataclass(frozen=True)
class FeatureSpec:
    """Which admissible columns a corpus offers, split by how a tree should read them.

    Carried as data rather than hardcoded so a second corpus can reuse this machinery
    without either one silently inheriting the other's column names — the failure mode
    that would make two "identical" experiments incomparable.

    `estimate` is the recorded human estimate: it always enters in log space, the scale
    every effort quantity in this repository is modelled on. `numeric` are magnitudes or
    ordered codes; `categorical` are identities (ids, types, projects) that get native
    categorical splits and route unseen levels down the missing-value branch.
    """

    estimate: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]


# SiP: Priority is an ordered integer (1..10) and stays numeric. AuthorisedByID is an
# id, not a magnitude, so it is categorical even though the raw column is a float.
SIP_SPEC = FeatureSpec(
    estimate="HoursEstimate",
    numeric=("Priority",),
    categorical=(
        "Category",
        "SubCategory",
        "ProjectCode",
        "ProjectBreakdownCode",
        "RaisedByID",
        "AssignedToID",
        "AuthorisedByID",
    ),
)


# Apache JIRA (post-G0, registry in_gate=False). The admissible set is the JIRA
# analogue of SiP's: everything below is fixed when the issue is filed and estimated.
# `timespent`, `resolutiondate` and the final status are the ex-post half and never
# enter — the same rule that threw out SiP's DeveloperPerformance and SEERA's
# realised-work attributes.
APACHE_SPEC = FeatureSpec(
    estimate="expert_estimate",
    numeric=("n_components",),
    categorical=("project", "issuetype", "priority", "reporter", "assignee"),
)


def design_matrix(
    df: pd.DataFrame,
    categories: dict[str, pd.Index] | None = None,
    spec: FeatureSpec = SIP_SPEC,
) -> pd.DataFrame:
    """Admissible features only, with categorical levels pinned to the TRAIN fold.

    `categories` fixes each categorical column's level set. Passing the train
    fold's levels when transforming the test fold is what makes an unseen level
    at test time become NaN — which XGBoost routes down its missing-value branch
    — instead of silently shifting every code by one and corrupting the split
    boundaries the model learned. Called with None it derives the levels from
    `df` itself, which is correct only when `df` IS the train fold.
    """
    out = pd.DataFrame(index=df.index)
    # log1p, not log: SiP records tasks down to 0.01 h, and log1p keeps the small
    # end finite and monotone without a special case.
    out["log_estimate"] = np.log1p(df[spec.estimate].astype(float))
    for col in spec.numeric:
        out[col] = df[col].astype(float)
    for col in spec.categorical:
        raw = df[col].astype("string")
        levels = categories[col] if categories is not None else pd.Index(sorted(raw.dropna().unique()))
        # Blank out unknown levels EXPLICITLY before constructing the Categorical:
        # pandas deprecated the silent coercion, and being explicit is also the
        # honest spelling of the rule — a level the train fold never saw is a
        # missing value to this model, not a category it can reason about.
        out[col] = pd.Categorical(raw.where(raw.isin(levels)), categories=levels)
    return out


#: Reserved level for a categorical column the train block never observed. It exists
#: only so the level set is non-empty: a column that is entirely missing in a fold
#: (TAWOS has projects where no issue carries a sprint) would otherwise produce a
#: zero-category dtype, which the booster cannot encode. No row is ever assigned to it,
#: so the column stays uninformative rather than becoming subtly informative. Plain
#: ASCII on purpose: the booster encodes category names as an arrow string array and
#: rejects a NUL byte outright.
_ABSENT_LEVEL = "<absent>"


def fold_categories(train_df: pd.DataFrame, spec: FeatureSpec = SIP_SPEC) -> dict[str, pd.Index]:
    """The categorical level sets a fold's model is allowed to know about."""
    out: dict[str, pd.Index] = {}
    for col in spec.categorical:
        levels = sorted(train_df[col].astype("string").dropna().unique())
        out[col] = pd.Index(levels if levels else [_ABSENT_LEVEL])
    return out


def _booster(seed: int, quantiles: tuple[float, ...] | None = None):
    """The house gradient-boosting configuration, with native categorical splits.

    Same conservative shape as `track_a.gbm.LogGBMQuantile` (shallow trees, slow
    rate) — SiP folds are thousands of rows rather than dozens, but the point of
    this experiment is whether a *modest* correction generalises, not how far an
    over-parameterised one can be pushed. No hyperparameter is tuned against a
    test block anywhere in this module.
    """
    from xgboost import XGBRegressor

    kwargs = dict(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        random_state=seed,
        enable_categorical=True,
        tree_method="hist",
    )
    if quantiles is not None:
        return XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=np.asarray(quantiles, dtype=float),
            **kwargs,
        )
    return XGBRegressor(**kwargs)


@dataclass
class ArmScores:
    """One arm's out-of-fold scores on the pooled test rows."""

    name: str
    pred25: CI
    mdape: CI
    n: int


@dataclass
class WithinOrgResult:
    """Everything the report needs, computed once from the pooled out-of-fold rows."""

    arms: dict[str, ArmScores]
    # Paired tests of every arm against the expert baseline.
    delta_pred25: dict[str, CI] = field(default_factory=dict)
    mcnemar: dict[str, dict] = field(default_factory=dict)
    # Pooled out-of-fold predictions, kept so the caller can slice sensitivities
    # (exact-tie rows, size strata) without refitting anything.
    frame: pd.DataFrame | None = None


# Frozen protocol parameters — the same values the gate used for SiP (v1.2), spelled
# out here so a change to the benchmark's defaults cannot silently redefine this run.
INITIAL_TRAIN_FRACTION = 0.5
N_FOLDS = 5
CALIBRATION_FRACTION = 0.2
MIN_TEST_PER_FOLD = 5
N_BOOT = 2000
BOOT_ALPHA = 0.05
SEED = 0


def run_within_org(
    df: pd.DataFrame,
    *,
    date_column: str = "date",
    effort_column: str = "effort",
    expert_column: str = "expert_estimate",
    spec: FeatureSpec = SIP_SPEC,
    seed: int = SEED,
    n_boot: int = N_BOOT,
) -> WithinOrgResult:
    """Fit and score the four arms out-of-fold on a rolling-origin temporal split.

    Every fold trains strictly on its past. The correction model never sees a test
    row's actual, and the categorical level sets come from the train block only.
    Rows without a positive effort, a positive expert estimate or a date are dropped
    before splitting — log-space modelling needs all three strictly positive, and the
    drop count is reported rather than absorbed.
    """
    usable = (
        df[effort_column].astype(float).gt(0)
        & df[expert_column].astype(float).gt(0)
        & df[date_column].notna()
    )
    data = df.loc[usable].copy().reset_index(drop=True)

    splits = rolling_origin_split(
        data,
        date_column=date_column,
        initial_train_fraction=INITIAL_TRAIN_FRACTION,
        n_folds=N_FOLDS,
        calibration_fraction=CALIBRATION_FRACTION,
        min_test_per_fold=MIN_TEST_PER_FOLD,
    )

    rows: list[pd.DataFrame] = []
    for split in splits:
        # The calibration block is part of the training data for these arms: no
        # conformal step is fitted here, so withholding it would only starve the
        # model. It stays chronologically before the test block either way.
        train_idx = np.concatenate([split.train, split.calibration])
        train, test = data.iloc[train_idx], data.iloc[split.test]

        expert_tr = train[expert_column].to_numpy(dtype=float)
        expert_te = test[expert_column].to_numpy(dtype=float)
        actual_tr = train[effort_column].to_numpy(dtype=float)

        # The correction target: how far the organisation's own estimate lands from
        # the truth, in log space, where a constant factor is a constant offset.
        ratio_tr = np.log(actual_tr) - np.log(expert_tr)

        cats = fold_categories(train, spec)
        X_tr = design_matrix(train, cats, spec)
        X_te = design_matrix(test, cats, spec)

        # Arm 2 — one global factor, the unconditional recalibration.
        shift = float(np.mean(ratio_tr))
        # Arm 2b — the same, centred robustly. Under log-normal residuals the two
        # coincide; under a heavy tail the mean chases the tail and the median does
        # not, and effort ratios in the wild are heavy-tailed.
        shift_med = float(np.median(ratio_tr))

        # Arm 3 — the conditional correction.
        corr = _booster(seed).fit(X_tr, ratio_tr)
        f_te = np.asarray(corr.predict(X_te), dtype=float)

        # Arm 4 — the control: same features, no expert anchor, predicting log-effort.
        X_tr_noexp = X_tr.drop(columns=["log_estimate"])
        X_te_noexp = X_te.drop(columns=["log_estimate"])
        solo = _booster(seed).fit(X_tr_noexp, np.log(actual_tr))
        solo_te = np.exp(np.asarray(solo.predict(X_te_noexp), dtype=float))

        rows.append(
            pd.DataFrame(
                {
                    "actual": test[effort_column].to_numpy(dtype=float),
                    "expert": expert_te,
                    "shift": expert_te * np.exp(shift),
                    "shift_med": expert_te * np.exp(shift_med),
                    "corrected": expert_te * np.exp(f_te),
                    "model_only": solo_te,
                    "exact_tie": np.isclose(
                        test[effort_column].to_numpy(dtype=float), expert_te
                    ),
                    "date": test[date_column].to_numpy(),
                }
            )
        )

    pooled = pd.concat(rows, ignore_index=True)
    return score_frame(pooled, n_boot=n_boot, seed=seed)


#: Arm columns produced by `run_within_org`, in report order. The first is the
#: baseline every other arm is tested against.
DEFAULT_ARMS = ("expert", "shift", "shift_med", "corrected", "model_only")


def score_frame(
    pooled: pd.DataFrame,
    *,
    arms_names: tuple[str, ...] = DEFAULT_ARMS,
    n_boot: int = N_BOOT,
    seed: int = SEED,
) -> WithinOrgResult:
    """Score an already-predicted frame — used for the run itself and for slices of it.

    `arms_names[0]` is the baseline: every other arm gets a paired bootstrap of its
    PRED(25) difference against it and an exact McNemar on the paired hits. A corpus
    whose arms differ (TAWOS converts story points through the team's own rate rather
    than correcting an hours estimate) passes its own names rather than pretending to
    have this one's.
    """
    y = pooled["actual"].to_numpy(dtype=float)
    baseline_name = arms_names[0]
    arms: dict[str, ArmScores] = {}
    deltas: dict[str, CI] = {}
    tests: dict[str, dict] = {}
    for name in arms_names:
        p = pooled[name].to_numpy(dtype=float)
        arms[name] = ArmScores(
            name=name,
            pred25=bootstrap_metric_ci(y, p, pred_at, n_boot=n_boot, alpha=BOOT_ALPHA, seed=seed),
            mdape=bootstrap_metric_ci(y, p, mdape, n_boot=n_boot, alpha=BOOT_ALPHA, seed=seed),
            n=len(y),
        )
        if name != baseline_name:
            base = pooled[baseline_name].to_numpy(dtype=float)
            deltas[name] = paired_bootstrap_diff(
                y, p, base, pred_at, n_boot=n_boot, alpha=BOOT_ALPHA, seed=seed,
            )
            tests[name] = mcnemar_exact(pred_hits(y, p), pred_hits(y, base))
    return WithinOrgResult(arms=arms, delta_pred25=deltas, mcnemar=tests, frame=pooled)
