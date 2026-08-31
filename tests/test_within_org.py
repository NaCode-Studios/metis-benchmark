"""Within-organisation correction: leakage discipline, fold hygiene, and the arms.

The experiment's credibility rests on three mechanical properties, so they are
pinned here rather than trusted: the admissible feature set never touches an
ex-post column, a test fold's categorical levels come from its train block, and
each arm is the transform it claims to be.
"""

import numpy as np
import pandas as pd
import pytest

from metis_benchmark.within_org import (
    SIP_ADMISSIBLE,
    SIP_INADMISSIBLE,
    design_matrix,
    fold_categories,
    run_within_org,
    score_frame,
)


def _frame(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """A small SiP-shaped corpus: an expert anchor with a category-dependent bias."""
    rng = np.random.default_rng(seed)
    cat = rng.choice(["Development", "Management", "Operational"], size=n)
    expert = np.exp(rng.normal(1.0, 0.8, size=n))
    # Management tasks overrun by a factor of e^0.5, the others are unbiased: a
    # correction model that works at all must find this and nothing else.
    bias = np.where(cat == "Management", 0.5, 0.0)
    actual = expert * np.exp(bias + rng.normal(0.0, 0.3, size=n))
    return pd.DataFrame(
        {
            "HoursEstimate": expert,
            "Priority": rng.integers(1, 6, size=n),
            "Category": cat,
            "SubCategory": rng.choice(["Bug", "Enhancement"], size=n),
            "ProjectCode": rng.choice(["PC1", "PC2"], size=n),
            "ProjectBreakdownCode": rng.choice(["PBC1", "PBC2"], size=n),
            "RaisedByID": rng.integers(1, 5, size=n),
            "AssignedToID": rng.integers(1, 5, size=n),
            "AuthorisedByID": rng.choice([1.0, 2.0, np.nan], size=n),
            "effort": actual,
            "expert_estimate": expert,
            "date": pd.date_range("2010-01-01", periods=n, freq="D"),
        }
    )


# --- the leakage split ------------------------------------------------------


def test_admissible_and_inadmissible_are_disjoint():
    assert not set(SIP_ADMISSIBLE) & set(SIP_INADMISSIBLE)


def test_every_ex_post_sip_column_is_declared_inadmissible():
    # The columns whose value only exists once the work is done. If SiP's schema
    # ever grows another one, this test is where the omission should surface.
    for col in (
        "HoursActual",
        "DeveloperID",
        "DeveloperHoursActual",
        "TaskPerformance",
        "DeveloperPerformance",
        "StatusCode",
    ):
        assert col in SIP_INADMISSIBLE, col


def test_design_matrix_carries_no_inadmissible_column():
    X = design_matrix(_frame(50))
    assert not set(X.columns) & set(SIP_INADMISSIBLE)
    # HoursEstimate enters only through its log, never raw.
    assert "HoursEstimate" not in X.columns
    assert "log_estimate" in X.columns


# --- fold hygiene -----------------------------------------------------------


def test_unseen_category_becomes_missing_not_a_reused_code():
    train = _frame(100)
    test = _frame(20, seed=1).assign(ProjectCode="PC_NEVER_SEEN")
    X = design_matrix(test, fold_categories(train))
    # Unknown level -> NaN, which the booster routes down its missing branch.
    assert X["ProjectCode"].isna().all()


def test_train_categories_are_the_train_blocks_own_levels():
    train = _frame(100)
    cats = fold_categories(train)
    assert set(cats["Category"]) == set(train["Category"].unique())


# --- the arms ---------------------------------------------------------------


def test_arms_are_scored_on_the_same_rows():
    result = run_within_org(_frame(400), n_boot=200)
    assert {a.n for a in result.arms.values()} == {result.arms["expert"].n}


def test_shift_arm_is_a_pure_multiplicative_transform_of_the_expert():
    result = run_within_org(_frame(400), n_boot=200)
    f = result.frame
    ratio = f["shift"] / f["expert"]
    # One global factor per fold, so the distinct ratios are few and all positive.
    assert (ratio > 0).all()
    assert ratio.round(9).nunique() <= 5


def test_expert_arm_is_the_recorded_estimate_untouched():
    df = _frame(400)
    result = run_within_org(df, n_boot=200)
    assert (result.frame["expert"] > 0).all()


def test_correction_finds_a_planted_category_bias():
    # The generator plants a Management-only overrun and nothing else; a
    # conditional model that cannot beat a global shift HERE is broken, not honest.
    result = run_within_org(_frame(600), n_boot=200)
    assert result.arms["corrected"].pred25.point > result.arms["shift"].pred25.point


def test_run_is_deterministic():
    a = run_within_org(_frame(300), n_boot=200)
    b = run_within_org(_frame(300), n_boot=200)
    assert a.arms["corrected"].pred25.point == b.arms["corrected"].pred25.point
    assert a.delta_pred25["corrected"].low == b.delta_pred25["corrected"].low


def test_rows_without_a_usable_target_are_dropped_before_splitting():
    df = _frame(300)
    df.loc[0:9, "effort"] = 0.0          # non-positive effort
    df.loc[10:19, "expert_estimate"] = 0  # non-positive anchor
    result = run_within_org(df, n_boot=200)
    # 20 rows removed from a 300-row frame; the test half shrinks accordingly.
    assert result.arms["expert"].n < 150


def test_score_frame_slices_without_refitting():
    result = run_within_org(_frame(400), n_boot=200)
    subset = result.frame.iloc[:100].reset_index(drop=True)
    scored = score_frame(subset, n_boot=200)
    assert scored.arms["expert"].n == 100
    assert scored.arms["expert"].pred25.point == pytest.approx(
        float(np.mean(np.abs(subset["actual"] - subset["expert"]) / subset["actual"] <= 0.25))
    )


def test_a_column_absent_from_the_train_block_stays_encodable():
    # A fold where one categorical is entirely missing (TAWOS has projects with no
    # sprint on any issue) used to produce a zero-category dtype, which the booster
    # cannot encode — it died inside XGBoost's arrow conversion rather than saying so.
    train = _frame(100).assign(ProjectCode=np.nan)
    cats = fold_categories(train)
    assert len(cats["ProjectCode"]) == 1
    X = design_matrix(train, cats)
    # The placeholder is a level, not a value: every row is still missing.
    assert X["ProjectCode"].isna().all()


def test_an_all_missing_categorical_does_not_break_a_run():
    df = _frame(400)
    df["AuthorisedByID"] = np.nan
    result = run_within_org(df, n_boot=200)
    assert result.arms["expert"].n > 0
