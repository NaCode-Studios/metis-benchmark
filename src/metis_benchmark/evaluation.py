"""Metrics and validation-protocol primitives.

All accuracy metrics follow the effort-estimation literature so results are
directly comparable with published papers: PRED(25), MdAPE and empirical
coverage of prediction intervals.

The temporal split is the only split allowed by the protocol
(reports/protocol.md): random splits leak future information into training
and inflate every metric. Where a dataset has no usable date field, the
documented fallback must be declared in the report.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


def ape(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """Absolute percentage error per sample. Requires strictly positive actuals."""
    # Normalize any array-like input (list, Series, ndarray) to float arrays.
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    # Reject mismatched shapes early: silent broadcasting would pair wrong samples.
    if y_true.shape != y_pred.shape:
        raise ValueError(f"shape mismatch: {y_true.shape} vs {y_pred.shape}")
    # APE divides by the actual value; non-positive actuals make it undefined
    # and indicate a data problem upstream.
    if np.any(y_true <= 0):
        raise ValueError("APE is undefined for non-positive actual effort")
    # Relative error per sample: |actual - predicted| / actual.
    return np.abs(y_true - y_pred) / y_true


def pred_at(y_true: np.ndarray, y_pred: np.ndarray, level: float = 0.25) -> float:
    """PRED(l): fraction of estimates within `level` of the actual value.

    Gate G0 threshold: PRED(25) >= 0.55 on at least 2 datasets per track.
    """
    # Share of samples whose relative error is within the tolerance.
    return float(np.mean(ape(y_true, y_pred) <= level))


def mdape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Median absolute percentage error.

    Robust to outliers and, unlike MAPE, does not systematically reward
    underestimation. Gate G0 threshold: MdAPE <= 0.22.
    """
    # Median rather than mean: a single blown-up project must not dominate the score.
    return float(np.median(ape(y_true, y_pred)))


def empirical_coverage(y_true: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    """Fraction of actuals falling inside [lower, upper].

    Gate G0 threshold: within 5 points of the nominal level (0.90).
    """
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    # An inverted interval is a construction bug in the caller, not a data case.
    if np.any(lower > upper):
        raise ValueError("interval with lower > upper")
    # Observed coverage: share of actuals captured by their interval. Must
    # match the declared nominal level (e.g. 0.90) on unseen data.
    return float(np.mean((y_true >= lower) & (y_true <= upper)))


@dataclass(frozen=True)
class Split:
    """Row positions of a train / calibration / test split of one DataFrame.

    The calibration block exists for conformal interval calibration: it is
    always disjoint from both train and test.
    """

    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray


# Backwards-compatible alias: temporal_split historically returned a
# "TemporalSplit"; the structure is identical for every split method.
TemporalSplit = Split


def temporal_split(
    df: pd.DataFrame,
    date_column: str,
    calibration_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> Split:
    """Chronological split: oldest records train, newest records test.

    The calibration block sits between train and test so that conformal
    calibration never sees the future relative to training, and the test
    set is strictly the most recent slice.
    """
    # Guard the block sizes: the two tail fractions must leave room for training.
    if calibration_fraction + test_fraction >= 1.0:
        raise ValueError("calibration + test fractions must leave room for training")
    # Unordered (missing) dates cannot be placed on the timeline; force the
    # caller to drop or fix them explicitly.
    if df[date_column].isna().any():
        raise ValueError(f"missing values in date column '{date_column}'")

    # Row positions sorted by date ascending; stable sort keeps ties
    # deterministic across runs.
    order = np.argsort(df[date_column].to_numpy(), kind="stable")
    n = len(order)
    # Compute block sizes; each tail block gets at least one row on tiny datasets.
    n_test = max(1, int(round(n * test_fraction)))
    n_cal = max(1, int(round(n * calibration_fraction)))
    n_train = n - n_test - n_cal
    if n_train < 1:
        raise ValueError(f"dataset too small for a temporal split: {n} rows")

    # Carve the ordered timeline into [train][calibration][test].
    return Split(
        train=order[:n_train],
        calibration=order[n_train : n_train + n_cal],
        test=order[n_train + n_cal :],
    )


def group_split(
    groups: pd.Series,
    calibration_fraction: float = 0.2,
    test_fraction: float = 0.2,
    seed: int = 0,
) -> Split:
    """Split so that no group (project) appears in more than one block.

    Used for the dateless Track B datasets (deepse, josse): the engine trains
    on some projects and is scored on entirely held-out projects. This is the
    cold-start scenario — a new client/project with no in-project history —
    which the protocol selects as the right fallback when a date is missing.

    Allocation is over groups, not rows: blocks therefore hold whole projects.
    Group order is shuffled with a fixed seed so the partition is reproducible
    but not tied to the (arbitrary) listing order of projects.
    """
    if calibration_fraction + test_fraction >= 1.0:
        raise ValueError("calibration + test fractions must leave room for training")

    # Unique groups in stable first-appearance order, then a deterministic shuffle.
    unique_groups = pd.unique(groups)
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_groups)

    # At least one group must land in each block, so we need >= 3 groups.
    n_groups = len(shuffled)
    if n_groups < 3:
        raise ValueError(f"group_split needs at least 3 groups, got {n_groups}")
    n_test = max(1, int(round(n_groups * test_fraction)))
    n_cal = max(1, int(round(n_groups * calibration_fraction)))
    n_train = n_groups - n_test - n_cal
    if n_train < 1:
        raise ValueError(f"too few groups for the requested fractions: {n_groups}")

    # Assign whole groups to each block.
    test_groups = set(shuffled[:n_test])
    cal_groups = set(shuffled[n_test : n_test + n_cal])
    # Anything not in test/calibration is training.

    # Map each row to its block via its group membership.
    values = groups.to_numpy()
    test_mask = np.isin(values, list(test_groups))
    cal_mask = np.isin(values, list(cal_groups))
    train_mask = ~test_mask & ~cal_mask

    # Return row positions (0..n-1) for each block.
    idx = np.arange(len(groups))
    return Split(train=idx[train_mask], calibration=idx[cal_mask], test=idx[test_mask])


def ordered_kfold(
    n_samples: int,
    n_splits: int = 5,
    calibration_fraction: float = 0.2,
    seed: int = 0,
) -> list[Split]:
    """Deterministic k-fold cross-validation (fixed seed), one Split per fold.

    Last-resort fallback for dateless Track A datasets with no project
    grouping (cocomo81, china, albrecht). "Ordered" means reproducible
    (seeded), not temporal: there is no time guarantee here, which the report
    must state as a caveat.

    Each row serves in the test block of exactly one fold. Within a fold, a
    calibration slice is carved out of the training rows so conformal
    intervals can still be calibrated on data unseen by the fitted model.
    """
    if n_splits < 2:
        raise ValueError("k-fold needs at least 2 splits")
    if n_samples < n_splits:
        raise ValueError(f"{n_samples} samples cannot fill {n_splits} folds")

    # Shuffle row positions once with the fixed seed, then cut into k folds.
    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(n_samples)
    fold_indices = np.array_split(shuffled, n_splits)

    splits: list[Split] = []
    for i, test_idx in enumerate(fold_indices):
        # Training pool is every fold except the current test fold.
        rest = np.concatenate([fold_indices[j] for j in range(n_splits) if j != i])
        # Reserve the tail of the training pool as the calibration block.
        n_cal = max(1, int(round(len(rest) * calibration_fraction)))
        cal_idx = rest[:n_cal]
        train_idx = rest[n_cal:]
        splits.append(Split(train=train_idx, calibration=cal_idx, test=test_idx))
    return splits
