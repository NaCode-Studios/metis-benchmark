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
class TemporalSplit:
    """Index sets of a chronological train / calibration / test split."""

    # Row positions into the original DataFrame for each block.
    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray


def temporal_split(
    df: pd.DataFrame,
    date_column: str,
    calibration_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> TemporalSplit:
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
    return TemporalSplit(
        train=order[:n_train],
        calibration=order[n_train : n_train + n_cal],
        test=order[n_train + n_cal :],
    )
