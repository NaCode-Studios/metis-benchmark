"""The three mandatory baselines of the G0 protocol.

The engine must beat all three on every dataset where they are computable:

1. median effort by category,
2. linear regression of log(effort) on log(size),
3. the human expert estimate, where the dataset records one
   (kitchenham, seera, josse, sip).

Baselines are deliberately simple: they are the bar, not the contender.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class MedianByCategory:
    """Predicts the training median effort of each category.

    Unseen categories fall back to the global training median.
    """

    def fit(self, categories: pd.Series, effort: pd.Series) -> "MedianByCategory":
        # Pair each training project with its category so we can group.
        frame = pd.DataFrame({"category": categories.values, "effort": effort.values})
        # Per-category median effort: the lookup table used at prediction time.
        self.medians_ = frame.groupby("category")["effort"].median()
        # Global median as fallback for categories never seen in training.
        self.global_median_ = float(frame["effort"].median())
        return self

    def predict(self, categories: pd.Series) -> np.ndarray:
        # Map each category to its training median; unknown categories get
        # the global median instead of NaN.
        return (
            categories.map(self.medians_).fillna(self.global_median_).to_numpy(dtype=float)
        )


class LogSizeRegression:
    """OLS of log(effort) on log(size); prediction back-transformed to hours.

    Size is whatever the dataset offers (function points, LOC, story points);
    the unit must be recorded in the data dictionary, never mixed across
    datasets.
    """

    def fit(self, size: np.ndarray, effort: np.ndarray) -> "LogSizeRegression":
        size = np.asarray(size, dtype=float)
        effort = np.asarray(effort, dtype=float)
        # log() requires strictly positive inputs; zero/negative values are
        # data errors that must be filtered upstream.
        if np.any(size <= 0) or np.any(effort <= 0):
            raise ValueError("log-log regression requires positive size and effort")
        # Move to log-log space, where the classic power law
        # effort = a * size^b becomes the straight line log(effort) = log(a) + b*log(size).
        x = np.log(size)
        y = np.log(effort)
        # Ordinary least squares fit of that line: slope_ estimates the
        # exponent b, intercept_ estimates log(a).
        self.slope_, self.intercept_ = np.polyfit(x, y, deg=1)
        return self

    def predict(self, size: np.ndarray) -> np.ndarray:
        size = np.asarray(size, dtype=float)
        if np.any(size <= 0):
            raise ValueError("size must be positive")
        # Evaluate the fitted line in log space, then exponentiate to return
        # predictions in the original effort unit.
        return np.exp(self.intercept_ + self.slope_ * np.log(size))


def expert_estimates(df: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    """Expert baseline: the estimate recorded in the dataset itself.

    Returns (mask, estimates) where mask marks the rows that actually carry
    an expert estimate; the model-vs-human comparison runs on that subset
    only (~19% of JOSSE, all of SiP).
    """
    # Coerce the raw column to numbers; non-numeric markers ("n/a", text)
    # become NaN rather than raising.
    values = pd.to_numeric(df[column], errors="coerce")
    # Boolean mask of rows where an expert estimate actually exists.
    mask = values.notna().to_numpy()
    return mask, values.to_numpy(dtype=float)
