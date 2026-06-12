"""The three mandatory baselines of the G0 protocol.

The engine must beat all three on every dataset where they are computable:

1. median effort by category,
2. linear regression of log(effort) on log(size),
3. the human expert estimate, where the dataset records one (JOSSE, SiP).

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
        frame = pd.DataFrame({"category": categories.values, "effort": effort.values})
        self.medians_ = frame.groupby("category")["effort"].median()
        self.global_median_ = float(frame["effort"].median())
        return self

    def predict(self, categories: pd.Series) -> np.ndarray:
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
        if np.any(size <= 0) or np.any(effort <= 0):
            raise ValueError("log-log regression requires positive size and effort")
        x = np.log(size)
        y = np.log(effort)
        self.slope_, self.intercept_ = np.polyfit(x, y, deg=1)
        return self

    def predict(self, size: np.ndarray) -> np.ndarray:
        size = np.asarray(size, dtype=float)
        if np.any(size <= 0):
            raise ValueError("size must be positive")
        return np.exp(self.intercept_ + self.slope_ * np.log(size))


def expert_estimates(df: pd.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    """Expert baseline: the estimate recorded in the dataset itself.

    Returns (mask, estimates) where mask marks the rows that actually carry
    an expert estimate; the model-vs-human comparison runs on that subset
    only (~19% of JOSSE, all of SiP).
    """
    values = pd.to_numeric(df[column], errors="coerce")
    mask = values.notna().to_numpy()
    return mask, values.to_numpy(dtype=float)
