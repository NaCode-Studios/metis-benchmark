"""Conformalized Quantile Regression (Romano, Patterson, Candès 2019).

The gate's promise is not just a point estimate but an interval that keeps its
word: a nominal-90% interval must actually contain the true effort ~90% of the
time on unseen projects. CQR delivers that distribution-free and finite-sample,
on top of any quantile regressor — even a miscalibrated one — by learning a
single additive correction on a held-out calibration set.

Why a from-scratch implementation: it is a dozen lines, it is fully transparent
for review, and it avoids MAPIE's 1.x API churn. The method is exactly the one
the protocol cites.
"""

from __future__ import annotations

import numpy as np


class ConformalizedQuantile:
    """Two-sided CQR at nominal coverage 1 - alpha.

    The base regressor supplies a lower and an upper quantile per point; CQR
    shifts both outward by one scalar `q`, chosen on the calibration set so the
    interval attains the nominal coverage with a finite-sample guarantee.
    Calibration is per dataset/scale: effort units differ across datasets, so
    the correction is fit on each calibration set, never pooled across units.
    """

    def __init__(self, alpha: float = 0.10) -> None:
        # alpha = 0.10 -> nominal two-sided coverage of 90%.
        self.alpha = alpha
        self.q_: float | None = None

    def calibrate(
        self, lower_cal: np.ndarray, upper_cal: np.ndarray, y_cal: np.ndarray
    ) -> "ConformalizedQuantile":
        lower_cal = np.asarray(lower_cal, dtype=float)
        upper_cal = np.asarray(upper_cal, dtype=float)
        y_cal = np.asarray(y_cal, dtype=float)
        # CQR nonconformity score: signed distance outside the predicted band.
        # Positive when the point falls outside [lower, upper]; negative inside.
        scores = np.maximum(lower_cal - y_cal, y_cal - upper_cal)
        n = len(y_cal)
        # Finite-sample level: the ceil((n+1)(1-alpha))/n-th empirical quantile
        # of the scores guarantees >= 1-alpha coverage in expectation.
        level = min(1.0, np.ceil((n + 1) * (1 - self.alpha)) / n)
        # "higher" interpolation keeps the conservative side of the guarantee.
        self.q_ = float(np.quantile(scores, level, method="higher"))
        return self

    def interval(self, lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.q_ is None:
            raise RuntimeError("calibrate() must be called before interval()")
        # Shift both quantiles outward by the calibrated correction. A negative
        # q (over-wide base interval) is allowed: CQR can also tighten.
        return np.asarray(lower, dtype=float) - self.q_, np.asarray(upper, dtype=float) + self.q_
