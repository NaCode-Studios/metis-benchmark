"""Conformalized Quantile Regression (Romano, Patterson, Candès 2019).

The gate's promise is not just a point estimate but an interval that keeps its
word: a nominal-90% interval must actually contain the true effort ~90% of the
time on unseen projects. CQR delivers that distribution-free and finite-sample,
on top of any quantile regressor — even a miscalibrated one — by learning a
single additive correction on a held-out calibration set.

Why a from-scratch implementation: it is a dozen lines, it is fully transparent
for review, and it avoids MAPIE's 1.x API churn. The method is exactly the one
the protocol cites.

Post-G0 additive extensions (defaults preserve the G0 behavior bit-per-bit):
- `space="log"`: the same CQR machinery, but scores are computed on log(y) and
  the shift is applied in log space. For a log-normal target one additive raw
  correction cannot fit every scale — the same q hours swamps small projects
  and vanishes on large ones — whereas a log shift is a uniform *multiplicative*
  correction, scale-equivariant by construction.
- `MondrianConformalizedQuantile`: per-group (Mondrian) calibration keyed by
  training log-size terciles, with an explicit marginal fallback when a group
  has too few calibration points for the finite-sample guarantee to bite.
"""

from __future__ import annotations

import numpy as np


def _nonconformity(
    lower: np.ndarray, upper: np.ndarray, y: np.ndarray, space: str
) -> np.ndarray:
    """CQR nonconformity scores in the chosen space.

    space="raw": the classic score max(lower - y, y - upper) in target units.
    space="log": the same score on log-transformed quantiles and target, so a
    later shift by q in log space equals multiplying the raw bounds by exp(q):
    one correction that scales with the project instead of a fixed hour count.
    """
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    y = np.asarray(y, dtype=float)
    if space == "raw":
        # Signed distance outside the predicted band: positive when the point
        # falls outside [lower, upper]; negative inside.
        return np.maximum(lower - y, y - upper)
    if space == "log":
        # log() is only defined for positive values; effort and its quantiles
        # are positive by construction, so a violation is a caller bug.
        if np.any(lower <= 0) or np.any(upper <= 0) or np.any(y <= 0):
            raise ValueError("space='log' requires strictly positive quantiles and targets")
        # Same signed-distance score, measured in log units (i.e. in ratios).
        return np.maximum(np.log(lower) - np.log(y), np.log(y) - np.log(upper))
    raise ValueError(f"unknown space {space!r}; expected 'raw' or 'log'")


def _finite_sample_q(scores: np.ndarray, alpha: float) -> float:
    """The finite-sample CQR correction: a conservative empirical quantile.

    The ceil((n+1)(1-alpha))/n-th empirical quantile of the scores guarantees
    >= 1-alpha coverage in expectation; "higher" interpolation keeps the
    conservative side of the guarantee.
    """
    n = len(scores)
    level = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
    return float(np.quantile(scores, level, method="higher"))


class ConformalizedQuantile:
    """Two-sided CQR at nominal coverage 1 - alpha.

    The base regressor supplies a lower and an upper quantile per point; CQR
    shifts both outward by one scalar `q`, chosen on the calibration set so the
    interval attains the nominal coverage with a finite-sample guarantee.
    Calibration is per dataset/scale: effort units differ across datasets, so
    the correction is fit on each calibration set, never pooled across units.

    `space` selects where the correction lives (post-G0, additive):
    - "raw" (default, the G0 behavior): q is in target units and the interval
      is [lower - q, upper + q].
    - "log": q is in log units and the interval is
      [lower * exp(-q), upper * exp(q)] — a uniform multiplicative widening.
    """

    def __init__(self, alpha: float = 0.10, space: str = "raw") -> None:
        # alpha = 0.10 -> nominal two-sided coverage of 90%.
        self.alpha = alpha
        # Validate eagerly so a typo fails at construction, not at calibrate().
        if space not in ("raw", "log"):
            raise ValueError(f"unknown space {space!r}; expected 'raw' or 'log'")
        self.space = space
        self.q_: float | None = None

    def calibrate(
        self, lower_cal: np.ndarray, upper_cal: np.ndarray, y_cal: np.ndarray
    ) -> "ConformalizedQuantile":
        y_cal = np.asarray(y_cal, dtype=float)
        # CQR nonconformity score in the configured space (raw units or log).
        scores = _nonconformity(lower_cal, upper_cal, y_cal, self.space)
        # Finite-sample level: the ceil((n+1)(1-alpha))/n-th empirical quantile
        # of the scores guarantees >= 1-alpha coverage in expectation.
        self.q_ = _finite_sample_q(scores, self.alpha)
        return self

    def interval(self, lower: np.ndarray, upper: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self.q_ is None:
            raise RuntimeError("calibrate() must be called before interval()")
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        # Shift both quantiles outward by the calibrated correction. A negative
        # q (over-wide base interval) is allowed: CQR can also tighten.
        if self.space == "raw":
            return lower - self.q_, upper + self.q_
        # Log-space shift back-transformed: exp(log(bound) -/+ q) is a uniform
        # multiplicative correction, so the lower bound stays positive and the
        # widening is proportional to the project's own scale.
        return lower * np.exp(-self.q_), upper * np.exp(self.q_)


def log_size_tercile_edges(size_train: np.ndarray) -> np.ndarray:
    """Inner tercile edges of log(size) computed on TRAINING sizes only.

    The Mondrian partition must be fixed before looking at calibration or test
    labels, otherwise the per-group guarantee is broken; deriving the edges
    from the training window keeps the partition test-blind. Log-size is the
    natural axis: project sizes are heavy-tailed, so raw-size terciles would
    collapse small and medium projects into one bin.
    """
    size_train = np.asarray(size_train, dtype=float)
    if np.any(size_train <= 0):
        raise ValueError("sizes must be strictly positive for log terciles")
    # The 33.3rd and 66.7th percentiles split the training log-sizes in three.
    return np.quantile(np.log(size_train), [1 / 3, 2 / 3])


def assign_terciles(size: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Map each size to its tercile group: 0=small, 1=medium, 2=large.

    Uses the *training-derived* edges so calibration and test points fall into
    the same partition the calibration was computed on.
    """
    size = np.asarray(size, dtype=float)
    if np.any(size <= 0):
        raise ValueError("sizes must be strictly positive for log terciles")
    # searchsorted with side="right" puts a point equal to an edge into the
    # upper bin, matching the half-open binning convention of np.digitize.
    return np.searchsorted(np.asarray(edges, dtype=float), np.log(size), side="right")


class MondrianConformalizedQuantile:
    """Mondrian (group-conditional) CQR with an explicit marginal fallback.

    One global q equalizes *marginal* coverage but can hide conditional gaps:
    the correction that fits medium projects may over-cover small ones and
    under-cover large ones. Mondrian CQR calibrates one q per group (here:
    training log-size terciles), which restores the finite-sample guarantee
    *within* each group — but only if the group has enough calibration points:
    with n_cal points the guaranteed coverage is ceil((n_cal+1)(1-alpha))/n_cal
    quantile-based, so tiny groups force q to the sample maximum and the
    "guarantee" degenerates. Below `min_group_cal` points the group therefore
    falls back, explicitly and observably (see `fallback_groups_`), to the
    marginal q computed on the full calibration set.

    Default `min_group_cal=15`: at alpha=0.10, n_cal=15 puts the finite-sample
    level at ceil(16*0.9)/15 = 15/15 — exactly the sample maximum. Fewer points
    cannot even express the 90th percentile, so 15 is the smallest calibration
    group where per-group calibration is meaningfully different from "take the
    worst score seen". The threshold is a parameter, not a constant, so a
    caller with a different alpha can tighten or relax it.
    """

    def __init__(
        self, alpha: float = 0.10, space: str = "raw", min_group_cal: int = 15
    ) -> None:
        self.alpha = alpha
        if space not in ("raw", "log"):
            raise ValueError(f"unknown space {space!r}; expected 'raw' or 'log'")
        self.space = space
        self.min_group_cal = int(min_group_cal)
        # group id -> per-group correction (populated by calibrate()).
        self.q_by_group_: dict[int, float] | None = None
        # The marginal correction, used for fallback groups and unseen groups.
        self.q_marginal_: float | None = None
        # Groups that fell back to the marginal q (too few calibration points).
        self.fallback_groups_: set[int] | None = None

    def calibrate(
        self,
        lower_cal: np.ndarray,
        upper_cal: np.ndarray,
        y_cal: np.ndarray,
        groups_cal: np.ndarray,
    ) -> "MondrianConformalizedQuantile":
        groups_cal = np.asarray(groups_cal)
        # Scores are computed once in the configured space; grouping only
        # changes *which* scores each correction is a quantile of.
        scores = _nonconformity(lower_cal, upper_cal, y_cal, self.space)
        # Marginal q on the full calibration set: the fallback correction.
        self.q_marginal_ = _finite_sample_q(scores, self.alpha)
        self.q_by_group_ = {}
        self.fallback_groups_ = set()
        for g in np.unique(groups_cal):
            mask = groups_cal == g
            if int(mask.sum()) < self.min_group_cal:
                # Too few points for the per-group finite-sample guarantee:
                # fall back to the marginal correction, and record it so the
                # caller can report which groups are NOT group-calibrated.
                self.q_by_group_[int(g)] = self.q_marginal_
                self.fallback_groups_.add(int(g))
            else:
                self.q_by_group_[int(g)] = _finite_sample_q(scores[mask], self.alpha)
        return self

    def interval(
        self, lower: np.ndarray, upper: np.ndarray, groups: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if self.q_by_group_ is None or self.q_marginal_ is None:
            raise RuntimeError("calibrate() must be called before interval()")
        lower = np.asarray(lower, dtype=float)
        upper = np.asarray(upper, dtype=float)
        groups = np.asarray(groups)
        # Per-point correction: the group's q, or the marginal q for a group
        # never seen in calibration (possible when test sizes drift).
        q = np.array([self.q_by_group_.get(int(g), self.q_marginal_) for g in groups])
        if self.space == "raw":
            return lower - q, upper + q
        return lower * np.exp(-q), upper * np.exp(q)
