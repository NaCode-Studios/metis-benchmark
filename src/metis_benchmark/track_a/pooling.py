"""Hierarchical partial pooling of the log-size power-law slope (v1.3 Phase 3).

Small datasets estimate their size→effort elasticity `b_d` from few points, so
`b_d` is noisy. Partial pooling shrinks each dataset's slope toward a global
`b_0` by an amount set by how much slopes actually vary between datasets
(empirical Bayes): a dataset with a precise local slope barely moves, a dataset
with a noisy slope leans on the global. The intercept `a_d` is NOT pooled — it
carries the effort unit (person-hours vs person-months) and is refit per
dataset on the pooled slope.

Leakage discipline (protocol v1.3 addendum): the global `b_0` and the
between-dataset variance `τ²` are estimated only from rows that are never test
rows anywhere — the initial training window of each rolling-origin dataset.
Dateless datasets receive shrinkage but do not contribute to the global.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _ols_loglog(size: np.ndarray, effort: np.ndarray) -> tuple[float, float, float]:
    """OLS of log(effort) on log(size). Returns (intercept, slope, slope_var)."""
    x = np.log(np.asarray(size, dtype=float))
    y = np.log(np.asarray(effort, dtype=float))
    n = len(x)
    xbar = x.mean()
    sxx = np.sum((x - xbar) ** 2)
    if n < 3 or sxx <= 0:
        # Too few points / no size variation: slope undefined, return a flat
        # fit with infinite variance so pooling ignores this local estimate.
        return float(y.mean()), 0.0, np.inf
    slope = np.sum((x - xbar) * (y - y.mean())) / sxx
    intercept = y.mean() - slope * xbar
    resid = y - (intercept + slope * x)
    sigma2 = np.sum(resid**2) / (n - 2)  # residual variance
    slope_var = sigma2 / sxx  # variance of the slope estimate
    return float(intercept), float(slope), float(slope_var)


def _dersimonian_laird(slopes: np.ndarray, variances: np.ndarray) -> tuple[float, float]:
    """Global slope b_0 and between-dataset variance τ² (random-effects).

    The DerSimonian–Laird method-of-moments estimator: it compares the observed
    spread of the per-dataset slopes (Q) against the spread expected from their
    within-dataset noise alone, attributing the excess to τ².
    """
    w = 1.0 / variances  # fixed-effect (inverse-variance) weights
    b_fixed = np.sum(w * slopes) / np.sum(w)
    Q = np.sum(w * (slopes - b_fixed) ** 2)  # weighted heterogeneity statistic
    k = len(slopes)
    c = np.sum(w) - np.sum(w**2) / np.sum(w)
    # Excess dispersion beyond within-noise, mapped to a variance; floored at 0.
    tau2 = max(0.0, (Q - (k - 1)) / c) if c > 0 else 0.0
    # Re-weight by total (within + between) variance for the global slope.
    w_re = 1.0 / (variances + tau2)
    b_0 = np.sum(w_re * slopes) / np.sum(w_re)
    return float(b_0), float(tau2)


@dataclass
class PooledPowerLaw:
    """Fitted global slope and between-dataset variance, plus shrinkage."""

    b_0: float
    tau2: float

    @classmethod
    def fit_global(cls, contributions: list[tuple[np.ndarray, np.ndarray]]) -> "PooledPowerLaw":
        """Estimate the global slope from each contributor's (size, effort).

        `contributions` are the never-test initial-window rows of the
        contributing datasets. Datasets with an undefined local slope
        (too few points / no size variation) are dropped from the global.
        """
        slopes, variances = [], []
        for size, effort in contributions:
            _, b, var = _ols_loglog(size, effort)
            if np.isfinite(var):
                slopes.append(b)
                variances.append(var)
        slopes = np.asarray(slopes, dtype=float)
        variances = np.asarray(variances, dtype=float)
        if len(slopes) < 2:
            # Not enough contributors to pool; fall back to a neutral elasticity
            # of 1 (effort proportional to size) with infinite spread (no shrink).
            return cls(b_0=1.0, tau2=np.inf)
        b_0, tau2 = _dersimonian_laird(slopes, variances)
        return cls(b_0=b_0, tau2=tau2)

    def baseline_log(self, size_tr: np.ndarray, effort_tr: np.ndarray, size_eval: np.ndarray) -> np.ndarray:
        """Pooled-slope baseline in log space, fit on one dataset's training rows.

        The local slope is shrunk toward the global; the intercept is then
        refit on the training rows (absorbing this dataset's unit).
        """
        _, b_local, var_local = _ols_loglog(size_tr, effort_tr)
        if not np.isfinite(var_local) or not np.isfinite(self.tau2):
            # No usable local slope, or no global spread: use whichever exists.
            b_pool = self.b_0 if not np.isfinite(var_local) else b_local
        else:
            # Precision-weighted shrink toward the global (empirical Bayes).
            w_local = 1.0 / var_local if var_local > 0 else np.inf
            w_global = 1.0 / self.tau2 if self.tau2 > 0 else np.inf
            if np.isinf(w_local):
                b_pool = b_local
            elif np.isinf(w_global):
                b_pool = self.b_0
            else:
                b_pool = (b_local * w_local + self.b_0 * w_global) / (w_local + w_global)
        # Refit the per-dataset intercept given the pooled slope (unit absorber).
        x_tr = np.log(np.asarray(size_tr, dtype=float))
        y_tr = np.log(np.asarray(effort_tr, dtype=float))
        a_pool = np.mean(y_tr - b_pool * x_tr)
        return a_pool + b_pool * np.log(np.asarray(size_eval, dtype=float))
