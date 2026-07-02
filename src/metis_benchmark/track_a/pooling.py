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

Post-G0 addendum (additive, gate behavior unchanged): `fit_intercept_prior`
puts random effects on the very intercept the gate refits away, measuring the
BETWEEN-dataset spread `τ_a` of mean log-productivity — the epoch/organization
bias that hits cold-start estimates when there is no local history to refit on.
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


# --------------------------------------------------------------------------
# Post-G0 R&D (additive; nothing above changes): random effects on the
# INTERCEPT of log-productivity.
#
# The gate treats the intercept as a per-dataset nuisance ("it carries the
# unit and is refit per dataset"). That is exactly why cold-start estimates on
# a never-seen organization can be off by a large multiplicative factor: the
# intercept differs BETWEEN organizations/epochs, and without local actuals
# there is nothing to refit it on. Instead of discovering that bias a
# posteriori (the observed 6-8x on real projects), we quantify it a priori:
# treat each dataset's mean log-productivity as a draw from a normal
# population and estimate the population spread tau_a with the same
# DerSimonian-Laird estimator already used for the slope. e^(±2·tau_a) is
# then the honest 95% multiplicative range a cold-start estimate should
# declare before any local recalibration.
# --------------------------------------------------------------------------

# Minimum admissible within-dataset variance of a mean intercept. A dataset
# whose projects all share one exact productivity would report se^2 = 0 and an
# infinite fixed-effect weight, which breaks the DL moment equations; flooring
# at a negligible epsilon keeps the algebra finite while changing nothing
# numerically (in the all-zero limit DL then degenerates, correctly, to the
# ddof=1 sample variance of the intercepts).
_VAR_FLOOR = 1e-12


def log_productivity_intercept(size: np.ndarray, effort: np.ndarray) -> tuple[float, float, int]:
    """Per-dataset intercept of log-productivity: mean of log(effort/size).

    Normalizing the target to log(effort/size) fixes the size elasticity at 1,
    so the intercept is a pure productivity level (log effort-per-size-unit)
    and its per-dataset estimate is simply the sample mean, with standard
    error s/sqrt(n). Returns (intercept, se, n). Both inputs must be strictly
    positive and share one unit system across every dataset that will be
    pooled — a unit difference is indistinguishable from an organization
    offset in log space, which is precisely what tau_a must NOT absorb.
    """
    size = np.asarray(size, dtype=float)
    effort = np.asarray(effort, dtype=float)
    if size.shape != effort.shape:
        raise ValueError(f"shape mismatch: {size.shape} vs {effort.shape}")
    # log() is undefined at or below zero; a non-positive row is a data bug
    # upstream, not a modeling case.
    if np.any(size <= 0) or np.any(effort <= 0):
        raise ValueError("size and effort must be strictly positive")
    # The normalized target: one number per project, its log-productivity.
    z = np.log(effort / size)
    n = len(z)
    # One project cannot yield a standard error (ddof=1 needs n >= 2).
    if n < 2:
        raise ValueError("need at least 2 projects to estimate an intercept SE")
    # Sample mean and its SE: the within-dataset noise of the intercept
    # estimate, which DL subtracts from the observed between-dataset spread.
    se = float(np.std(z, ddof=1) / np.sqrt(n))
    return float(np.mean(z)), se, n


def _q_profile_tau_ci(
    estimates: np.ndarray, variances: np.ndarray, alpha: float = 0.05
) -> tuple[float, float]:
    """Q-profile confidence interval for the between-dataset SD tau.

    Viechtbauer (2007): under the random-effects model the generalized Q
    statistic Q(t2) = sum w_i(t2)·(a_i - a_hat(t2))^2 with w_i = 1/(v_i + t2)
    follows a chi-square with k-1 df when t2 is the TRUE between variance.
    The 95% CI is therefore the set of t2 whose Q(t2) falls between the 2.5%
    and 97.5% chi-square quantiles. Q is decreasing in t2, so each bound is a
    single root. This is the standard small-k interval (bootstrap over k=5
    datasets would be meaningless), and it is exact under normality.
    """
    from scipy.optimize import brentq  # scalar root finder for the Q profile
    from scipy.stats import chi2      # chi-square quantiles for the pivots

    k = len(estimates)

    def q_gen(t2: float) -> float:
        # Weights at candidate between-variance t2; the floor keeps them finite.
        w = 1.0 / (np.maximum(variances, _VAR_FLOOR) + t2)
        mu = np.sum(w * estimates) / np.sum(w)
        return float(np.sum(w * (estimates - mu) ** 2))

    hi_q = chi2.ppf(1 - alpha / 2, k - 1)  # large quantile -> LOWER tau bound
    lo_q = chi2.ppf(alpha / 2, k - 1)      # small quantile -> UPPER tau bound

    # Lower bound: if even t2=0 does not push Q above the upper chi-square
    # quantile, tau=0 is inside the interval and the bound is 0.
    if q_gen(0.0) <= hi_q:
        t2_lo = 0.0
    else:
        # Q is decreasing: bracket the root by expanding the right edge.
        upper = 1.0
        while q_gen(upper) > hi_q:
            upper *= 10.0
        t2_lo = brentq(lambda t2: q_gen(t2) - hi_q, 0.0, upper)

    # Upper bound: the t2 at which Q drops to the LOWER chi-square quantile.
    if q_gen(0.0) <= lo_q:
        # Degenerate: the data are more homogeneous than chance allows even at
        # t2=0 (possible with few datasets); the interval collapses to 0.
        t2_hi = 0.0
    else:
        upper = 1.0
        while q_gen(upper) > lo_q:
            upper *= 10.0
        t2_hi = brentq(lambda t2: q_gen(t2) - lo_q, 0.0, upper)

    return float(np.sqrt(t2_lo)), float(np.sqrt(t2_hi))


@dataclass(frozen=True)
class InterceptPrior:
    """Random-effects summary of the log-productivity intercept across datasets.

    tau_a is the between-dataset (organization/epoch) standard deviation of
    mean log-productivity: cold-starting a never-seen organization, the p50
    scale factor is expected within e^(±2·tau_a) at ~95%. a_0 is the pooled
    mean intercept (only meaningful within the unit system of the inputs).
    """

    a_0: float                       # pooled mean log-productivity intercept
    tau_a: float                     # between-dataset SD of the intercept (log space)
    tau_a_ci: tuple[float, float]    # Q-profile 95% CI for tau_a
    intercepts: tuple[float, ...]    # per-dataset intercept estimates a_d
    std_errors: tuple[float, ...]    # their standard errors se_d
    n_projects: tuple[int, ...]      # projects behind each a_d


def fit_intercept_prior(
    contributions: list[tuple[np.ndarray, np.ndarray]], alpha: float = 0.05
) -> InterceptPrior:
    """DerSimonian-Laird random effects on the log-productivity intercept.

    `contributions` are per-dataset (size, effort) arrays, all in ONE common
    unit system (e.g. function points and person-hours): the caller converts
    units BEFORE pooling, because in log space a unit mismatch is an additive
    intercept offset that would masquerade as organization bias. Each dataset
    yields (a_d, se_d) via `log_productivity_intercept`; DL then splits the
    observed spread of the a_d into within noise (se_d^2) and the between
    component tau_a^2 — the epoch/organization bias this experiment measures.
    """
    if len(contributions) < 2:
        # One dataset shows no between spread at all; refuse rather than
        # return a fake tau of 0.
        raise ValueError("intercept pooling needs at least 2 datasets")
    stats = [log_productivity_intercept(size, effort) for size, effort in contributions]
    a = np.array([s[0] for s in stats], dtype=float)
    se = np.array([s[1] for s in stats], dtype=float)
    n = tuple(s[2] for s in stats)
    # DL over (estimate, variance) pairs — the same moment estimator the slope
    # pooling uses; the variance floor keeps se=0 degenerations finite.
    a_0, tau2 = _dersimonian_laird(a, np.maximum(se**2, _VAR_FLOOR))
    ci = _q_profile_tau_ci(a, se**2, alpha=alpha)
    return InterceptPrior(
        a_0=a_0,
        tau_a=float(np.sqrt(tau2)),
        tau_a_ci=ci,
        intercepts=tuple(float(x) for x in a),
        std_errors=tuple(float(x) for x in se),
        n_projects=n,
    )
