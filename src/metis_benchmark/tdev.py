"""TDEV (schedule) power law: fit and interval prediction from real durations.

The product derives workforce from the COCOMO-II schedule equation
TDEV = c * PM^e with literature constants (c=3.67, e=0.30) that were never
validated against data. Several Track A datasets record the REAL project
duration — correctly excluded as an effort *feature* (it is an outcome, so it
would leak the target), but perfectly legitimate as a *target* of its own:
schedule is one of the three product dimensions (hours, cost, workforce).

The model is a power law, so the fit is ordinary least squares in log-log
space: log(TDEV) = log(c) + e*log(PM). OLS on logs is the maximum-likelihood
fit under multiplicative log-normal noise — exactly the error structure
schedule data shows (a 24-month project misses by months, a 2-month project by
weeks). The residual sigma of that regression is then the honest, measured
uncertainty band around any predicted duration.

Post-G0 additive module: nothing existing imports it, no default changes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Product convention: one person-month = 152 person-hours (the same constant
# metis-core uses to derive person-months from estimated hours).
HOURS_PER_PERSON_MONTH = 152.0
# Calendar conversion: 52 weeks / 12 months. Durations are fitted in months
# (the datasets' native unit) and reported in weeks (the product's unit).
WEEKS_PER_MONTH = 52.0 / 12.0
# COCOMO-II literature schedule constants, the product's current defaults.
LITERATURE_C = 3.67
LITERATURE_E = 0.30


@dataclass(frozen=True)
class TdevFit:
    """A fitted TDEV = c * PM^e law with bootstrap uncertainty.

    c, e        : the fitted multiplier and exponent
    sigma       : residual std of log(TDEV) (multiplicative noise scale)
    r2          : R^2 of the log-log regression (share of log-variance explained)
    n           : number of (PM, TDEV) pairs the fit consumed
    c_ci, e_ci  : 95% percentile-bootstrap CIs on c and e
    """

    c: float
    e: float
    sigma: float
    r2: float
    n: int
    c_ci: tuple[float, float]
    e_ci: tuple[float, float]


def fit_tdev(
    pm: np.ndarray,
    tdev_months: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> TdevFit:
    """Fit TDEV = c * PM^e by OLS on logs, with bootstrap CIs on (c, e).

    `pm` is effort in person-months, `tdev_months` the realized duration in
    months. Both must be strictly positive (log-log fit). The bootstrap
    resamples (PM, TDEV) pairs jointly, so the CIs reflect the actual project
    sample, not an assumed error law.
    """
    pm = np.asarray(pm, dtype=float)
    tdev_months = np.asarray(tdev_months, dtype=float)
    if pm.shape != tdev_months.shape:
        raise ValueError(f"shape mismatch: {pm.shape} vs {tdev_months.shape}")
    if np.any(pm <= 0) or np.any(tdev_months <= 0):
        raise ValueError("fit_tdev requires strictly positive PM and durations")
    n = len(pm)
    # Two free parameters (log c, e) need at least 3 points for a residual df.
    if n < 3:
        raise ValueError(f"fit_tdev needs at least 3 pairs, got {n}")

    x = np.log(pm)
    z = np.log(tdev_months)

    def ols(xs: np.ndarray, zs: np.ndarray) -> tuple[float, float]:
        # polyfit degree 1: slope = exponent e, intercept = log(c).
        slope, intercept = np.polyfit(xs, zs, 1)
        return float(np.exp(intercept)), float(slope)

    c, e = ols(x, z)
    resid = z - (np.log(c) + e * x)
    # ddof=2: two parameters were estimated from the same data.
    sigma = float(np.sqrt(np.sum(resid**2) / (n - 2)))
    # R^2 in log space: 1 - SSR/SST of log-duration.
    sst = float(np.sum((z - z.mean()) ** 2))
    r2 = float(1.0 - np.sum(resid**2) / sst) if sst > 0 else 0.0

    # Pairs bootstrap on (c, e): resample projects with replacement, refit.
    rng = np.random.default_rng(seed)
    cs = np.empty(n_boot)
    es = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        # A degenerate replicate (all-identical PM) has no slope; polyfit would
        # be singular. Redraw — the event has vanishing probability for n >= 3.
        while np.ptp(x[idx]) == 0.0:
            idx = rng.integers(0, n, n)
        cs[b], es[b] = ols(x[idx], z[idx])
    c_lo, c_hi = np.percentile(cs, [2.5, 97.5])
    e_lo, e_hi = np.percentile(es, [2.5, 97.5])

    return TdevFit(
        c=c, e=e, sigma=sigma, r2=r2, n=n,
        c_ci=(float(c_lo), float(c_hi)), e_ci=(float(e_lo), float(e_hi)),
    )


def predict_tdev_interval(
    pm: np.ndarray | float,
    c: float,
    e: float,
    sigma: float,
    level: float = 0.90,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predicted duration in WEEKS with a two-sided uncertainty band.

    Returns (point, lower, upper), each in weeks. The point is c * PM^e months
    converted to weeks; the band multiplies it by exp(±z * sigma), the
    two-sided `level` interval of the fitted multiplicative log-normal noise —
    so the band is symmetric in log space and always strictly positive.
    """
    # Lazy import: only the interval needs the normal quantile, and scipy is
    # already a declared dependency of the package.
    from scipy.stats import norm

    pm_arr = np.asarray(pm, dtype=float)
    if np.any(pm_arr <= 0):
        raise ValueError("predict_tdev_interval requires strictly positive PM")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if not 0.0 < level < 1.0:
        raise ValueError("level must be in (0, 1)")
    # Point prediction of the power law, months -> weeks.
    point = c * pm_arr**e * WEEKS_PER_MONTH
    # Two-sided normal quantile in log space, e.g. 1.645 at level=0.90.
    z = float(norm.ppf(1.0 - (1.0 - level) / 2.0))
    lower = point * np.exp(-z * sigma)
    upper = point * np.exp(z * sigma)
    return point, lower, upper
