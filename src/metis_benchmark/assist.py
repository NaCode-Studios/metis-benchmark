"""Inverse-variance assist blending (post-G0 R&D; additive, gate untouched).

The product's assist mode blends the model p50 with the client's own expert
estimate using a FIXED weight the user picks by hand. That weight should be
set by information, not by taste: if two noisy log-space estimators of the
same quantity are combined linearly, the variance-minimizing weights are the
inverse error variances (the classic precision-weighting result). Working in
log space is not cosmetic — effort errors are multiplicative (log-normal,
confirmed in EDA), so the arithmetic blend of raw hours would let the larger
estimate dominate; the log blend is a geometric weighted mean, symmetric in
over/under-estimation.

All functions are pure and unit-agnostic: sigma values are standard
deviations of log(actual) - log(estimate), measured OUT-OF-SAMPLE on blocks
the model never trained on (the caller owns that discipline; see
scripts/run_assist_ivw.py for the protocol-consistent measurement).
"""

from __future__ import annotations

import numpy as np


def log_residual_sigma(actual: np.ndarray, estimate: np.ndarray) -> float:
    """Sample SD (ddof=1) of log(actual) - log(estimate): the log error scale.

    This is the sigma the inverse-variance weight consumes. It is the SD, not
    the RMS: a constant multiplicative bias (mean log residual != 0) is a
    separate, correctable quantity and deliberately does not inflate the
    spread — the weight should reflect irreducible noise, not fixable offset.
    """
    actual = np.asarray(actual, dtype=float)
    estimate = np.asarray(estimate, dtype=float)
    if actual.shape != estimate.shape:
        raise ValueError(f"shape mismatch: {actual.shape} vs {estimate.shape}")
    # log residuals need strictly positive values on both sides.
    if np.any(actual <= 0) or np.any(estimate <= 0):
        raise ValueError("actual and estimate must be strictly positive")
    resid = np.log(actual) - np.log(estimate)
    # ddof=1 needs at least two residuals; one residual has no spread.
    if len(resid) < 2:
        raise ValueError("need at least 2 pairs to estimate a residual sigma")
    return float(np.std(resid, ddof=1))


def inverse_variance_weight(sigma_model: float, sigma_expert: float) -> float:
    """Weight on the MODEL: w = sigma_m^-2 / (sigma_m^-2 + sigma_e^-2).

    Algebraically rewritten as sigma_e^2 / (sigma_e^2 + sigma_m^2) so the
    limits are finite without special-casing division by zero:
      sigma_expert -> 0  =>  w -> 0  (perfect expert: trust the expert),
      sigma_model  -> 0  =>  w -> 1  (perfect model: trust the model),
      sigma_expert == sigma_model  =>  w = 0.5 (equal information).
    Both sigmas zero is ill-posed (two perfect estimators); by symmetry we
    return 0.5, where the blend is exact anyway because both agree with truth.
    """
    if sigma_model < 0 or sigma_expert < 0:
        raise ValueError("sigmas are standard deviations and must be >= 0")
    denom = sigma_expert**2 + sigma_model**2
    if denom == 0.0:
        return 0.5
    return float(sigma_expert**2 / denom)


def blend_log_space(p50_model: np.ndarray, expert: np.ndarray, w: float) -> np.ndarray:
    """Geometric blend: exp(w·log(model) + (1-w)·log(expert)).

    The log-space convex combination of two positive estimates — equal to the
    weighted geometric mean, and the estimator whose variance the
    inverse-variance w minimizes when both errors are log-normal.
    """
    if not 0.0 <= w <= 1.0:
        raise ValueError(f"blend weight must be in [0, 1], got {w}")
    p50_model = np.asarray(p50_model, dtype=float)
    expert = np.asarray(expert, dtype=float)
    if p50_model.shape != expert.shape:
        raise ValueError(f"shape mismatch: {p50_model.shape} vs {expert.shape}")
    if np.any(p50_model <= 0) or np.any(expert <= 0):
        raise ValueError("estimates must be strictly positive for a log blend")
    return np.exp(w * np.log(p50_model) + (1.0 - w) * np.log(expert))


def ivw_blend(
    p50_model: np.ndarray,
    expert: np.ndarray,
    sigma_model: float,
    sigma_expert: float,
) -> np.ndarray:
    """Inverse-variance weighted blend of model p50 and expert estimate.

    Convenience composition: computes w from the two measured log-sigmas and
    applies the log-space blend. Pure function — measuring the sigmas
    out-of-sample (never on the rows being blended) is the caller's job.
    """
    w = inverse_variance_weight(sigma_model, sigma_expert)
    return blend_log_space(p50_model, expert, w)
