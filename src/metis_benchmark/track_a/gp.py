"""Gaussian Process regressor for the Track A tabular channel.

This is the engine's primary regressor for the small-data regime (the V20
strategy: GP first, gradient boosting once volumes grow). A GP is chosen here
because it returns a calibrated predictive variance for free, which feeds the
conformal interval step later — though the gate intervals are conformalized,
not taken from the Gaussian assumption directly.

Effort is modeled in log space (z = log(effort)): software effort is roughly
log-normal (confirmed in EDA, |skew| < 0.7 on every dataset), so the additive
GP noise is far more reasonable on the log scale than on raw hours.

Mean function (v1.3): the GP can fit the *residual* of a log-size power law
instead of effort from a zero mean. The caller supplies `baseline_log` — the
log-space prediction of the size law `log(a) + b·log(size)` — at fit and
predict time; the GP then models only `log(effort) - baseline_log` over the
local features. This injects the strong, well-understood size→effort signal as
structure rather than forcing the GP to relearn it from few points, and lets a
*pooled* size law (Phase 3) feed the same GP unchanged. With `baseline_log=None`
the GP keeps its original zero-mean behavior.
"""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
from sklearn.preprocessing import StandardScaler


class LogGaussianProcess:
    """GP on standardized features predicting log-effort.

    predict() returns the median effort estimate (p50) in the original unit.
    predict_log() exposes the log-space mean and standard deviation for the
    conformal/quantile machinery added in the next step.
    """

    def __init__(self, seed: int = 0, log_features: bool = True) -> None:
        # Function-point counts are heavy-tailed (China AFP ranges 9..17518);
        # a log1p transform compresses that tail into a near-linear scale so the
        # RBF length-scales become meaningful instead of being dominated by a
        # handful of huge projects. log1p (not log) tolerates zero counts.
        self._log_features = log_features
        # Feature standardization: a GP's length-scales assume comparable
        # feature magnitudes, and China's columns span very different ranges.
        self._scaler = StandardScaler()
        # Kernel = signal variance * RBF (smooth non-linear trend) + white
        # noise (irreducible scatter between similarly-sized projects).
        # Bounds are wide so marginal-likelihood optimization can move freely.
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(
            length_scale=1.0, length_scale_bounds=(1e-1, 1e2)
        ) + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e1))
        # normalize_y centers the log-targets; n_restarts re-runs the kernel
        # optimizer from several inits to avoid poor local optima.
        self._gp = GaussianProcessRegressor(
            kernel=kernel,
            normalize_y=True,
            n_restarts_optimizer=4,
            random_state=seed,
        )

    def _transform(self, X: np.ndarray, *, fit: bool) -> np.ndarray:
        # Optional log1p on the raw features, then standardization. The scaler
        # is fit on the training fold only (fit=True) and merely applied at
        # predict time (fit=False) to avoid leaking test statistics.
        X = np.asarray(X, dtype=float)
        if self._log_features:
            X = np.log1p(X)
        return self._scaler.fit_transform(X) if fit else self._scaler.transform(X)

    def fit(
        self, X: np.ndarray, effort: np.ndarray, baseline_log: np.ndarray | None = None
    ) -> "LogGaussianProcess":
        effort = np.asarray(effort, dtype=float)
        # log() requires positive effort; the caller filters non-positive rows.
        if np.any(effort <= 0):
            raise ValueError("effort must be positive for log-space modeling")
        # Target is log-effort, optionally minus a size-law baseline (mean
        # function): the GP then learns only what the size law leaves over.
        target = np.log(effort)
        if baseline_log is not None:
            target = target - np.asarray(baseline_log, dtype=float)
        # Transform features (train-fold statistics), then fit the GP on the target.
        self._gp.fit(self._transform(X, fit=True), target)
        return self

    def predict_log(
        self, X: np.ndarray, baseline_log: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        # GP posterior over the (residual) target; re-add the baseline so the
        # mean is back on the log-effort scale. The std is unchanged by the
        # additive baseline shift.
        mean, std = self._gp.predict(self._transform(X, fit=False), return_std=True)
        if baseline_log is not None:
            mean = mean + np.asarray(baseline_log, dtype=float)
        return mean, std

    def predict(self, X: np.ndarray, baseline_log: np.ndarray | None = None) -> np.ndarray:
        # Point estimate in the original unit. exp(mean) maps the log-space
        # mean back to hours; because log is monotone this is the predicted
        # median (p50), which is exactly what PRED(25)/MdAPE should score.
        mean, _ = self.predict_log(X, baseline_log)
        return np.exp(mean)
