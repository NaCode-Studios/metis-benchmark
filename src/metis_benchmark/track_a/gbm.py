"""Gradient-boosting quantile regressor for the Track A tabular channel.

The second regressor of the V20 strategy (GP for tiny data, gradient boosting
as signal/volume grow). XGBoost is trained with the pinball (quantile) loss so
it produces the p50 and p90 quantiles natively — the p50 is the point estimate
scored here, and both quantiles feed the conformal interval step (Step 3).

As with the GP, effort is modeled in log space and the heavy-tailed count
features are passed through log1p, so the trees split on a near-linear scale.
Hyperparameters are deliberately conservative (shallow trees, few rounds):
Track A training folds are small (12-145 rows) and a deep ensemble would just
memorize them. No hyperparameter is tuned against the test block.
"""

from __future__ import annotations

import numpy as np
from xgboost import XGBRegressor


class LogGBMQuantile:
    """XGBoost pinball-loss quantile regressor on log-effort.

    predict() returns the median (p50) estimate in the original unit;
    predict_quantiles() exposes every fitted quantile for interval building.
    """

    def __init__(self, quantiles: tuple[float, ...] = (0.5, 0.9), seed: int = 0,
                 log_features: bool = True) -> None:
        self._quantiles = quantiles
        self._log_features = log_features
        # One XGBoost model fits all requested quantiles jointly (vector-valued
        # quantile regression, XGBoost >= 2.0).
        self._model = XGBRegressor(
            objective="reg:quantileerror",
            quantile_alpha=np.asarray(quantiles, dtype=float),
            n_estimators=200,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_lambda=1.0,
            random_state=seed,
        )

    def _prep(self, X: np.ndarray) -> np.ndarray:
        # Same feature treatment as the GP: log1p compresses count tails.
        X = np.asarray(X, dtype=float)
        return np.log1p(X) if self._log_features else X

    def fit(self, X: np.ndarray, effort: np.ndarray) -> "LogGBMQuantile":
        effort = np.asarray(effort, dtype=float)
        if np.any(effort <= 0):
            raise ValueError("effort must be positive for log-space modeling")
        # Fit all quantiles on log-effort.
        self._model.fit(self._prep(X), np.log(effort))
        return self

    def predict_quantiles(self, X: np.ndarray) -> dict[float, np.ndarray]:
        # XGBoost returns shape (n, n_quantiles); exp() maps each back to hours.
        raw = np.atleast_2d(self._model.predict(self._prep(X)))
        # A single-quantile model returns (n,) -> reshape to (n, 1).
        if raw.shape[0] == 1 and len(self._quantiles) > 1:
            raw = raw.T
        if raw.ndim == 1:
            raw = raw[:, None]
        return {q: np.exp(raw[:, i]) for i, q in enumerate(self._quantiles)}

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Point estimate = the median quantile, back-transformed to hours.
        return self.predict_quantiles(X)[0.5]
