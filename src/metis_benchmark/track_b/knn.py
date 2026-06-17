"""k-NN Nadaraya-Watson effort predictor over text embeddings (V20 Fig 3).

For a query task, retrieve the k most cosine-similar training tasks and predict
a kernel-weighted mean of their (log) efforts. The weight decays with angular
distance: a neighbour at similarity 0.94 should count far more than one at 0.78,
which a plain median would ignore. The prediction is formed in log space because
effort is log-normal, then exponentiated back.

    ŷ = exp( Σ wᵢ·log(yᵢ) / Σ wᵢ ),   wᵢ = exp((sᵢ − 1)/τ)

τ controls how fast the weight falls off with similarity (small τ = only the
nearest neighbours matter). k and τ are selected by test-blind train-internal
validation; this class just applies a given (k, τ).
"""

from __future__ import annotations

import numpy as np


class NadarayaWatsonKNN:
    """Cosine k-NN with exponential-similarity kernel weighting, in log space."""

    def __init__(self, k: int = 20, tau: float = 0.1) -> None:
        self.k = k
        self.tau = tau

    def fit(self, emb_train: np.ndarray, effort_train: np.ndarray) -> "NadarayaWatsonKNN":
        # Store the (normalized) training embeddings and their log-efforts; the
        # "index" is just the matrix — retrieval is a single matmul at predict.
        self._emb = np.asarray(emb_train, dtype=np.float32)
        effort_train = np.asarray(effort_train, dtype=float)
        if np.any(effort_train <= 0):
            raise ValueError("effort must be positive for log-space weighting")
        self._logy = np.log(effort_train)
        return self

    def predict(self, emb_query: np.ndarray) -> np.ndarray:
        emb_query = np.asarray(emb_query, dtype=np.float32)
        # Cosine similarity of every query against every training task (both are
        # unit-normalized, so the dot product is the cosine).
        sims = emb_query @ self._emb.T  # (n_query, n_train)
        k = min(self.k, sims.shape[1])
        # Indices of the top-k most similar training tasks per query.
        topk = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
        rows = np.arange(sims.shape[0])[:, None]
        s = sims[rows, topk]  # (n_query, k) similarities of the neighbours
        logy = self._logy[topk]  # (n_query, k) their log-efforts
        # Exponential kernel weight: closer (higher s) -> exponentially larger.
        w = np.exp((s - 1.0) / self.tau)
        wsum = w.sum(axis=1)
        # Weighted mean of log-efforts, exponentiated back to the effort scale.
        logpred = (w * logy).sum(axis=1) / np.where(wsum > 0, wsum, 1.0)
        return np.exp(logpred)
