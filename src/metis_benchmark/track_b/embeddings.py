"""Local sentence embeddings for the Track B semantic channel.

Each task's text is projected to a dense vector with BAAI/bge-small-en-v1.5
(384-d, local, normalized). Embeddings are deterministic — the model is never
trained here — so they leak nothing about effort; they only encode meaning, so
that semantically similar tasks land close in cosine angle (V20 Fig 2).

Embedding 12k-23k texts is the slow step, so the matrix is cached to disk per
dataset (data/processed/emb_<key>_<model>.npy) and reused. The cache is keyed
by dataset, model and row count; a row-count mismatch invalidates it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

MODEL_NAME = "BAAI/bge-small-en-v1.5"
_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"

# Lazily instantiated singleton so the (heavy) model loads at most once.
_model = None


def _get_model():
    global _model
    if _model is None:
        # Imported lazily: sentence-transformers pulls torch, which we only want
        # to load when embeddings are actually needed.
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer(MODEL_NAME)
    return _model


def _cache_path(key: str, n_rows: int) -> Path:
    tag = hashlib.md5(f"{MODEL_NAME}:{n_rows}".encode()).hexdigest()[:8]
    return _CACHE_DIR / f"emb_{key}_{tag}.npy"


def embed_texts(texts: list[str], key: str) -> np.ndarray:
    """Return the (n, 384) normalized embedding matrix for `texts`, cached by key.

    `key` identifies the dataset; the cache also encodes the row count so a
    changed loader invalidates it automatically.
    """
    n = len(texts)
    cache = _cache_path(key, n)
    if cache.exists():
        emb = np.load(cache)
        if emb.shape[0] == n:
            return emb
    model = _get_model()
    # normalize_embeddings=True -> unit vectors, so a dot product is the cosine
    # similarity used by the k-NN retrieval.
    emb = model.encode(
        texts, normalize_embeddings=True, batch_size=128, show_progress_bar=True
    ).astype(np.float32)
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(cache, emb)
    return emb
