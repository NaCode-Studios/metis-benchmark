"""Multilingual sentence embeddings (cached, deterministic).

The Track B channel used an English model (bge-small-en); some inputs are
non-English (e.g. Italian), so a *multilingual* embedder is needed. We use
`intfloat/multilingual-e5-large` (1024-d, ~100 languages), wrapped with the two
properties the protocol demands:

  * **deterministic** — the model is never trained, so embeddings leak nothing
    about effort and re-running gives identical vectors; and
  * **cached** — embedding text is the slow step, so the matrix is memoized to
    disk keyed by *content* (model + text hash), surviving row reordering.

To keep the consuming pipeline (and its smoke test) runnable with no 2 GB download,
the embedder is an interface: `MultilingualEmbedder` loads the real e5 model
lazily, while `HashingEmbedder` is a deterministic offline stand-in for tests.
Both return an (n, dim) float32 matrix of unit-normalized rows, so a dot product
is the cosine similarity the composite GP kernel consumes.
"""

from __future__ import annotations

# hashlib gives a stable content key for the disk cache and the offline embedder.
import hashlib
# Protocols let callers type against "anything with .encode()" — real or fake.
from typing import Protocol

import numpy as np

# Where cached embedding matrices live (shared with the Track B cache dir).
from pathlib import Path

_CACHE_DIR = Path(__file__).resolve().parents[3] / "data" / "processed"
# The default multilingual model.
DEFAULT_MODEL = "intfloat/multilingual-e5-large"


def _l2_normalize(matrix: np.ndarray) -> np.ndarray:
    """Scale each row to unit length so a dot product equals cosine similarity."""
    # Row norms, clamped away from zero so an all-zero row cannot divide by 0.
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)
    return (matrix / norms).astype(np.float32)


class Embedder(Protocol):
    """The minimal surface a caller needs: text -> (n, dim) unit-norm matrix."""

    # The embedding width, so the composite GP knows where the embedding block ends.
    dim: int

    def encode(self, texts: list[str]) -> np.ndarray:  # pragma: no cover - interface
        ...


class MultilingualEmbedder:
    """Lazy, disk-cached wrapper around multilingual-e5-large (1024-d).

    e5 models expect each input prefixed with a task tag; for symmetric
    requirement-to-requirement similarity the e5 card recommends the "query: "
    prefix on both sides, which is the default here.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL, prefix: str = "query: ") -> None:
        # Store config; the heavy model itself is loaded only on first encode().
        self.model_name = model_name
        self.prefix = prefix
        # multilingual-e5-large is 1024-dimensional; fixed by the model card.
        self.dim = 1024
        # Singleton handle, populated lazily.
        self._model = None

    def _get_model(self):
        # Import + load only when embeddings are actually needed (torch is heavy).
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _cache_path(self, texts: list[str]) -> Path:
        # Content key: model + prefix + every text, so any change invalidates it.
        digest = hashlib.md5(
            ("\n".join([self.model_name, self.prefix, *texts])).encode("utf-8")
        ).hexdigest()[:16]
        return _CACHE_DIR / f"mlemb_{digest}.npy"

    def encode(self, texts: list[str]) -> np.ndarray:
        # Serve from disk if this exact text set was embedded before.
        cache = self._cache_path(texts)
        if cache.exists():
            return np.load(cache)
        # Apply the e5 prefix, embed, normalize to unit rows.
        prefixed = [self.prefix + t for t in texts]
        model = self._get_model()
        emb = _l2_normalize(
            np.asarray(model.encode(prefixed, batch_size=64, show_progress_bar=False))
        )
        # Persist for the next run (deterministic, so the cache is always valid).
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        np.save(cache, emb)
        return emb


class HashingEmbedder:
    """Deterministic offline embedder: hash(text) -> seeded Gaussian -> unit vec.

    Carries no semantic signal (by design) — it exists so the consuming pipeline and
    its smoke test run with zero downloads and identical output on every machine.
    A real run swaps in MultilingualEmbedder; the pipeline code is unchanged.
    """

    def __init__(self, dim: int = 64) -> None:
        # A small width keeps the synthetic smoke test fast; any dim works.
        self.dim = dim

    def encode(self, texts: list[str]) -> np.ndarray:
        rows = np.empty((len(texts), self.dim), dtype=np.float32)
        for i, text in enumerate(texts):
            # Hash the text to a stable 32-bit seed, then draw a fixed vector.
            seed = int(hashlib.md5(text.encode("utf-8")).hexdigest()[:8], 16)
            rows[i] = np.random.default_rng(seed).standard_normal(self.dim)
        # Unit-normalize so it behaves like the real embedder downstream.
        return _l2_normalize(rows)
