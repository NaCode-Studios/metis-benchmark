"""Pre-compute and cache the Track B embeddings (slow step, run once).

Usage: python scripts/embed_track_b.py
"""

from __future__ import annotations

import time

from metis_benchmark.datasets.loaders import load
from metis_benchmark.track_b.embeddings import embed_texts

DATASETS = ["deepse", "sip", "josse"]


def main() -> None:
    for key in DATASETS:
        df = load(key)
        texts = df["text"].fillna("").astype(str).tolist()
        t0 = time.time()
        emb = embed_texts(texts, key)
        print(f"{key:<8} n={len(texts):6d}  emb={emb.shape}  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
