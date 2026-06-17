"""Track B honest feasibility ceiling + stop rule (protocol v2.0).

Best out-of-sample PRED(25) over a zoo on the gate datasets (JOSSE, SiP):
  - k-NN Nadaraya-Watson over bge embeddings (k/τ test-blind selected)
  - a LEARNED regressor on the embeddings (GBM, RF) — can a model find structure
    retrieval misses?
  - TF-IDF k-NN (lexical reference)
plus the text-blind global-median floor and the human expert for context.

Pre-registered stop rule: if the honest ceiling is below 55% on both gate
datasets, 55% is unreachable on this data and Track B closes as demonstrated.

Usage: python scripts/run_ceiling_b.py
"""

from __future__ import annotations

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from xgboost import XGBRegressor

from metis_benchmark.datasets.loaders import load
from metis_benchmark.evaluation import bootstrap_metric_ci, pred_at
from metis_benchmark.track_b.experiment import knn_oof, load_emb_dataset

GATE = ["sip", "josse"]
THRESHOLD = 0.55


def emb_regressor_oof(data, make_model) -> np.ndarray:
    """Out-of-fold predictions of a learned regressor on raw embeddings."""
    n = len(data.y)
    pred = np.full(n, np.nan)
    for f in data.folds:
        idx = np.concatenate([f.train, f.cal])
        m = make_model()
        m.fit(data.emb[idx], np.log(data.y[idx]))  # raw embeddings (have signs)
        pred[f.test] = np.exp(m.predict(data.emb[f.test]))
    return pred


def tfidf_knn_oof(data, key, k: int = 10, tau: float = 0.1) -> np.ndarray:
    """Out-of-fold TF-IDF cosine k-NN Nadaraya-Watson (lexical reference).

    Uses sklearn NearestNeighbors(metric=cosine) on the sparse TF-IDF matrix,
    which never materializes the full dense similarity matrix (the previous
    dense approach blew up on josse).
    """
    df = load(key).reset_index(drop=True)
    keep = df["effort"].notna() & (df["effort"] > 0) & df["text"].notna()
    if data.split == "rolling":
        keep &= df["date"].notna()
    texts = df.loc[keep, "text"].fillna("").astype(str).to_numpy()
    n = len(data.y)
    pred = np.full(n, np.nan)
    for f in data.folds:
        idx = np.concatenate([f.train, f.cal])
        vec = TfidfVectorizer(max_features=4000, stop_words="english")
        Xtr = vec.fit_transform(texts[idx])
        Xte = vec.transform(texts[f.test])
        kk = min(k, Xtr.shape[0])
        nn = NearestNeighbors(n_neighbors=kk, metric="cosine").fit(Xtr)
        dist, nbr = nn.kneighbors(Xte)  # cosine distance = 1 - cosine sim
        s = 1.0 - dist  # back to similarity
        logy = np.log(data.y[idx])[nbr]
        w = np.exp((s - 1.0) / tau)
        wsum = w.sum(1)
        pred[f.test] = np.exp((w * logy).sum(1) / np.where(wsum > 0, wsum, 1.0))
    return pred


def main() -> None:
    print("Track B honest feasibility ceiling (gate datasets)\n")
    below = []
    for key in GATE:
        data = load_emb_dataset(key)
        tested, knn_pred, _ = knn_oof(data)
        gbm = emb_regressor_oof(data, lambda: XGBRegressor(
            n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8,
            colsample_bytree=0.8, random_state=0))
        rf = emb_regressor_oof(data, lambda: RandomForestRegressor(n_estimators=200, n_jobs=-1, random_state=0))
        tfidf = tfidf_knn_oof(data, key)
        gmed = np.full(len(data.y), np.nan)
        for f in data.folds:
            gmed[f.test] = np.median(data.y[np.concatenate([f.train, f.cal])])

        models = {"k-NN-NW (bge)": knn_pred, "emb->GBM": gbm, "emb->RF": rf,
                  "TF-IDF k-NN": tfidf, "global-median (floor)": gmed}
        print(f"== {key} ({data.split}, n_test={tested.sum()}) ==")
        best_p = -1.0
        for name, pred in models.items():
            m = tested & ~np.isnan(pred)
            ci = bootstrap_metric_ci(data.y[m], pred[m], pred_at, seed=0)
            if name != "global-median (floor)":
                best_p = max(best_p, ci.point)
            print(f"  {name:<22} PRED(25)={ci}")
        flag = "<55%" if best_p < THRESHOLD else ">=55%"
        if best_p < THRESHOLD:
            below.append(key)
        print(f"  honest ceiling = {best_p:.1%}  {flag}\n")

    print(f"Gate datasets with ceiling < 55%: {len(below)}/{len(GATE)} ({', '.join(below)})")
    if len(below) == len(GATE):
        print("STOP RULE TRIGGERED (pre-registered): 55% unreachable on both gate datasets.")
        print("Track B outcome: semantic channel does not meet the gate on public logged-time data.")
    else:
        print("Stop rule NOT triggered.")


if __name__ == "__main__":
    main()
