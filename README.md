# metis-benchmark

Public, reproducible backtest of the **Metis** software effort-estimation
engine on open datasets. This repository is the Phase 0 gate (G0) of the
Metis MVP by [NaCode Studios](https://nacodestudios.it): if the engine does
not beat the standard baselines — and the human expert, where datasets record
one — the product does not get built.

> 📄 **Read the report: [Can a model estimate software effort from public data?](reports/REPORT.md)**
> A pre-registered blind test across nine datasets — and an honest negative result.

## What is being tested

Two tracks, mirroring the engine's two channels:

| Track | Level | Channel | Datasets |
|-------|-------|---------|----------|
| A | Project | Tabular regression (GP / gradient boosting, quantile + CQR) | PROMISE (Desharnais, COCOMO81, China, Kitchenham, Maxwell, Albrecht), SEERA |
| B | Task (with text) | Semantic similarity (embeddings, k-NN, reranking) | Deep-SE, JOSSE, SiP; TAWOS for retrieval scale |

Metrics: **PRED(25)**, **MdAPE**, **empirical coverage** of conformalized
quantile intervals (CQR). Temporal splits only — never random. Full rules in
[reports/protocol.md](reports/protocol.md), frozen before experiments run.

Mandatory baselines the engine must beat:

1. median effort by category;
2. linear regression of log(effort) on log(size);
3. the human expert estimate recorded in the dataset (JOSSE, SiP).

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[models,dev]" -c constraints-g0.txt
make test
make status        # which datasets are present in data/raw/ (status-a / status-b per track)
make reproduce-g0  # full G0 chain (REPORT.md §8): download → both channels → ceilings → tests
```

`constraints-g0.txt` pins the exact package versions the G0 numbers were
produced with; install with `-c` to reproduce them bit-for-bit.

Datasets are **not** committed: each has its own license and citation
requirements. Sources are listed in
[src/metis_benchmark/datasets/registry.py](src/metis_benchmark/datasets/registry.py).

## Status

- [x] Metrics, baselines, temporal split (tested)
- [x] Week 1 — dataset download (10/10 gate datasets), EDA,
      [data dictionary](reports/data_dictionary.md), baseline smoke test,
      protocol frozen
- [x] Week 2 — Track A GP + GBM quantile + CQR; rolling-origin estimator (v1.2);
      [partial Track A verdict](reports/results/track_a_G0_verdict.md)
- [x] Week 3 — Track A feature expansion (cocomo81, seera), mean-function GP,
      hierarchical pooling, honest feasibility ceiling →
      **[Track A final verdict](reports/results/track_a_W3_verdict.md): not
      passing, 55% proven unreachable on public tabular data; closed
      best-effort; proceeding to Track B**
- [x] Track B — semantic channel (JOSSE, SiP): bge embeddings + k-NN
      Nadaraya-Watson + cross-encoder + learned-regressor ceiling →
      **[Track B verdict](reports/results/track_b_semantic.md): not passing —
      text does not predict logged effort cross-project (ceiling 15–20%)**
- [x] **[Full G0 verdict](reports/results/G0_full_verdict.md): NO-GO on the
      public gate — both channels proven below 55%; intervals honest;
      expert-assist + proprietary-data pilot are the evidence-based next steps**

## Citations

This benchmark builds on public datasets by their respective authors —
PROMISE repository (CC-BY, Zenodo mirrors), SEERA (PROMISE/ACM 2020),
JOSSE (Alhamed & Storer 2022), TAWOS (MSR 2022), Deep-SE
(Choetkiertikul et al. 2019), SiP (Jones & Cullum 2019) — and on
Conformalized Quantile Regression (Romano, Patterson, Candès 2019) via the
MAPIE library. Cite the original sources when reusing the data.

## License

Code: [MIT](LICENSE). Datasets: see their respective licenses.
