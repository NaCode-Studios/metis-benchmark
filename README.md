# metis-benchmark

Public, reproducible backtest of the **Metis** software effort-estimation
engine on open datasets. This repository is the Phase 0 gate (G0) of the
Metis MVP by [NaCode Studios](https://nacodestudios.it): if the engine does
not beat the standard baselines — and the human expert, where datasets record
one — the product does not get built.

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
pip install -e ".[models,dev]"
make test
make status        # which datasets are present in data/raw/
```

Datasets are **not** committed: each has its own license and citation
requirements. Sources are listed in
[src/metis_benchmark/datasets/registry.py](src/metis_benchmark/datasets/registry.py).

## Status

- [x] Metrics, baselines, temporal split (tested)
- [ ] Week 1 — dataset download, EDA, data dictionary, protocol freeze
- [ ] Week 2 — Track A (GP + GBM quantile + CQR), Track B (embeddings + k-NN)
- [ ] Week 3 — full backtest, calibration plots, cold-start curve, model vs expert
- [ ] Week 4 — public technical report, G0 decision

## Citations

This benchmark builds on public datasets by their respective authors —
PROMISE repository (CC-BY, Zenodo mirrors), SEERA (PROMISE/ACM 2020),
JOSSE (Alhamed & Storer 2022), TAWOS (MSR 2022), Deep-SE
(Choetkiertikul et al. 2019), SiP (Jones & Cullum 2019) — and on
Conformalized Quantile Regression (Romano, Patterson, Candès 2019) via the
MAPIE library. Cite the original sources when reusing the data.

## License

Code: [MIT](LICENSE). Datasets: see their respective licenses.
