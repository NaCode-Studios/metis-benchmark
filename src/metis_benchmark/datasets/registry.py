"""Registry of the benchmark datasets (Metis MVP V20, sez. 12).

Track A — project level, validates the tabular channel.
Track B — task level with text, validates the semantic channel.

Raw files live in data/raw/<key>/ and are never committed: licenses differ
per dataset and each must be cited as required by its source. Sources were
verified against the Zenodo API and GitHub on 2026-06-12; scripts/download.py
is the reproducible path to the raw files.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DATA_RAW = Path(__file__).resolve().parents[3] / "data" / "raw"


@dataclass(frozen=True)
class DatasetSource:
    key: str
    track: str  # "A" (project level) or "B" (task level with text)
    name: str
    source: str  # DOI / URL / where to obtain it
    license: str
    in_gate: bool  # counts toward the G0 "2 datasets per track" rule
    notes: str = ""


SOURCES: list[DatasetSource] = [
    # ----- Track A: project level (tabular channel) -----
    DatasetSource(
        key="desharnais",
        track="A",
        name="PROMISE / Desharnais",
        source="github.com/Derek-Jones/Software-estimation-datasets (Desharnais.csv)",
        license="see meta-catalog",
        in_gate=True,
        notes="81 projects; classic literature dataset; smoke-test entry point; "
        "not on Zenodo -> served by the declared reserve meta-catalog",
    ),
    DatasetSource(
        key="cocomo81",
        track="A",
        name="PROMISE / COCOMO81",
        source="DOI 10.5281/zenodo.268424 (coc81-dem.arff)",
        license="CC-BY-4.0",
        in_gate=True,
        notes="63 projects, 1980s; declared-limits dataset",
    ),
    DatasetSource(
        key="china",
        track="A",
        name="PROMISE / China",
        source="DOI 10.5281/zenodo.268446",
        license="CC-BY-4.0",
        in_gate=True,
        notes="499 projects; largest PROMISE set, main Track A dataset",
    ),
    DatasetSource(
        key="kitchenham",
        track="A",
        name="PROMISE / Kitchenham",
        source="DOI 10.5281/zenodo.268457",
        license="CC-BY-4.0",
        in_gate=True,
        notes="145 projects; has dates -> true temporal split",
    ),
    DatasetSource(
        key="maxwell",
        track="A",
        name="PROMISE / Maxwell",
        source="DOI 10.5281/zenodo.268461",
        license="CC-BY-4.0",
        in_gate=True,
    ),
    DatasetSource(
        key="albrecht",
        track="A",
        name="PROMISE / Albrecht",
        source="DOI 10.5281/zenodo.268467",
        license="CC-BY-4.0",
        in_gate=True,
        notes="24 projects; too small for the gate on its own, kept for completeness",
    ),
    DatasetSource(
        key="seera",
        track="A",
        name="SEERA (PROMISE/ACM 2020)",
        source="DOI 10.5281/zenodo.4312777",
        license="CC-BY-4.0",
        in_gate=True,
        notes="~120 real projects, 70+ cost drivers with economic attributes",
    ),
    # ----- Track B: task level with text (semantic channel) -----
    DatasetSource(
        key="deepse",
        track="B",
        name="Deep-SE issue dataset (Choetkiertikul et al. 2019)",
        source="github.com/SOLAR-group/AgileEffortEstimation (Choet_Dataset)",
        license="see replication study repo",
        in_gate=True,
        notes="14 of the original 16 project CSVs (2 withdrawn for GDPR); "
        "fastest Track B entry point",
    ),
    DatasetSource(
        key="josse",
        track="B",
        name="JOSSE (Alhamed & Storer 2022)",
        source="DOI 10.5281/zenodo.7022735",
        license="CC-BY-4.0",
        in_gate=True,
        notes="JIRA tasks (Apache/JBoss/Spring) with real effort; ~19% carry an "
        "expert estimate -> direct model-vs-human comparison",
    ),
    DatasetSource(
        key="sip",
        track="B",
        name="SiP (Jones & Cullum 2019)",
        source="github.com/Derek-Jones/SiP_dataset",
        license="see repo README",
        in_gate=True,
        notes="10,100 task estimates with actuals, 22 developers; calibration "
        "validation: CQR coverage, underestimation bias",
    ),
    DatasetSource(
        key="tawos",
        track="B",
        name="TAWOS (MSR 2022)",
        source="DOI 10.5522/04/21308124 (MySQL dump; repo github.com/SOLAR-group/TAWOS)",
        license="Apache-2.0, research-purposes terms of use",
        in_gate=False,
        notes="~458k issues, 39+ JIRA projects; retrieval/reranking scale only — "
        "subset declared in the protocol, not part of the gate",
    ),
]


def raw_path(key: str) -> Path:
    """Directory where the raw files of a dataset are expected."""
    return DATA_RAW / key


def status() -> dict[str, bool]:
    """Which datasets are present locally (downloaded into data/raw/<key>/)."""
    return {s.key: raw_path(s.key).is_dir() and any(raw_path(s.key).iterdir()) for s in SOURCES}
