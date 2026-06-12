"""Download the benchmark datasets into data/raw/<key>/.

Sources were verified against the Zenodo API and GitHub on 2026-06-12; every
Zenodo record is CC-BY 4.0. Raw files are never committed — this script is
the reproducible path to them.

Usage:
    python scripts/download.py            # everything except TAWOS
    python scripts/download.py china sip  # only the given keys
"""

from __future__ import annotations

import io
import sys
import urllib.request
import zipfile
from pathlib import Path

# Destination root: data/raw relative to the repository, regardless of cwd.
DATA_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

# key -> list of (url, target filename); None filename = unzip archive in place.
DOWNLOADS: dict[str, list[tuple[str, str | None]]] = {
    # ----- Track A (Zenodo PROMISE mirrors, CC-BY 4.0) -----
    "cocomo81": [
        ("https://zenodo.org/api/records/268424/files/coc81-dem.arff/content", "coc81-dem.arff"),
    ],
    "china": [
        ("https://zenodo.org/api/records/268446/files/china.arff/content", "china.arff"),
    ],
    "kitchenham": [
        ("https://zenodo.org/api/records/268457/files/kitchenham.arff/content", "kitchenham.arff"),
    ],
    "maxwell": [
        ("https://zenodo.org/api/records/268461/files/maxwell.arff/content", "maxwell.arff"),
    ],
    "albrecht": [
        ("https://zenodo.org/api/records/268467/files/albrecht.arff/content", "albrecht.arff"),
    ],
    "seera": [
        # Single zip containing the Excel/ARFF variants and documentation.
        (
            "https://zenodo.org/api/records/4312777/files/The%20SEERA%20Dataset.zip/content",
            None,
        ),
    ],
    # Desharnais is not on Zenodo; the declared reserve meta-catalog
    # (Derek Jones, Software-estimation-datasets) carries it.
    "desharnais": [
        (
            "https://raw.githubusercontent.com/Derek-Jones/Software-estimation-datasets/master/Desharnais.csv",
            "desharnais.csv",
        ),
    ],
    # ----- Track B -----
    "sip": [
        # Task metadata (estimates, actuals, categories) ...
        (
            "https://raw.githubusercontent.com/Derek-Jones/SiP_dataset/master/Sip-task-info.csv",
            "Sip-task-info.csv",
        ),
        # ... and the estimate/actual dates, paired by the loader.
        (
            "https://raw.githubusercontent.com/Derek-Jones/SiP_dataset/master/est-act-dates.csv",
            "est-act-dates.csv",
        ),
    ],
    "josse": [
        # Single zip with the sqlite database and the raw JIRA exports.
        ("https://zenodo.org/api/records/7022735/files/JOSSE_Dataset.zip/content", None),
    ],
    # Deep-SE (Choetkiertikul et al. 2019): 14 of the original 16 project CSVs
    # (2 withdrawn for GDPR), republished by the SOLAR-group replication study
    # "Agile Effort Estimation: Have We Solved the Problem Yet?" (Tawosi et al.).
    "deepse": [
        (
            "https://raw.githubusercontent.com/SOLAR-group/AgileEffortEstimation/main/"
            f"datasets/Choet_Dataset/{name}_deep-se.csv",
            f"{name}_deep-se.csv",
        )
        for name in [
            "APSTUD", "BAM", "CLOV", "DM", "DURACLOUD", "JRESERVER", "MDL",
            "MESOS", "MULESTUDIO", "MULE", "TIMOB", "TISTUD", "USERGRID", "XD",
        ]
    ],
    # TAWOS: ~458k issues, MySQL dump at DOI 10.5522/04/21308124 (Apache 2.0,
    # research-purposes terms of use) — census only, subset declared in protocol.
}


def fetch(url: str) -> bytes:
    # Plain GET with an identifying User-Agent (Zenodo rejects anonymous bots).
    req = urllib.request.Request(url, headers={"User-Agent": "metis-benchmark/0.1"})
    with urllib.request.urlopen(req) as resp:  # noqa: S310 - pinned https sources
        return resp.read()


def download_key(key: str) -> None:
    # Each dataset gets its own directory under data/raw.
    target_dir = DATA_RAW / key
    target_dir.mkdir(parents=True, exist_ok=True)
    for url, filename in DOWNLOADS[key]:
        data = fetch(url)
        if filename is None:
            # Archive entry: extract in place instead of saving the zip alone.
            zipfile.ZipFile(io.BytesIO(data)).extractall(target_dir)
            print(f"{key}: unzipped {url.split('/')[-2]} -> {target_dir}")
        else:
            (target_dir / filename).write_bytes(data)
            print(f"{key}: {filename} ({len(data)} bytes)")


def main() -> None:
    # No arguments = download everything; otherwise only the requested keys.
    keys = sys.argv[1:] or list(DOWNLOADS)
    unknown = [k for k in keys if k not in DOWNLOADS]
    if unknown:
        sys.exit(f"unknown dataset keys: {unknown}; available: {list(DOWNLOADS)}")
    for key in keys:
        download_key(key)


if __name__ == "__main__":
    main()
