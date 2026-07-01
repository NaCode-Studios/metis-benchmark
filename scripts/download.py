"""Download the benchmark datasets into data/raw/<key>/.

Sources were verified against the Zenodo API and GitHub on 2026-06-12; every
Zenodo record is CC-BY 4.0. Raw files are never committed — this script is
the reproducible path to them.

Every download is pinned to a SHA-256 checksum (computed 2026-07-02 from the
exact local artifacts that produced the G0 numbers; the two Zenodo archives
were additionally cross-checked against the record's declared checksum). A
mismatch aborts with an explicit error instead of silently benchmarking
different data: with upstream files on mutable hosts (raw.githubusercontent),
the pin is what makes "reproduce the report" a checkable claim.

Usage:
    python scripts/download.py            # everything except TAWOS
    python scripts/download.py china sip  # only the given keys
"""

from __future__ import annotations

import hashlib
import io
import sys
import urllib.request
import zipfile
from pathlib import Path

# Destination root: data/raw relative to the repository, regardless of cwd.
DATA_RAW = Path(__file__).resolve().parents[1] / "data" / "raw"

# Deep-SE per-project CSV checksums (the 14 SOLAR-group republished files).
_DEEPSE_SHA256 = {
    "APSTUD": "c7af639b52a5a679dbfb29ac65d5c8558c18af45946932cd4e89a73c2ff34172",
    "BAM": "90d64e163896168edad7ad5ed4411fcd0da103cf6d9c0bcde67fa97add2f1db9",
    "CLOV": "8218c05d864d2130e88ae24b6a90f318c5bba3e76692b132dbf51880f6c73736",
    "DM": "5bed6be21916f86aeba375a06e4d2e3466257f5c2be78201e56effe26f0053a8",
    "DURACLOUD": "ca9c2252eea0848cec3724e864ce782f4f1ac16838ebad19128bf67068f306cf",
    "JRESERVER": "2e6f525c0d88ed36985c287e9cdc42bf6800f66be0e4e900854368e98cd5c6aa",
    "MDL": "5aff0df8cf96cdaa4295341c6e35fe20c550f088979ca07cee74bc9b8f3397ce",
    "MESOS": "192814946b6d61e01a2d068cc957eade8402b40a864d6be42032f73aaa7f891e",
    "MULESTUDIO": "e2a6019d1bcce88845667fe28d7c94212c181c030462b9805953490f756411cc",
    "MULE": "a4e7cf46f94ba3ece298c24c49eee41c195326d65a6451f40fcf2f6fbe4759a2",
    "TIMOB": "a1ecb54d8a8a15b9c93f9fbdc5868404719c1df063504321375fdf98106aa9b5",
    "TISTUD": "43f2de728c4251084782cc13bcc339e67e4cbbfc1d6f07be9d2be6b799bbf3b4",
    "USERGRID": "71fabca08e9172cc401ba57ad4dd138dbb17fbe46fcccd883d1452248aed08a6",
    "XD": "edbef829fbfb9dae56085db0804ce3077aa3f44ca65f95f80909750d301abbc5",
}

# key -> list of (url, target filename, sha256 of the downloaded bytes);
# None filename = the download is a zip archive, unzipped in place (the hash
# pins the archive itself, so its extracted contents are pinned transitively).
DOWNLOADS: dict[str, list[tuple[str, str | None, str]]] = {
    # ----- Track A (Zenodo PROMISE mirrors, CC-BY 4.0) -----
    "cocomo81": [
        (
            "https://zenodo.org/api/records/268424/files/coc81-dem.arff/content",
            "coc81-dem.arff",
            "ac058fc65dc66bfe95643e2b3d14c30fef4393be85e7fb18497d93eb049019a1",
        ),
    ],
    "china": [
        (
            "https://zenodo.org/api/records/268446/files/china.arff/content",
            "china.arff",
            "ae9df9c52c5bc5ba03b9732ecc159093f0c500ffa0f3d8260ddd7350f0a22c58",
        ),
    ],
    "kitchenham": [
        (
            "https://zenodo.org/api/records/268457/files/kitchenham.arff/content",
            "kitchenham.arff",
            "0832342b248418cca324cebe77d86d2f90ae223c1119af9360783bdbad0b2ac0",
        ),
    ],
    "maxwell": [
        (
            "https://zenodo.org/api/records/268461/files/maxwell.arff/content",
            "maxwell.arff",
            "13ea83d5cded12ec694f3e1a72e24c55708bf6e28ea810ee88f1d7c8da910fde",
        ),
    ],
    "albrecht": [
        (
            "https://zenodo.org/api/records/268467/files/albrecht.arff/content",
            "albrecht.arff",
            "cfd9713c6a0a45e318fd5345d3f45ac1cfbbf59191d91a645eabd6541890fd18",
        ),
    ],
    "seera": [
        # Single zip containing the Excel/ARFF variants and documentation.
        (
            "https://zenodo.org/api/records/4312777/files/The%20SEERA%20Dataset.zip/content",
            None,
            "4eb4ccd32d3c6d86f0ef6a8feec6ad823a0feae56556aa54b622ff94119dcc75",
        ),
    ],
    # Desharnais is not on Zenodo; the declared reserve meta-catalog
    # (Derek Jones, Software-estimation-datasets) carries it.
    "desharnais": [
        (
            "https://raw.githubusercontent.com/Derek-Jones/Software-estimation-datasets/master/Desharnais.csv",
            "desharnais.csv",
            "38c5d00053555cbe7f3c10c2b026990ea981612ddefd5a7e0fcf8aa52bc4fd64",
        ),
    ],
    # ----- Track B -----
    "sip": [
        # Task metadata (estimates, actuals, categories) ...
        (
            "https://raw.githubusercontent.com/Derek-Jones/SiP_dataset/master/Sip-task-info.csv",
            "Sip-task-info.csv",
            "28621aa8b0ce05c270085a78e96a4d37f1bb39c1a5d260059ff3c197de961a4a",
        ),
        # ... and the estimate/actual dates, paired by the loader.
        (
            "https://raw.githubusercontent.com/Derek-Jones/SiP_dataset/master/est-act-dates.csv",
            "est-act-dates.csv",
            "4d5a5ea633969dd782ab3d6b2b0c6cf2fcc65d9bb1c22f332ac7d9abf5ead1e9",
        ),
    ],
    "josse": [
        # Single zip with the sqlite database and the raw JIRA exports.
        (
            "https://zenodo.org/api/records/7022735/files/JOSSE_Dataset.zip/content",
            None,
            "d6cf0c893055c875c0f3fe2a685ba6d2e858e02e5fd2b7e3c151b980966ed264",
        ),
    ],
    # Deep-SE (Choetkiertikul et al. 2019): 14 of the original 16 project CSVs
    # (2 withdrawn for GDPR), republished by the SOLAR-group replication study
    # "Agile Effort Estimation: Have We Solved the Problem Yet?" (Tawosi et al.).
    "deepse": [
        (
            "https://raw.githubusercontent.com/SOLAR-group/AgileEffortEstimation/main/"
            f"datasets/Choet_Dataset/{name}_deep-se.csv",
            f"{name}_deep-se.csv",
            sha,
        )
        for name, sha in _DEEPSE_SHA256.items()
    ],
    # TAWOS: ~458k issues, MySQL dump at DOI 10.5522/04/21308124 (Apache 2.0,
    # research-purposes terms of use) — census only, subset declared in protocol.
}


def fetch(url: str, sha256: str) -> bytes:
    """GET `url` and verify the payload against its pinned SHA-256.

    Verification happens BEFORE anything touches the disk: a corrupted or
    upstream-modified file must fail loudly here, not surface later as a
    silently different benchmark result.
    """
    # Plain GET with an identifying User-Agent (Zenodo rejects anonymous bots).
    req = urllib.request.Request(url, headers={"User-Agent": "metis-benchmark/0.1"})
    with urllib.request.urlopen(req) as resp:  # noqa: S310 - pinned https sources
        data = resp.read()
    digest = hashlib.sha256(data).hexdigest()
    if digest != sha256:
        raise RuntimeError(
            f"checksum mismatch for {url}\n"
            f"  expected sha256 {sha256}\n"
            f"  got      sha256 {digest}\n"
            "The upstream file changed or the download was corrupted; refusing "
            "to save it. If upstream legitimately republished the dataset, "
            "re-verify its provenance and update the pin in scripts/download.py."
        )
    return data


def safe_extractall(archive: zipfile.ZipFile, target_dir: Path) -> None:
    """extractall with a zip-slip guard: no member may escape `target_dir`.

    A hostile archive can carry member names like '../../.ssh/authorized_keys'
    or absolute paths; resolving each destination and requiring it to stay
    under the target directory blocks the traversal before extraction starts
    (all-or-nothing: one bad member rejects the whole archive).
    """
    root = target_dir.resolve()
    for member in archive.infolist():
        dest = (root / member.filename).resolve()
        if not dest.is_relative_to(root):
            raise RuntimeError(
                f"zip-slip blocked: archive member {member.filename!r} would "
                f"extract outside {root}; refusing the whole archive"
            )
    archive.extractall(root)


def download_key(key: str) -> None:
    # Each dataset gets its own directory under data/raw.
    target_dir = DATA_RAW / key
    target_dir.mkdir(parents=True, exist_ok=True)
    for url, filename, sha256 in DOWNLOADS[key]:
        data = fetch(url, sha256)
        if filename is None:
            # Archive entry: extract in place instead of saving the zip alone.
            safe_extractall(zipfile.ZipFile(io.BytesIO(data)), target_dir)
            print(f"{key}: unzipped {url.split('/')[-2]} -> {target_dir} (sha256 ok)")
        else:
            (target_dir / filename).write_bytes(data)
            print(f"{key}: {filename} ({len(data)} bytes, sha256 ok)")


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
