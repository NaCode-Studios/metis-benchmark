"""Integrity tests for the dataset downloader (checksum pin + zip-slip guard).

scripts/download.py is the only path from the network to data/raw, so its two
defenses are load-bearing for reproducibility: the SHA-256 pin (a changed
upstream file must abort, never silently benchmark different data) and the
extraction guard (a hostile archive must not write outside its dataset dir).
The script is not a package module, so it is loaded here by file path.
"""

import hashlib
import importlib.util
import io
import zipfile
from pathlib import Path

import pytest

# Load scripts/download.py as a module: scripts/ is intentionally not a
# package (they are runners), so a plain import cannot reach it.
_PATH = Path(__file__).resolve().parents[1] / "scripts" / "download.py"
_SPEC = importlib.util.spec_from_file_location("download_script", _PATH)
download = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(download)


class _FakeResponse:
    """Minimal stand-in for urlopen's response: a context manager with read()."""

    def __init__(self, payload: bytes) -> None:
        self._payload = payload

    def read(self) -> bytes:
        return self._payload

    def __enter__(self) -> "_FakeResponse":
        return self

    def __exit__(self, *exc) -> None:
        return None


def test_fetch_accepts_matching_checksum(monkeypatch):
    payload = b"benchmark bytes"
    monkeypatch.setattr(
        download.urllib.request, "urlopen", lambda req: _FakeResponse(payload)
    )
    # The pinned hash matches the payload -> the bytes come through untouched.
    ok = hashlib.sha256(payload).hexdigest()
    assert download.fetch("https://example.org/x", ok) == payload


def test_fetch_rejects_checksum_mismatch(monkeypatch):
    monkeypatch.setattr(
        download.urllib.request, "urlopen", lambda req: _FakeResponse(b"tampered")
    )
    # A wrong pin must abort with an explicit, actionable error — the payload
    # must never be returned (and therefore never written to disk).
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        download.fetch("https://example.org/x", "0" * 64)


def _zip_with(names_to_bytes: dict[str, bytes]) -> zipfile.ZipFile:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in names_to_bytes.items():
            zf.writestr(name, data)
    return zipfile.ZipFile(io.BytesIO(buf.getvalue()))


def test_safe_extractall_extracts_benign_archive(tmp_path):
    archive = _zip_with({"a.txt": b"A", "sub/b.txt": b"B"})
    download.safe_extractall(archive, tmp_path)
    assert (tmp_path / "a.txt").read_bytes() == b"A"
    assert (tmp_path / "sub" / "b.txt").read_bytes() == b"B"


def test_safe_extractall_blocks_zip_slip(tmp_path):
    # '../evil.txt' resolves to tmp_path's PARENT: the classic zip-slip escape.
    target = tmp_path / "dataset"
    target.mkdir()
    archive = _zip_with({"ok.txt": b"fine", "../evil.txt": b"escape"})
    with pytest.raises(RuntimeError, match="zip-slip"):
        download.safe_extractall(archive, target)
    # All-or-nothing: nothing may have been written, in or out of the target.
    assert not (tmp_path / "evil.txt").exists()
    assert not (target / "ok.txt").exists()
