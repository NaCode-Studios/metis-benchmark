"""Loader tests — run only for datasets present in data/raw/ (see scripts/download.py)."""

import numpy as np
import pytest

from metis_benchmark.datasets import status
from metis_benchmark.datasets.loaders import EFFORT_UNITS, LOADERS, load

PRESENT = status()

# minimum expected rows per dataset (sanity floor, not exact counts)
MIN_ROWS = {
    "desharnais": 75,
    "cocomo81": 60,
    "china": 490,
    "kitchenham": 140,
    "maxwell": 55,
    "albrecht": 20,
    "seera": 100,
    "deepse": 20000,
    "josse": 20000,
    "sip": 9000,
}

TRACK_B = {"deepse", "josse", "sip"}
HAS_EXPERT = {"kitchenham", "seera", "josse", "sip"}
HAS_DATE = {"desharnais", "kitchenham", "maxwell", "seera", "sip"}


def needs(key):
    return pytest.mark.skipif(not PRESENT.get(key), reason=f"{key} not downloaded")


@pytest.mark.parametrize("key", [pytest.param(k, marks=needs(k)) for k in LOADERS])
def test_loader_canonical_schema(key):
    df = load(key)
    assert len(df) >= MIN_ROWS[key], f"{key}: only {len(df)} rows"
    assert key in EFFORT_UNITS

    effort = df["effort"].dropna()
    assert len(effort) > 0
    assert (effort > 0).mean() > 0.95, f"{key}: too many non-positive efforts"

    if key in TRACK_B:
        assert df["text"].str.len().gt(0).mean() > 0.95, f"{key}: empty texts"
        assert df["project"].notna().all()

    if key in HAS_EXPERT:
        expert = df["expert_estimate"].dropna()
        assert len(expert) > 0, f"{key}: expert estimates expected"
        assert (expert > 0).all()

    if key in HAS_DATE:
        assert df["date"].notna().mean() > 0.9, f"{key}: dates expected"


@needs("josse")
def test_josse_expert_subset_is_minority():
    df = load("josse")
    share = df["expert_estimate"].notna().mean()
    # paper reports ~19% of tasks carry an expert estimate
    assert 0.05 < share < 0.5, f"unexpected expert share: {share:.2%}"
