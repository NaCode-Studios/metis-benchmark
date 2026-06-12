"""Week-1 EDA: per-dataset summary feeding reports/data_dictionary.md.

Usage: python scripts/eda.py
"""

from __future__ import annotations

import numpy as np

from metis_benchmark.datasets import status
from metis_benchmark.datasets.loaders import EFFORT_UNITS, LOADERS, load


def summarize(key: str) -> dict:
    df = load(key)
    # Keep only valid (positive) efforts: the basis for every statistic below.
    effort = df["effort"].dropna()
    effort = effort[effort > 0]
    out = {
        "rows": len(df),
        "effort_valid": len(effort),
        "effort_unit": EFFORT_UNITS[key],
        # Median and p90 sketch the effort distribution; max flags outliers.
        "effort_median": float(effort.median()),
        "effort_p90": float(effort.quantile(0.9)),
        "effort_max": float(effort.max()),
        # Skew of log(effort): values near 0 support the log-space target
        # z = log(effort) assumed by the protocol.
        "log_skew": float(np.log(effort).skew()),
    }
    # The remaining fields exist only on some datasets; report when present.
    if "date" in df:
        dates = df["date"].dropna()
        # Share of rows with a date decides temporal-split feasibility.
        out["date_coverage"] = f"{len(dates) / len(df):.0%}"
        if len(dates):
            out["date_range"] = f"{dates.min():%Y-%m} .. {dates.max():%Y-%m}"
    if "expert_estimate" in df:
        # Share of rows carrying a human estimate = size of the expert baseline.
        expert = df["expert_estimate"].dropna()
        out["expert_share"] = f"{len(expert) / len(df):.0%}"
    if "size" in df:
        out["size_coverage"] = f"{df['size'].notna().mean():.0%}"
    if "category" in df:
        out["categories"] = int(df["category"].nunique())
    if "text" in df:
        # Median text length indicates how much signal the semantic channel gets.
        out["text_median_chars"] = int(df["text"].str.len().median())
    if "project" in df:
        out["projects"] = int(df["project"].nunique())
    return out


def main() -> None:
    present = status()
    # Iterate all registered loaders, skipping datasets not yet downloaded.
    for key in LOADERS:
        if not present.get(key):
            print(f"## {key}: NOT DOWNLOADED\n")
            continue
        print(f"## {key}")
        for field, value in summarize(key).items():
            print(f"  {field}: {value}")
        print()


if __name__ == "__main__":
    main()
