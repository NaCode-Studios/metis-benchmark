"""Backtest orchestration entry point.

Week 1 scope: dataset availability report. The per-track experiment
runners land in week 2-3 once the protocol (reports/protocol.md) is frozen.

Usage:
    python -m metis_benchmark.backtest          # availability of all datasets
    python -m metis_benchmark.backtest --track A
"""

from __future__ import annotations

import argparse

from metis_benchmark.datasets import SOURCES, status


def main() -> None:
    parser = argparse.ArgumentParser(description="Metis benchmark backtest")
    parser.add_argument("--track", choices=["A", "B"], help="restrict to one track")
    args = parser.parse_args()

    # Check which datasets have files in data/raw/<key>/.
    present = status()

    # One line per dataset: track, gate membership, local availability, source.
    print(f"{'dataset':<14} {'track':<6} {'gate':<6} {'local':<6} source")
    for s in SOURCES:
        if args.track and s.track != args.track:
            continue
        print(
            f"{s.key:<14} {s.track:<6} {'yes' if s.in_gate else 'no':<6} "
            f"{'yes' if present[s.key] else 'NO':<6} {s.source}"
        )

    # Point the user at the download path for anything still missing.
    missing = [s.key for s in SOURCES if not present[s.key] and (not args.track or s.track == args.track)]
    if missing:
        print(f"\nmissing locally: {', '.join(missing)}")
        print("download into data/raw/<key>/ — sources above, licenses in registry.py")


if __name__ == "__main__":
    main()
