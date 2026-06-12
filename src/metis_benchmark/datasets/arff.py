"""Minimal ARFF reader for the PROMISE/SEERA files.

Supports the subset of ARFF these datasets use: numeric, string, nominal and
date attributes, dense CSV data section, '?' (and SEERA's 'N/A') as missing.
Kept dependency-free on purpose: scipy's reader chokes on the date attributes
of kitchenham.arff.
"""

from __future__ import annotations

import csv
import re
from pathlib import Path

import numpy as np
import pandas as pd

_ATTRIBUTE = re.compile(
    r"@attribute\s+(?:'(?P<quoted>[^']+)'|\"(?P<dquoted>[^\"]+)\"|(?P<bare>\S+))\s+(?P<type>.+)",
    re.IGNORECASE,
)
_MISSING = {"?", "N/A", ""}


def read_arff(path: str | Path, trailing_extra: tuple[str, ...] = ()) -> pd.DataFrame:
    """Read an ARFF file.

    `trailing_extra` names data columns present in the rows but missing from
    the attribute declarations (coc81-dem.arff ships an undeclared trailing
    month count).
    """
    names: list[str] = []
    numeric: list[bool] = []
    data_lines: list[str] = []
    in_data = False

    for raw_line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("%"):
            continue
        if in_data:
            data_lines.append(line)
            continue
        lowered = line.lower()
        if lowered.startswith("@data"):
            in_data = True
        elif lowered.startswith("@attribute"):
            match = _ATTRIBUTE.match(line)
            if not match:
                raise ValueError(f"unparsable attribute line in {path}: {line!r}")
            name = match["quoted"] or match["dquoted"] or match["bare"]
            names.append(name.strip())
            numeric.append(match["type"].strip().lower().startswith(("numeric", "real", "integer")))

    names += list(trailing_extra)
    numeric += [True] * len(trailing_extra)

    # PROMISE files come in two dialects: comma-separated and space-separated
    if data_lines and "," not in data_lines[0]:
        records = [line.split() for line in data_lines]
    else:
        records = list(csv.reader(data_lines, quotechar="'"))

    rows = []
    for record in records:
        if len(record) != len(names):
            raise ValueError(f"row with {len(record)} fields, expected {len(names)} in {path}")
        rows.append([None if v.strip() in _MISSING else v.strip() for v in record])

    df = pd.DataFrame(rows, columns=names)
    for name, is_numeric in zip(names, numeric):
        if is_numeric:
            df[name] = pd.to_numeric(df[name], errors="coerce")
    return df.replace({None: np.nan})
