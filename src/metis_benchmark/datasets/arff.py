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

# Parses "@attribute <name> <type>" header lines. The name may be bare,
# single-quoted or double-quoted (SEERA quotes names containing spaces).
_ATTRIBUTE = re.compile(
    r"@attribute\s+(?:'(?P<quoted>[^']+)'|\"(?P<dquoted>[^\"]+)\"|(?P<bare>\S+))\s+(?P<type>.+)",
    re.IGNORECASE,
)
# Tokens treated as missing values in the data section.
_MISSING = {"?", "N/A", ""}


def read_arff(path: str | Path, trailing_extra: tuple[str, ...] = ()) -> pd.DataFrame:
    """Read an ARFF file.

    `trailing_extra` names data columns present in the rows but missing from
    the attribute declarations (coc81-dem.arff ships an undeclared trailing
    month count).
    """
    names: list[str] = []  # attribute names, in declaration order
    numeric: list[bool] = []  # whether each attribute should become a float column
    data_lines: list[str] = []  # raw rows of the @data section
    in_data = False

    # Single pass over the file: collect attribute declarations until @data,
    # then accumulate raw data rows.
    for raw_line in Path(path).read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        # Skip blanks and ARFF comments.
        if not line or line.startswith("%"):
            continue
        if in_data:
            data_lines.append(line)
            continue
        lowered = line.lower()
        if lowered.startswith("@data"):
            # Header is finished; everything that follows is data.
            in_data = True
        elif lowered.startswith("@attribute"):
            match = _ATTRIBUTE.match(line)
            if not match:
                raise ValueError(f"unparsable attribute line in {path}: {line!r}")
            # Whichever quoting style matched provides the attribute name.
            name = match["quoted"] or match["dquoted"] or match["bare"]
            names.append(name.strip())
            # Only declared numeric types get converted to floats later;
            # nominal/string/date attributes stay as strings for the loaders.
            numeric.append(match["type"].strip().lower().startswith(("numeric", "real", "integer")))

    # Register the undeclared trailing columns (numeric by convention).
    names += list(trailing_extra)
    numeric += [True] * len(trailing_extra)

    # PROMISE files come in two dialects: comma-separated and space-separated.
    # Detect from the first data row and tokenize accordingly.
    if data_lines and "," not in data_lines[0]:
        records = [line.split() for line in data_lines]
    else:
        records = list(csv.reader(data_lines, quotechar="'"))

    rows = []
    for record in records:
        # Every row must match the declared schema; a mismatch means the file
        # has a dialect we do not handle, so fail instead of misaligning columns.
        if len(record) != len(names):
            raise ValueError(f"row with {len(record)} fields, expected {len(names)} in {path}")
        # Map the missing-value markers to None, keep everything else as text.
        rows.append([None if v.strip() in _MISSING else v.strip() for v in record])

    df = pd.DataFrame(rows, columns=names)
    # Convert declared-numeric columns; unparsable values become NaN rather
    # than raising, since PROMISE files occasionally mix markers into them.
    for name, is_numeric in zip(names, numeric):
        if is_numeric:
            df[name] = pd.to_numeric(df[name], errors="coerce")
    # Normalize remaining Nones (string columns) to NaN for a uniform missing marker.
    return df.replace({None: np.nan})
