"""Stream the TAWOS `Issue` table out of the MySQL dump, without a MySQL server.

TAWOS ships as a 4.3 GB mysqldump (638 MB zipped) and its README says to load it into
MySQL 8. That is a service to install and a database to populate for eleven columns, so
this reads the dump as a stream instead: find the `INSERT INTO \\`Issue\\`` statements,
walk the tuples with a scanner that respects SQL quoting and backslash escapes (issue
descriptions contain `),(` and quotes of their own, so a regex or a naive split silently
corrupts rows), and write only the columns the experiment needs.

The column order comes from the dump's own `CREATE TABLE \\`Issue\\`` and is verified
against it at run time — a schema change upstream must fail loudly, not shift every
value one place to the left.

    python scripts/extract_tawos.py     # data/raw/tawos/issues.csv

Dataset: TAWOS (MSR 2022), DOI 10.5522/04/21308124, Apache-2.0, research use only.
"""

from __future__ import annotations

import csv
import io
import sys
import zipfile
from pathlib import Path

RAW = Path(__file__).resolve().parents[1] / "data" / "raw" / "tawos"
ZIP = RAW / "TAWOS.sql.zip"
OUT = RAW / "issues.csv"

# Columns to keep, by name; positions are resolved from the dump's own DDL.
WANTED = (
    "Issue_Key", "Type", "Priority", "Status", "Resolution",
    "Creation_Date", "Estimation_Date", "Resolution_Date",
    "Story_Point", "Timespent", "Total_Effort_Minutes",
    "Reporter_ID", "Assignee_ID", "Project_ID", "Sprint_ID",
)
INSERT_PREFIX = "INSERT INTO `Issue` VALUES "


def _issue_columns(stream) -> list[str]:
    """Column order of `Issue`, read from the dump's CREATE TABLE."""
    columns: list[str] = []
    capturing = False
    for line in stream:
        if line.startswith("CREATE TABLE `Issue`"):
            capturing = True
            continue
        if capturing:
            stripped = line.strip()
            if stripped.startswith("`"):
                columns.append(stripped.split("`")[1])
            elif stripped.startswith(("PRIMARY KEY", "KEY", "CONSTRAINT", ") ENGINE")):
                break
    if not columns:
        raise SystemExit("no CREATE TABLE `Issue` found in the dump")
    missing = [c for c in WANTED if c not in columns]
    if missing:
        raise SystemExit(f"the dump's Issue table no longer has {missing}; check the schema")
    return columns


def _tuples(values: str):
    """Yield the top-level `(...)` groups of a VALUES clause as lists of fields.

    A hand-written scanner rather than a split: MySQL quotes strings with single
    quotes and escapes with backslashes, and issue descriptions contain both, plus
    literal `),(` sequences. Splitting on punctuation would corrupt exactly the rows
    with the richest text.
    """
    field: list[str] = []
    row: list[str] = []
    in_string = False
    escaped = False
    depth = 0
    for ch in values:
        if depth == 0:
            if ch == "(":
                depth = 1
                field, row = [], []
            continue
        if in_string:
            field.append(ch)
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == "'":
                in_string = False
            continue
        if ch == "'":
            in_string = True
            field.append(ch)
        elif ch == ",":
            row.append("".join(field))
            field = []
        elif ch == ")":
            row.append("".join(field))
            yield row
            depth = 0
        else:
            field.append(ch)


def _clean(value: str) -> str:
    value = value.strip()
    if value == "NULL":
        return ""
    if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
        # Unescape the subset mysqldump produces; the kept columns are ids, enums
        # and datetimes, so this never has to survive a mediumtext round trip.
        return (
            value[1:-1]
            .replace("\\'", "'")
            .replace('\\"', '"')
            .replace("\\n", " ")
            .replace("\\r", " ")
            .replace("\\\\", "\\")
        )
    return value


def main() -> None:
    if not ZIP.exists():
        raise SystemExit(
            f"{ZIP} not found. Download TAWOS.sql.zip from DOI 10.5522/04/21308124 "
            f"into {RAW} first (638 MB; Apache-2.0, research use only)."
        )
    zf = zipfile.ZipFile(ZIP)
    member = zf.namelist()[0]

    with zf.open(member) as handle:
        columns = _issue_columns(io.TextIOWrapper(handle, encoding="utf-8", errors="replace"))
    index = {name: columns.index(name) for name in WANTED}
    print(f"Issue has {len(columns)} columns; keeping {len(WANTED)}")

    kept = 0
    scanned = 0
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with zf.open(member) as handle, open(OUT, "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(WANTED)
        for line in io.TextIOWrapper(handle, encoding="utf-8", errors="replace"):
            if not line.startswith(INSERT_PREFIX):
                continue
            for row in _tuples(line[len(INSERT_PREFIX):]):
                scanned += 1
                if len(row) != len(columns):
                    # A malformed tuple means the scanner lost the quoting state; that
                    # is a bug, not a data quirk, and silently dropping rows would hide
                    # it. Fail loudly with enough context to reproduce.
                    raise SystemExit(
                        f"row {scanned} parsed into {len(row)} fields, expected "
                        f"{len(columns)} — the tuple scanner is wrong, not the data"
                    )
                writer.writerow([_clean(row[index[name]]) for name in WANTED])
                kept += 1
            if scanned % 100_000 < 1000:
                print(f"  {scanned:,} issues", flush=True)
    print(f"wrote {OUT}: {kept:,} issues")


if __name__ == "__main__":
    sys.exit(main())
