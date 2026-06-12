"""Loaders normalizing every dataset to the per-track canonical schema.

Track A (project level): one row per project with canonical columns
    effort           actual effort, unit in EFFORT_UNITS (never mixed across datasets)
    size             primary size measure (FP, object points, KLOC...)
    category         categorical field for the median-by-category baseline
    date             project date where the dataset has one (temporal split)
    expert_estimate  estimate recorded at the time, where available
plus the original feature columns untouched.

Track B (task level): one row per task with
    text             title + description (the semantic channel input)
    effort           actual effort or story points (unit declared)
    expert_estimate  human estimate where available
    project          source project (grouping / per-project splits)
    date             estimation date where available

Effort units are heterogeneous across datasets by nature; the unit is part of
the dataset's identity and datasets are never pooled in one training set.
"""

from __future__ import annotations

import sqlite3
from typing import Callable

import numpy as np
import pandas as pd

from metis_benchmark.datasets.arff import read_arff
from metis_benchmark.datasets.registry import raw_path

# effort unit per dataset; "verify" entries are confirmed during EDA
EFFORT_UNITS: dict[str, str] = {
    "desharnais": "person-hours",
    "cocomo81": "person-months",
    "china": "person-hours",
    "kitchenham": "person-hours",
    "maxwell": "person-hours",
    "albrecht": "kilo person-hours (verify against literature)",
    "seera": "person-hours (verify against SEERA ReadMe)",
    "deepse": "story points",
    # EDA: median 3600, p90 28800 (1h / 8h) -> JIRA timespent in seconds
    "josse": "seconds",
    "sip": "person-hours",
}


# ----- Track A -----


def load_desharnais() -> pd.DataFrame:
    df = pd.read_csv(raw_path("desharnais") / "desharnais.csv")
    df = df.replace(-1, np.nan)  # Desharnais encodes missing as -1
    df["effort"] = df["Effort"].astype(float)
    df["size"] = df["PointsAjust"].astype(float)
    df["category"] = df["Language"].astype("string")
    # YearEnd is a two-digit 1980s year: enough ordering for a temporal split
    df["date"] = pd.to_datetime(df["YearEnd"].astype(int) + 1900, format="%Y")
    return df


def load_cocomo81() -> pd.DataFrame:
    df = read_arff(raw_path("cocomo81") / "coc81-dem.arff", trailing_extra=("months",))
    df["effort"] = df["effort"].astype(float)
    df["size"] = df["kloc"].astype(float)
    return df


def load_china() -> pd.DataFrame:
    df = read_arff(raw_path("china") / "china.arff")
    df["effort"] = df["Effort"].astype(float)
    df["size"] = df["AFP"].astype(float)
    return df


def load_kitchenham() -> pd.DataFrame:
    df = read_arff(raw_path("kitchenham") / "kitchenham.arff")
    df["effort"] = df["Actual.effort"].astype(float)
    df["size"] = df["Adjusted.function.points"].astype(float)
    df["category"] = df["Project.type"].astype("string")
    df["date"] = pd.to_datetime(df["Actual.start.date"], errors="coerce")
    df["expert_estimate"] = df["First.estimate"].astype(float)
    return df


def load_maxwell() -> pd.DataFrame:
    df = read_arff(raw_path("maxwell") / "maxwell.arff")
    df["effort"] = df["Effort"].astype(float)
    df["size"] = df["Size"].astype(float)
    df["category"] = df["App"].astype("string")
    # Syear is a two-digit 1980s/90s year
    df["date"] = pd.to_datetime(df["Syear"].astype(int) + 1900, format="%Y")
    return df


def load_albrecht() -> pd.DataFrame:
    df = read_arff(raw_path("albrecht") / "albrecht.arff")
    df["effort"] = df["Effort"].astype(float)
    df["size"] = df["AdjFP"].astype(float)
    return df


def load_seera() -> pd.DataFrame:
    df = read_arff(
        raw_path("seera")
        / "The SEERA Dataset"
        / "Dataset Files"
        / "ARFF Format"
        / "ARFF_SEERA cost estimation dataset.csv.arff"
    )
    df["effort"] = pd.to_numeric(df["Actual effort"], errors="coerce")
    df["size"] = pd.to_numeric(df["Object points"], errors="coerce")
    df["category"] = df["Application domain"].astype("string")
    df["date"] = pd.to_datetime(df["Year of project"].astype("Int64"), format="%Y", errors="coerce")
    df["expert_estimate"] = pd.to_numeric(df["Estimated effort"], errors="coerce")
    return df


# ----- Track B -----

DEEPSE_PROJECTS = [
    "APSTUD", "BAM", "CLOV", "DM", "DURACLOUD", "JRESERVER", "MDL",
    "MESOS", "MULESTUDIO", "MULE", "TIMOB", "TISTUD", "USERGRID", "XD",
]


def load_deepse() -> pd.DataFrame:
    frames = []
    for project in DEEPSE_PROJECTS:
        df = pd.read_csv(raw_path("deepse") / f"{project}_deep-se.csv")
        df["project"] = project
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    out["text"] = out["title"].fillna("") + "\n\n" + out["description"].fillna("")
    out["effort"] = out["storypoint"].astype(float)
    return out


def load_josse() -> pd.DataFrame:
    db = raw_path("josse") / "josse" / "JOSSE_18092020.sqlite3"
    with sqlite3.connect(db) as conn:
        df = pd.read_sql_query('SELECT * FROM "case"', conn)
    df["text"] = df["corpus"].fillna("")
    df["effort"] = pd.to_numeric(df["actual_effort"], errors="coerce")
    # JOSSE encodes "no expert estimate" as -1
    expert = pd.to_numeric(df["expert_estimated_effort"], errors="coerce")
    df["expert_estimate"] = expert.where(expert > 0)
    df["project"] = df["id"].astype(str).str.split("-").str[0]
    return df


def load_sip() -> pd.DataFrame:
    tasks = pd.read_csv(raw_path("sip") / "Sip-task-info.csv", encoding="cp1252")
    dates = pd.read_csv(raw_path("sip") / "est-act-dates.csv")
    # Both files hold one row per estimate event and tasks can be re-estimated;
    # pair the k-th occurrence of each task in one file with the k-th in the
    # other (a plain TaskNumber merge would cross-join repeated tasks).
    dates["EstimateOn"] = pd.to_datetime(dates["EstimateOn"], format="%d-%b-%y", errors="coerce")
    tasks["occurrence"] = tasks.groupby("TaskNumber").cumcount()
    dates["occurrence"] = dates.sort_values("EstimateOn").groupby("TaskNumber").cumcount()
    df = tasks.merge(dates, on=["TaskNumber", "occurrence"], how="left")
    df["text"] = df["Summary"].fillna("")
    df["effort"] = pd.to_numeric(df["HoursActual"], errors="coerce")
    df["expert_estimate"] = pd.to_numeric(df["HoursEstimate"], errors="coerce")
    df["category"] = df["Category"].astype("string")
    df["date"] = df["EstimateOn"]
    df["project"] = df["ProjectCode"].astype("string")
    return df


LOADERS: dict[str, Callable[[], pd.DataFrame]] = {
    "desharnais": load_desharnais,
    "cocomo81": load_cocomo81,
    "china": load_china,
    "kitchenham": load_kitchenham,
    "maxwell": load_maxwell,
    "albrecht": load_albrecht,
    "seera": load_seera,
    "deepse": load_deepse,
    "josse": load_josse,
    "sip": load_sip,
}


def load(key: str) -> pd.DataFrame:
    """Load a dataset by registry key, normalized to its track's schema."""
    if key not in LOADERS:
        raise KeyError(f"no loader for {key!r}; available: {sorted(LOADERS)}")
    return LOADERS[key]()
