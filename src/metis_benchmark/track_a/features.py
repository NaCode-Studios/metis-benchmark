"""Per-dataset feature columns for the Track A tabular channel.

Only fields available at estimation time are listed. Anything derived from the
actual effort or the realized schedule is excluded by the frozen protocol's
anti-leakage rules (reports/protocol.md sec. 6). For cocomo81 and seera the
feature sets were fixed by the v1.3 adversarial leakage audit (sec. 8).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# registry key -> ordered list of numeric feature columns
FEATURE_COLUMNS: dict[str, list[str]] = {
    # China: the function-point breakdown plus the change counts. AFP is also
    # the canonical "size" used by the log-size baseline. Excluded as leakage:
    # Duration, N_effort, PDR_*, NPDR_*, NPDU_* (all derived from actual effort).
    "china": [
        "AFP", "Input", "Output", "Enquiry", "File", "Interface",
        "Added", "Changed", "Deleted",
    ],
    # Desharnais: team/manager experience, transaction and entity counts, and
    # the adjusted/unadjusted function points. Excluded as leakage: Length
    # (realized project duration in months).
    "desharnais": [
        "TeamExp", "ManagerExp", "Transactions", "Entities",
        "PointsNonAdjust", "Adjustment", "PointsAjust",
    ],
    # Kitchenham: adjusted function points and the client/project codes known
    # at estimation time. Excluded as leakage: Actual.duration. (First.estimate
    # is the human estimate -> kept as the expert baseline, not as a feature,
    # to keep the engine-vs-expert comparison clean.)
    "kitchenham": ["Adjusted.function.points", "Client.code"],
    # Maxwell: application/hardware/DB environment flags, language count and the
    # T01-T15 ordinal productivity factors, plus function-point Size. Excluded
    # as leakage: Duration and Time (realized schedule).
    "maxwell": [
        "App", "Har", "Dba", "Ifc", "Source", "Telonuse", "Nlan", "Size",
        "T01", "T02", "T03", "T04", "T05", "T06", "T07", "T08",
        "T09", "T10", "T11", "T12", "T13", "T14", "T15",
    ],
    # COCOMO81 (v1.3): the COCOMO scale factors and effort multipliers (ordinal
    # ratings) plus kloc as size. Excluded as outcome/leakage: effort (target),
    # defects, months (realized). Ratings ordinal-encoded vl<l<n<h<vh<xh.
    "cocomo81": [
        "prec", "flex", "resl", "team", "pmat", "rely", "data", "cplx",
        "ruse", "docu", "time", "stor", "pvol", "acap", "pcap", "pcon",
        "apex", "plex", "ltex", "tool", "site", "sced", "kloc",
    ],
    # SEERA (v1.3): the 51 leakage-safe attributes (organizational, contractual,
    # sizing, planned-team, requirements-level, product-requirement) confirmed by
    # the adversarial leakage audit. Column names preserve the ARFF spelling and
    # spacing verbatim. SEERA is the FINAL HOLDOUT — wired here, but its data is
    # not touched during development (Phases 2-4); opened once at the verdict.
    "seera": [
        "Organization type", "Role in organization", "Size of organization",
        "Size of IT department", "Customer organization type", "Estimated  duration",
        "Development type", "Application domain", "Contract maturity",
        "Government policy impact", "Economic instability impact",
        "Organization management structure clarity", "Developer hiring policy",
        "Developer incentives policy ", "Developer training",
        "Development team management", "Top management support",
        "Top management opinion of previous system", "Clarity of manual system",
        "User resistance", "User computer experience", "Project manager experience",
        "Consultant availability", "DBMS  expert availability", "Precedentedness",
        "Software tool experience", "Programmers experience in programming language",
        "Team selection", "Team size", "Dedicated team members", "Daily working hours",
        "Income satisfaction", "Development environment adequacy", "Tool availability ",
        "Methodology", "# Multiple programing languages ", "Programming language used",
        "DBMS used", "Technical stability", "Open source software",
        "Level of outsourcing", "Degree of software reuse ", "Degree of risk management",
        "Use of standards", "Degree of standards usage", "Required reusability",
        "Performance requirements", "Product complexity", "Security requirements",
        "Reliability requirements", "Specified H/W",
    ],
}

# Split method assigned to each Track A dataset by the frozen protocol (sec. 2).
SPLIT_METHOD: dict[str, str] = {
    "desharnais": "temporal",
    "kitchenham": "temporal",
    "maxwell": "temporal",
    "china": "ordered_kfold",
    "cocomo81": "ordered_kfold",  # dateless
    "seera": "temporal",  # has 'Year of project' (year-level)
}

# COCOMO ordinal ratings -> integer codes (very-low .. extra-high).
COCOMO_ORDINAL = {"vl": 1, "l": 2, "n": 3, "h": 4, "vh": 5, "xh": 6}


def encode_features(df: pd.DataFrame, key: str) -> pd.DataFrame:
    """Return the dataset's feature columns as a numeric frame (NaN for missing).

    Stateless and leakage-free: ordinal encoding for COCOMO ratings, numeric
    coercion elsewhere. Missing values are left as NaN here and imputed later
    with the *training-fold* median inside the fold loop (v1.3), so test-fold
    statistics never influence the imputation.
    """
    cols = FEATURE_COLUMNS[key]
    out = pd.DataFrame(index=df.index)
    if key == "cocomo81":
        # Map the ordinal rating strings to their rank; kloc stays numeric.
        for c in cols:
            if c == "kloc":
                out[c] = pd.to_numeric(df[c], errors="coerce")
            else:
                out[c] = df[c].map(COCOMO_ORDINAL).astype(float)
    else:
        # Coerce every feature to a float; non-numeric markers become NaN.
        for c in cols:
            out[c] = pd.to_numeric(df[c], errors="coerce")
    return out


def impute_train_median(X_train: np.ndarray, *Xs: np.ndarray) -> tuple[np.ndarray, ...]:
    """Fill NaNs with per-column training medians (fit on train, applied to all).

    Returns the imputed X_train followed by each additional matrix imputed with
    the same train-derived medians. A column that is all-NaN on train falls back
    to 0 after centering (it carries no information either way).
    """
    med = np.nanmedian(X_train, axis=0)
    med = np.where(np.isnan(med), 0.0, med)

    def fill(M: np.ndarray) -> np.ndarray:
        M = np.array(M, dtype=float, copy=True)
        idx = np.where(np.isnan(M))
        M[idx] = np.take(med, idx[1])
        return M

    return (fill(X_train), *(fill(M) for M in Xs))
