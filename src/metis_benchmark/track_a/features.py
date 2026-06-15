"""Per-dataset feature columns for the Track A tabular channel.

Only fields available at estimation time are listed. Anything derived from the
actual effort or the realized schedule is excluded by the frozen protocol's
anti-leakage rules (reports/protocol.md sec. 6): for China that means the
productivity ratios (PDR_*, NPDR_*, NPDU_*), Duration and N_effort.
"""

from __future__ import annotations

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
}

# Split method assigned to each Track A dataset by the frozen protocol (sec. 2),
# verified by feasibility on 2026-06-12.
SPLIT_METHOD: dict[str, str] = {
    "desharnais": "temporal",
    "kitchenham": "temporal",
    "maxwell": "temporal",
    "china": "ordered_kfold",
}
