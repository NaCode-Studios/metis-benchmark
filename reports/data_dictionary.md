# Data dictionary — week 1 EDA

> Generated from `scripts/eda.py` on the raw files downloaded by
> `scripts/download.py` (sources verified 2026-06-12, see registry.py).
> Effort units are per-dataset and datasets are **never pooled** in training.

## Track A — project level

| Dataset | Rows | Effort unit | Median / p90 effort | Size | Category | Date (temporal split) | Expert estimate |
|---|---|---|---|---|---|---|---|
| desharnais | 81 | person-hours | 3,647 / 10,577 | adjusted FP (`PointsAjust`) | language (3) | year-end 1982–1988 (year only) | — |
| cocomo81 | 63 | person-months | 98 / 1,233 | KLOC | — | — (fallback split) | — |
| china | 499 | person-hours | 1,829 / 8,708 | adjusted FP (`AFP`) | — | — (fallback split) | — |
| kitchenham | 145 | person-hours | 1,557 / 5,417 | adjusted FP | project type (6) | start date 1994-01..1998-11 (day) | **100%** (`First.estimate`) |
| maxwell | 62 | person-hours | 5,190 / 14,947 | FP (`Size`) | app type (5) | start year 1985–1993 (year only) | — |
| albrecht | 24 | kilo person-hours¹ | 11.45 / 54.3 | adjusted FP | — | — (fallback split) | — |
| seera | 120 | person-hours¹ | 4,576 / 21,146 | object points (98%) | app domain (6) | year 1993–2019 (year only) | **100%** (`Estimated effort`) |

¹ unit to be confirmed against the dataset's own documentation before the
protocol freeze (Albrecht effort is conventionally reported in thousands of
hours; SEERA ReadMe.pdf states the unit).

## Track B — task level with text

| Dataset | Rows | Effort unit | Median / p90 | Text (median chars) | Projects | Date | Expert estimate |
|---|---|---|---|---|---|---|---|
| deepse | 21,064 | story points | 4 / 13 | 328 | 14² | — (issue keys give per-project ordering) | — |
| josse | 23,186 | seconds³ | 3,600 / 28,800 | 295 | 371 | — (reference URLs allow recovery if needed) | **19%** (paper-consistent) |
| sip | 12,299 estimate events (10,266 tasks) | person-hours | 3 / 24 | 43 | 20 | EstimateOn 2004-02..2014-12 (day) | **100%** (`HoursEstimate`) |

² 14 of the original 16 Deep-SE projects: two were withdrawn from the public
domain for GDPR compliance (documented in the SOLAR-group replication repo).
³ JIRA `timespent` convention; EDA medians (3600 = 1h, 28800 = 8h) confirm
seconds. Convert to hours for reporting.

## Notes for the protocol

- **Log-normality**: log-effort skew is between −0.34 and +0.64 on all ten
  datasets — modeling z = log(effort) is appropriate everywhere.
- **True temporal split** (day-level dates): kitchenham, sip. **Year-level
  ordering** (coarse but usable): desharnais, maxwell, seera. **No dates**:
  cocomo81, china, albrecht, deepse, josse → fallback split must be declared.
- **Expert baseline available on 4 datasets**: kitchenham, seera (Track A),
  josse, sip (Track B) — the model-vs-human comparison is possible on both
  tracks, not only Track B.
- **SiP re-estimates**: tasks can be estimated more than once; the unit of
  analysis is the estimate event (12,299), paired k-th occurrence to k-th
  date. Tasks: 10,266.
- **Albrecht** (24 rows) is too small to count toward the gate; keep for
  completeness only.
- **SEERA missing values**: encoded as `?` and `N/A` in the ARFF; handled by
  the reader. `% project gain (loss)` is target leakage for cost — exclude
  from features.
- **JOSSE project skew**: 371 JIRA projects, long tail — per-project splits
  would starve; treat as a single corpus with project as a feature.
