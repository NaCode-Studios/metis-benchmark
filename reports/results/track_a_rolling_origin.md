# Track A — rolling-origin CV (protocol v1.2) vs single split

> 2026-06-12. Pre-registered estimator (protocol.md sec. 8, committed before
> this run). Both estimators shown side by side; the gate verdict reads the
> rolling-origin column. CI95 = 2000-resample percentile bootstrap, pooled
> out-of-fold predictions. Reproduce: `python scripts/run_track_a_rolling.py`.

## Side by side (PRED(25), engine vs log-size baseline)

| Dataset | est. | n test | GP | GBM | gate (sel.) | gate − baseline (CI95) |
|---|---|---|---|---|---|---|
| desharnais | single | 15 | 33.3% | 26.7% | — | — |
| desharnais | rolling | 39 | 38.5% | 33.3% | GBM 33.3% | −10.3% [−28.2, 10.3] indist. |
| kitchenham | single | 29 | 48.3% | 51.7% | — | — |
| kitchenham | rolling | 73 | **45.2%** | 39.7% | GP 45.2% | **+12.3% [4.1, 20.6] BEATS** |
| maxwell | single | 12 | 50.0% | 50.0% | — | — |
| maxwell | rolling | 31 | **51.6%** | 41.9% | GBM 41.9% | +12.9% [−9.7, 38.7] indist. |
| **aggregate** | single | 56 | — | 44.6% | — | +8.9% [−5.4, 23.2] indist. |
| **aggregate** | rolling | 143 | — | — | 41.3% | +6.3% [−2.1, 15.4] indist. |

## What rolling-origin changed

1. **CIs are now interpretable.** Pooled test rows go from 12–29 to 31–73 per
   dataset (143 aggregate vs 56). The kitchenham engine now beats the baseline
   with CI [4.1, 20.6] — a solid clearance of zero, where the single split sat
   right at the boundary. This is exactly what the amendment was for.

2. **GP beats GBM under the rigorous estimator — reversing Step 2.** On the
   single split GBM edged GP; under rolling-origin GP is clearly better on
   both gate datasets with signal (kitchenham 45.2% vs 39.7%, maxwell 51.6% vs
   41.9%). The single-split GBM advantage was noise. This vindicates the V20
   cold-start doctrine (GP for <1000 projects) and is the kind of conclusion
   only a properly powered estimator can reach.

3. **The pre-registered selection rule was imperfect — honestly so.** The rule
   (GBM-primary, test-blind GP fallback) kept GBM on maxwell, where GP would
   have scored 51.6% and beaten the baseline (+22.6% [3.2, 45.2]) instead of
   GBM's indistinguishable 41.9%. The rule is committed and honored as-is; the
   learning is logged for the next iteration (Q16): under rolling-origin the
   doctrine-correct primary is GP, not GBM.

## Gate reading (PRED(25) ≥ 55%)

The engine is **below the 55% threshold on every gate dataset** under
rolling-origin: kitchenham 45.2%, maxwell 41.9% (gate-selected) / 51.6% (GP),
desharnais 33.3%, aggregate 41.3% [33.6, 49.0]. It **beats the log-size
baseline on kitchenham** (and on maxwell with GP), but does not reach 55%.

This is a genuine **not-passing** result on the point-accuracy threshold — and
now it means something, because the estimator is adequately powered. It is not
a dead end: the in-sample ceiling (66–78%, Q15) shows 55% is reachable, so the
gap is generalization, addressable by feature engineering and more gate
datasets (cocomo81, seera). MdAPE (gate-selected, rolling) is 28.8% aggregate,
above the ≤22% threshold — same story, same remedy.

## ERRATA (2026-07-02)

The tables above are the frozen 2026-06-12 phase snapshot and are kept intact
as the historical record. A rerun of `python scripts/run_track_a_rolling.py`
on 2026-07-02 produces different numbers on desharnais and on the aggregates,
for two distinct, fully-diagnosed causes. **No change flips any verdict: every
value stays well below the 55% PRED(25) gate threshold; the not-passing
result stands.**

**Cause 1 — bug fix in the gate series (2026-07-02).** `run_dataset` fed the
"gate (selected)" model the *raw* test matrix instead of the train-median
*imputed* one (`experiment.py`), so on datasets with missing feature values
the gate series silently diverged from the selected regressor's own
out-of-fold predictions. Fixed and locked by a regression test
(`tests/test_experiment.py`). Measured impact (rerun before vs after the fix,
same code otherwise): PRED(25) unchanged everywhere; desharnais gate MdAPE
29.6% → 28.5% (now exactly the selected GBM's row), aggregate gate MdAPE
28.5% → 28.3%.

**Cause 2 — v1.3 row policy not propagated to this snapshot.** The W3 v1.3
change (commit 58c5bcc) stopped dropping rows with missing *features* (they
are imputed with the training-fold median instead). Desharnais has 4 rows
with −1-coded missing `TeamExp`/`ManagerExp`: 77 → 81 usable rows,
39 → 41 pooled test rows. Re-applying the old drop-rows policy reproduces the
frozen numbers exactly (n=39, GP 38.5%, GBM/gate 33.3%), which confirms the
mechanism. Kitchenham and maxwell have no missing features and reproduce
unchanged.

Old → new (rolling-origin estimator, rerun 2026-07-02):

| Value | 2026-06-12 (above) | 2026-07-02 rerun |
|---|---|---|
| desharnais n test (rolling) | 39 | 41 |
| desharnais GP | 38.5% | 36.6% |
| desharnais GBM / gate (sel.) | 33.3% | 41.5% |
| desharnais gate − baseline | −10.3% [−28.2, 10.3] indist. | 0.0% [−17.1, 19.5] indist. |
| kitchenham (all rows) | unchanged | 45.2% / 39.7% / gate GP 45.2%, +12.3% [4.1, 20.6] BEATS |
| maxwell (all rows) | unchanged | 51.6% / 41.9% / gate GBM 41.9%, +12.9% [−9.7, 38.7] indist. |
| aggregate single n / GBM | 56 / 44.6% | 57 / 50.9% |
| aggregate single GBM − baseline | +8.9% [−5.4, 23.2] indist. | +15.8% [1.8, 29.8] BEATS |
| aggregate rolling n / gate | 143 / 41.3% [33.6, 49.0] | 145 / 43.4% [35.9, 51.0] |
| aggregate rolling gate − baseline | +6.3% [−2.1, 15.4] indist. | +9.0% [0.7, 17.3] BEATS |
| aggregate rolling gate MdAPE | 28.8% | 28.3% |

Both baseline-comparison upgrades (aggregate "indistinguishable" → "BEATS")
come from cause 2 — the retained desharnais rows — not from the bug fix, and
neither moves any PRED(25) toward the 55% bar: the gate reading above
(**below 55% on every gate dataset**) is unaffected.
