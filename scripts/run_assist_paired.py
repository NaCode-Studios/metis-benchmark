"""Paired significance test for the assist mode (Kitchenham).

`run_assist.py` reported three point estimates (expert alone, Metis assist,
cold-start) each with its own bootstrap CI. Three *independent* CIs cannot tell
us whether assist truly beats the expert: the expert and the assist model are
scored on the *same* projects, so their errors are correlated, and the honest
question is a *paired* one — on the same rolling-origin folds, does adding the
expert estimate to the cold-start features move the per-project hit/miss pattern
in a way the data can distinguish from noise?

This script answers exactly that, with two paired tests (both from the shared
`evaluation` module so every paired comparison uses one implementation):
  1. McNemar's exact test on the PRED(25) hit indicators (assist vs expert);
  2. a paired bootstrap of the metric difference d(PRED25) and d(MdAPE).

Like run_assist.py it lives ENTIRELY OUTSIDE THE GATE: it uses First.estimate
(the expert's own number), which is barred from any gate-carrying model
(protocol v1.2 sec. 5). It answers an applied question, never a G0 gate number.

Caveat: at n=73 this test is under-powered (~11
discordant pairs → McNemar power ≈0.16). "Assist not demonstrated" therefore
means *absence of proof*, never *proof of absence*.

Usage: PYTHONPATH=src python scripts/run_assist_paired.py
"""

from __future__ import annotations

# numpy for the vectorized fold bookkeeping and prediction buffers.
import numpy as np

# The dataset loader maps raw dataset columns to the engine's canonical names
# (e.g. First.estimate -> expert_estimate), so the same code reads every dataset.
from metis_benchmark.datasets.loaders import load

# Shared metrics + paired tests: one source of truth for paired comparisons.
from metis_benchmark.evaluation import (
    mcnemar_exact,          # exact paired McNemar on two hit vectors
    mdape,                  # median APE across the test rows
    paired_bootstrap_diff,  # CI of metric(a) - metric(b) on shared resamples
    pred_at,                # PRED(25): fraction of rows with APE <= 0.25
    pred_hits,              # boolean PRED(25) hit vector (the unit McNemar pairs)
    rolling_origin_split,   # expanding-window temporal folds (protocol v1.2)
)

# The cold-start regressor: a log-space GP, the engine's primary small-data model.
from metis_benchmark.track_a.gp import LogGaussianProcess

# --- Frozen experiment constants (mirror run_assist.py so the two agree) ------
# Kitchenham is the only gate dataset that ships a recorded expert estimate AND
# a usable date, so it is the one place a paired assist-vs-expert test is valid.
KEY = "kitchenham"
# Cold-start (gate-legal) features: adjusted function points + the client code.
COLD_FEATURES = ["Adjusted.function.points", "Client.code"]
# The expert's own estimate; loader name for the raw First.estimate column.
EXPERT_COL = "expert_estimate"
# Bootstrap replicate count and seed: fixed so the CIs are bit-for-bit reproducible.
N_BOOT = 2000
SEED = 0


def predictions() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Out-of-fold predictions for expert, assist and cold-start, aligned by row."""
    # Load Kitchenham and drop rows missing any field this experiment needs;
    # reset_index so positional (integer) indexing into numpy arrays stays valid.
    df = (
        load(KEY)
        .dropna(subset=["effort", "date", EXPERT_COL, *COLD_FEATURES])
        .reset_index(drop=True)
    )
    # log-space modeling needs strictly positive effort; drop any non-positive row.
    df = df[df["effort"] > 0].reset_index(drop=True)
    n = len(df)

    # Actual effort (target), cold-start features, and the assist matrix that
    # appends the expert estimate as an extra input column.
    y = df["effort"].to_numpy(dtype=float)
    X_cold = df[COLD_FEATURES].to_numpy(dtype=float)
    X_assist = np.column_stack([X_cold, df[EXPERT_COL].to_numpy(dtype=float)])
    expert = df[EXPERT_COL].to_numpy(dtype=float)

    # Prediction buffers; NaN marks "never tested", flipped true as folds fill in.
    cold = np.full(n, np.nan)
    assist = np.full(n, np.nan)
    tested = np.zeros(n, dtype=bool)

    # Walk the same expanding-window folds the gate uses, so the numbers compare.
    for fold in rolling_origin_split(df, "date"):
        # The GP fits on train+calibration (no held-out calibration needed for a
        # point test — CQR intervals are not part of this comparison).
        tr = np.concatenate([fold.train, fold.calibration])
        te = fold.test
        tested[te] = True
        cold[te] = LogGaussianProcess(seed=0).fit(X_cold[tr], y[tr]).predict(X_cold[te])
        assist[te] = LogGaussianProcess(seed=0).fit(X_assist[tr], y[tr]).predict(X_assist[te])

    # Keep only rows that some fold tested; this is the common, aligned sample.
    m = tested
    return y[m], expert[m], assist[m], cold[m]


def main() -> None:
    # Generate the aligned out-of-fold predictions for all three models.
    y, expert, assist, cold = predictions()

    # Paired McNemar: does assist's hit pattern systematically beat the expert's?
    mc = mcnemar_exact(pred_hits(y, assist), pred_hits(y, expert))
    # Paired bootstrap of the PRED(25) gap (assist minus expert): CI excluding 0
    # would mean a real difference; CI straddling 0 means indistinguishable.
    dP = paired_bootstrap_diff(y, assist, expert, pred_at, n_boot=N_BOOT, seed=SEED)
    # Same for MdAPE; here a *negative* gap would favor assist (lower error).
    dM = paired_bootstrap_diff(y, assist, expert, mdape, n_boot=N_BOOT, seed=SEED)

    # --- Report -------------------------------------------------------------
    print(f"== assist paired test ({KEY}, rolling-origin, n={len(y)}) ==")
    print("  (OUTSIDE the gate — uses First.estimate; applied question only)")
    print(f"  expert alone   PRED(25)={pred_at(y, expert):.1%}  MdAPE={mdape(y, expert):.1%}")
    print(f"  Metis assist   PRED(25)={pred_at(y, assist):.1%}  MdAPE={mdape(y, assist):.1%}")
    print(f"  cold-start     PRED(25)={pred_at(y, cold):.1%}  MdAPE={mdape(y, cold):.1%}")
    print(f"  McNemar exact: b(assist>expert)={mc['b']} c(expert>assist)={mc['c']} "
          f"discordant={mc['discordant']} p={mc['p']:.4f}")
    print(f"  dPRED25 = {dP.point:+.1%} [{dP.low:+.1%}, {dP.high:+.1%}]  "
          f"(CI includes 0 -> not demonstrated; underpowered)")
    print(f"  dMdAPE  = {dM.point:+.1%} [{dM.low:+.1%}, {dM.high:+.1%}]  "
          f"(positive -> assist no better on error)")


if __name__ == "__main__":
    main()
