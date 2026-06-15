"""Metis assist mode (Q13): refine an existing expert estimate — NEVER the gate.

`First.estimate` (the expert's own number) is available only on annotated
datasets, not when a client uploads a fresh cold-start spec. A model that uses
it is not deployable for cold-start estimation and would inflate PRED(25) by
copying the target — so it is barred from the gate (protocol v1.2 sec. 5, Q13).

This experiment lives entirely outside the gate. It answers a product question:
"if the client already has an estimate, by how much does Metis improve it?" On
kitchenham it compares, under the same rolling-origin folds:
  - expert alone (the recorded First.estimate),
  - assist model = cold-start features + First.estimate as an input,
  - cold-start model = the gate model (no First.estimate), for reference.

Usage: python scripts/run_assist.py
"""

from __future__ import annotations

import numpy as np

from metis_benchmark.datasets.loaders import load
from metis_benchmark.evaluation import bootstrap_metric_ci, mdape, pred_at, rolling_origin_split
from metis_benchmark.track_a.gp import LogGaussianProcess

KEY = "kitchenham"
# Cold-start (gate) features, plus the expert estimate for the assist variant.
COLD_FEATURES = ["Adjusted.function.points", "Client.code"]
EXPERT_COL = "expert_estimate"  # loader maps First.estimate -> expert_estimate


def main() -> None:
    df = load(KEY).dropna(subset=["effort", "date", EXPERT_COL, *COLD_FEATURES]).reset_index(drop=True)
    df = df[df["effort"] > 0].reset_index(drop=True)
    n = len(df)

    y = df["effort"].to_numpy(dtype=float)
    X_cold = df[COLD_FEATURES].to_numpy(dtype=float)
    # Assist matrix appends the expert estimate as an extra input column.
    X_assist = np.column_stack([X_cold, df[EXPERT_COL].to_numpy(dtype=float)])
    expert = df[EXPERT_COL].to_numpy(dtype=float)

    cold_pred = np.full(n, np.nan)
    assist_pred = np.full(n, np.nan)
    tested = np.zeros(n, dtype=bool)

    # Same rolling-origin folds as the gate, so the three numbers are comparable.
    for fold in rolling_origin_split(df, "date"):
        tr = np.concatenate([fold.train, fold.calibration])
        te = fold.test
        tested[te] = True
        cold_pred[te] = LogGaussianProcess(seed=0).fit(X_cold[tr], y[tr]).predict(X_cold[te])
        assist_pred[te] = LogGaussianProcess(seed=0).fit(X_assist[tr], y[tr]).predict(X_assist[te])

    print(f"== {KEY} assist experiment (rolling-origin, n_test={tested.sum()}) ==")
    print("  (outside the gate — Q13; product question, not a G0 number)")
    for name, pred in [
        ("expert alone", expert),
        ("Metis assist (+expert)", assist_pred),
        ("cold-start (gate, no expert)", cold_pred),
    ]:
        yt, yp = y[tested], pred[tested]
        print(f"  {name:<30} PRED(25)={bootstrap_metric_ci(yt, yp, pred_at, seed=0)}  "
              f"MdAPE={bootstrap_metric_ci(yt, yp, mdape, seed=0)}")


if __name__ == "__main__":
    main()
