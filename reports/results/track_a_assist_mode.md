# Track A — assist mode (Q13): refining an existing estimate

> **NOT A GATE RESULT.** These numbers must never appear in the G0 verdict.
> They use `First.estimate` (the expert's own number), which exists only on
> annotated datasets, never on a cold-start client spec. A model that consumes
> it is not deployable for cold-start estimation and inflates PRED(25) by
> partially copying the target. Protocol v1.2 sec. 5 / Q13 bars it from the
> gate. This section answers a separate *product* question.
> Reproduce: `python scripts/run_assist.py`.

## Kitchenham, rolling-origin folds, n_test = 73

| Model | Uses expert input? | PRED(25) | MdAPE |
|---|---|---|---|
| Expert alone (First.estimate) | — | 61.6% [50.7, 72.6] | 16.7% [9.7, 24.6] |
| **Metis assist** (cold features + expert) | yes | **68.5% [57.5, 79.5]** | 17.3% [14.5, 21.9] |
| Cold-start gate model (no expert) | no | 45.2% [34.2, 56.2] | 28.1% [20.5, 39.5] |

## Reading

- **The product story holds:** when the client already has an estimate, Metis
  refines it — PRED(25) 61.6% → 68.5% (+6.9 points), and MdAPE stays tight
  (~17%) with a much narrower CI than the expert alone. This is the "if you
  have a number, we sharpen it" mode.
- **It is not the gate.** The cold-start model (45.2%) is the deployable,
  gate-relevant number; the assist model's 68.5% is unavailable to a new
  client with a fresh spec. The two live in separate sections by design and the
  G0 verdict reads only the cold-start column.
- The gap between assist (68.5%) and cold-start (45.2%) quantifies how much
  signal the expert's prior carries on this dataset — useful product framing,
  irrelevant to the gate.
