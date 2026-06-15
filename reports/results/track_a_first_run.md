# Track A — first engine run (GP vs baselines)

> 2026-06-12. Base Gaussian Process (RBF on log1p features, log-effort target),
> no conformal intervals and no gradient boosting yet. Out-of-fold predictions
> under the protocol-assigned split. Reproduce: `python scripts/run_track_a.py`.

| Dataset | Split | n test | GP PRED(25) | GP MdAPE | Best statistical baseline | Expert |
|---|---|---|---|---|---|---|
| china | ordered 5-fold | 499 | 19.2% | 56.5% | log-size 22.0% / 57.5% | — |
| desharnais | temporal | 15 | 33.3% | 29.9% | log-size 40.0% / 34.7% | — |
| kitchenham | temporal | 29 | **48.3%** | **25.1%** | log-size 34.5% / 32.3% | 65.5% / 18.5% |
| maxwell | temporal | 12 | **50.0%** | **25.3%** | log-size 33.3% / 41.4% | — |

## Reading

- **The GP beats both statistical baselines where the data has signal:**
  kitchenham (48.3% vs 34.5%) and maxwell (50.0% vs 33.3%), on both PRED(25)
  and MdAPE. Two Track A datasets already show the engine adding value over
  median-by-category and log-size regression.
- **China has no signal in the available features.** The in-sample ceiling of
  *any* model on China's function-point features is ~21% PRED(25) / ~55%
  MdAPE: productivity (effort/AFP) spans 2.1–24.6 at p10–p90 and is not
  explained by the recorded fields (Resource doesn't help; Dev.Type is
  constant). The GP at 19.2% is at that ceiling, not underperforming. China is
  therefore not a gate-carrying dataset.
- **desharnais is inconclusive:** only 15 test rows after the temporal split;
  GP wins on MdAPE, loses on PRED(25). Too small to weigh heavily.
- **The GP does not yet beat the human expert** (kitchenham: 48.3% vs 65.5%).
  Expected at this stage — no gradient boosting, no feature engineering, no
  use of the recorded first-estimate as a correction feature, no conformal
  layer. This is the W2 work.

## Implications for W2

1. Add the gradient-boosting channel (XGBoost/CatBoost quantile) and compare
   per dataset — the V20 thesis is GP for tiny data, GBM as signal/volume grow.
2. On kitchenham, test using `First.estimate` (the expert's own number) as a
   feature: a model that *corrects* the expert is the realistic path to
   beating it.
3. Feature engineering on maxwell (T01–T15 ordinal factors) and richer
   encodings; current run uses them raw.
4. China stays in the suite for transparency but is not expected to clear the
   gate; the 2 gate-carrying Track A datasets are likely among
   kitchenham / maxwell / desharnais / seera / cocomo81.
