"""Shape-contract tests for the GBM quantile regressor.

XGBoost's predict() output shape depends on the case — (n, m) for a
multi-quantile model but (n,) for a single-quantile one, hence (1, m)/(1,)
for one row — so predict_quantiles must normalize explicitly. These tests pin
the contract for every (n_rows, n_quantiles) combination, because the CQR
interval step consumes single rows at serving time while the backtest feeds
whole test blocks: both paths must yield one value per row per quantile.
"""

import numpy as np
import pytest

from metis_benchmark.track_a.gbm import LogGBMQuantile

# Small synthetic regression problem: positive effort with a size-driven trend
# so the model has something real to fit (values themselves are irrelevant —
# only the output SHAPES are under test).
rng = np.random.default_rng(0)
X_TRAIN = rng.uniform(1.0, 100.0, size=(40, 3))
Y_TRAIN = 10.0 + 5.0 * X_TRAIN[:, 0] + rng.uniform(0.0, 10.0, size=40)


def _fitted(quantiles: tuple[float, ...]) -> LogGBMQuantile:
    return LogGBMQuantile(quantiles=quantiles, seed=0).fit(X_TRAIN, Y_TRAIN)


@pytest.mark.parametrize(
    ("n_rows", "quantiles"),
    [
        (1, (0.1, 0.5, 0.9)),  # single row, multi-quantile: XGBoost emits (1, 3)
        (7, (0.5,)),           # many rows, single quantile: XGBoost emits (7,)
        (7, (0.1, 0.5, 0.9)),  # the general n x m case: XGBoost emits (7, 3)
    ],
)
def test_predict_quantiles_shape_contract(n_rows, quantiles):
    model = _fitted(quantiles)
    out = model.predict_quantiles(X_TRAIN[:n_rows])
    # One entry per requested quantile, keyed by the quantile itself.
    assert set(out) == set(quantiles)
    for q, values in out.items():
        # The contract: exactly one prediction per input row, 1-D, finite,
        # and positive (log-space model back-transformed with exp).
        assert values.shape == (n_rows,), f"quantile {q}: {values.shape}"
        assert np.isfinite(values).all()
        assert (values > 0).all()


def test_single_row_equals_batch_row():
    # Predicting one row must give the same numbers as slicing that row out of
    # a batch prediction: a reshape bug (transposing quantiles into rows)
    # breaks exactly this equivalence.
    model = _fitted((0.1, 0.5, 0.9))
    single = model.predict_quantiles(X_TRAIN[:1])
    batch = model.predict_quantiles(X_TRAIN[:7])
    for q in (0.1, 0.5, 0.9):
        np.testing.assert_allclose(single[q][0], batch[q][0], rtol=1e-6)


def test_point_prediction_is_the_median_row_per_sample():
    # predict() must return the p50 series with one value per row.
    model = _fitted((0.5, 0.9))
    pred = model.predict(X_TRAIN[:7])
    assert pred.shape == (7,)
    np.testing.assert_allclose(pred, model.predict_quantiles(X_TRAIN[:7])[0.5])
