"""Mean-function GP (v1.3): the GP fits residuals of a log-size power law.

Graceful-degradation property: when the local features carry no residual signal,
the mean-function GP must fall back to (roughly) the size law itself — it must
not invent structure that isn't there.
"""

import numpy as np

from metis_benchmark.baselines import LogSizeRegression
from metis_benchmark.evaluation import mdape
from metis_benchmark.track_a.gp import LogGaussianProcess


def test_mean_function_gp_degrades_to_size_law_without_residual_signal():
    # Effort follows an exact power law in size, times small multiplicative
    # noise; the features X are pure noise (no residual signal to learn).
    rng = np.random.default_rng(0)
    n = 300
    size = rng.uniform(10, 1000, size=n)
    effort = 3.0 * size**0.9 * rng.lognormal(0.0, 0.10, size=n)
    # Non-negative noise features (real Track A features are counts / ordinal
    # codes >= 0, which the GP's log1p transform assumes).
    X = rng.uniform(0, 5, size=(n, 5))  # uninformative features

    tr, te = slice(0, 200), slice(200, n)
    # Per-dataset size law baseline (log space), fit on train only.
    reg = LogSizeRegression().fit(size[tr], effort[tr])
    base_tr = np.log(reg.predict(size[tr]))
    base_te = np.log(reg.predict(size[te]))

    # Mean-function GP fits the residual over the noise features.
    gp = LogGaussianProcess(seed=0).fit(X[tr], effort[tr], baseline_log=base_tr)
    gp_pred = gp.predict(X[te], baseline_log=base_te)
    size_pred = reg.predict(size[te])

    # The GP must track the size law closely (residual learned ~0), not degrade
    # it: its accuracy should be within a couple of points of the baseline's.
    assert mdape(effort[te], gp_pred) <= mdape(effort[te], size_pred) + 0.03
    # And predictions should be highly correlated with the size law.
    assert np.corrcoef(np.log(gp_pred), np.log(size_pred))[0, 1] > 0.95


def test_mean_function_gp_recovers_residual_signal_when_present():
    # Now a feature genuinely shifts effort above/below the size law; the
    # mean-function GP should beat the size law by exploiting that residual.
    rng = np.random.default_rng(1)
    n = 300
    size = rng.uniform(10, 1000, size=n)
    factor = rng.choice([0.5, 2.0], size=n)  # a 4x productivity split...
    signal = (factor == 2.0).astype(float)   # ...exposed by one feature
    effort = 3.0 * size**0.9 * factor * rng.lognormal(0.0, 0.05, size=n)
    X = np.column_stack([signal, rng.uniform(0, 5, size=(n, 3))])

    tr, te = slice(0, 200), slice(200, n)
    reg = LogSizeRegression().fit(size[tr], effort[tr])
    base_tr, base_te = np.log(reg.predict(size[tr])), np.log(reg.predict(size[te]))
    gp = LogGaussianProcess(seed=0).fit(X[tr], effort[tr], baseline_log=base_tr)

    gp_err = mdape(effort[te], gp.predict(X[te], baseline_log=base_te))
    size_err = mdape(effort[te], reg.predict(size[te]))
    # Exploiting the residual signal must materially beat the size law alone.
    assert gp_err < size_err - 0.05
