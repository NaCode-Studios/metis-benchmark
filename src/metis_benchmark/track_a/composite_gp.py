"""Composite-kernel GP: cosine(text) + RBF(tabular).

The cold-start engine (`LogGaussianProcess`) runs one RBF over all features. This
model adds a semantic channel: each project carries an embedding of its
requirement text alongside its tabular features. Those two blocks have different
geometry — embeddings are unit vectors compared by angle, tabular features are
magnitudes compared by scaled distance — so they need *different* kernels summed:

    k(x, x') = k_text(emb, emb') + k_tab(tab, tab') + white-noise

`k_text` is an RBF on the unit-normalized embeddings: on the unit sphere
‖a−b‖² = 2 − 2·cos(a,b), so an RBF there is a monotone function of cosine
similarity — i.e. a cosine kernel with a tunable bandwidth (a cosine kernel on
text). `k_tab` is the familiar RBF on standardized tabular features.

sklearn kernels act on *all* columns, so `OnColumns` restricts a kernel to a
contiguous column slice and forwards the hyperparameter gradient unchanged; the
two restricted kernels then compose with `+` and are tuned jointly by the
marginal likelihood, exactly like any sklearn kernel.
"""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    RBF,
    ConstantKernel,
    Hyperparameter,
    Kernel,
    WhiteKernel,
)
from sklearn.preprocessing import StandardScaler


class OnColumns(Kernel):
    """Apply `kernel` to the column slice [start:stop] of X, forwarding the rest.

    Everything that makes a kernel tunable — its hyperparameters, theta, bounds,
    and the gradient returned by __call__ — is delegated to the wrapped kernel on
    the sliced data, so OnColumns composes with Sum/Product/WhiteKernel and is
    optimized by GaussianProcessRegressor with no special handling.
    """

    def __init__(self, kernel: Kernel, start: int, stop: int) -> None:
        # The kernel to restrict, and the half-open column range it sees.
        self.kernel = kernel
        self.start = start
        self.stop = stop

    def get_params(self, deep: bool = True) -> dict:
        # Expose own params plus the nested kernel's (prefixed) so sklearn.clone
        # can rebuild OnColumns *and* its inner kernel during theta cloning.
        params = dict(kernel=self.kernel, start=self.start, stop=self.stop)
        if deep:
            params.update(
                ("kernel__" + name, value) for name, value in self.kernel.get_params().items()
            )
        return params

    @property
    def hyperparameters(self) -> list[Hyperparameter]:
        # Re-export the inner kernel's hyperparameters under a "kernel__" prefix.
        return [
            Hyperparameter("kernel__" + hp.name, hp.value_type, hp.bounds, hp.n_elements)
            for hp in self.kernel.hyperparameters
        ]

    @property
    def theta(self) -> np.ndarray:
        # The (log) tunable params are exactly the inner kernel's.
        return self.kernel.theta

    @theta.setter
    def theta(self, theta: np.ndarray) -> None:
        # Pushing theta down to the inner kernel is what lets the optimizer tune it.
        self.kernel.theta = theta

    @property
    def bounds(self) -> np.ndarray:
        # Optimization bounds also come straight from the inner kernel.
        return self.kernel.bounds

    def __call__(self, X, Y=None, eval_gradient: bool = False):
        # Slice both inputs to this kernel's column block, then delegate. The
        # gradient w.r.t. the inner hyperparameters is unchanged by slicing.
        Xs = X[:, self.start : self.stop]
        Ys = None if Y is None else Y[:, self.start : self.stop]
        return self.kernel(Xs, Ys, eval_gradient=eval_gradient)

    def diag(self, X):
        # Diagonal (self-similarity) on the sliced columns.
        return self.kernel.diag(X[:, self.start : self.stop])

    def is_stationary(self) -> bool:
        # Stationarity is a property of the inner kernel (RBF is stationary).
        return self.kernel.is_stationary()


def _composite_kernel(n_emb: int, n_total: int) -> Kernel:
    """cosine-RBF on the embedding block + RBF on the tabular block + noise."""
    # Embedding kernel: amplitude * RBF over the unit-sphere geometry of the text.
    text = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e2))
    # Tabular kernel: amplitude * RBF over the standardized numeric features.
    tab = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(length_scale=1.0, length_scale_bounds=(1e-1, 1e2))
    # White noise: the irreducible project-to-project scatter (the σ the gate hit).
    noise = WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-3, 1e1))
    # Sum the column-restricted kernels: text sees [0:n_emb], tabular sees the rest.
    return OnColumns(text, 0, n_emb) + OnColumns(tab, n_emb, n_total) + noise


class CompositeKernelGP:
    """Log-effort GP with a cosine(text)+RBF(tabular) composite kernel.

    Mirrors LogGaussianProcess (log-space target, p50 = exp(mean), predict_log for
    the conformal machinery) but (a) uses the composite kernel and (b) standardizes
    ONLY the tabular block — embeddings are already unit-normalized and scaling
    them would destroy the cosine geometry the text kernel relies on.

    X must be laid out as [embedding columns | tabular columns] with the embedding
    width given as `n_emb`.
    """

    def __init__(self, n_emb: int, seed: int = 0) -> None:
        # Number of leading embedding columns (the boundary in the composite kernel).
        self.n_emb = n_emb
        # Scaler for the tabular tail only; fit on train, applied at predict time.
        self._scaler = StandardScaler()
        self._seed = seed
        # Built in fit() once the total column count is known.
        self._gp: GaussianProcessRegressor | None = None

    def _transform(self, X: np.ndarray, *, fit: bool) -> np.ndarray:
        # Split into the embedding block (left untouched) and the tabular block.
        X = np.asarray(X, dtype=float)
        emb, tab = X[:, : self.n_emb], X[:, self.n_emb :]
        # Standardize the tabular block only; embeddings keep their unit norm.
        tab = self._scaler.fit_transform(tab) if fit else self._scaler.transform(tab)
        # Re-concatenate so the composite kernel's column slices line up.
        return np.column_stack([emb, tab])

    def fit(self, X: np.ndarray, effort: np.ndarray) -> "CompositeKernelGP":
        effort = np.asarray(effort, dtype=float)
        # log-space modeling requires strictly positive effort.
        if np.any(effort <= 0):
            raise ValueError("effort must be positive for log-space modeling")
        # Build the kernel now that the total feature width is known.
        n_total = np.asarray(X).shape[1]
        self._gp = GaussianProcessRegressor(
            kernel=_composite_kernel(self.n_emb, n_total),
            normalize_y=True,           # center the log-targets
            n_restarts_optimizer=4,     # escape poor local optima in the marginal lik.
            random_state=self._seed,
        )
        # Fit on the transformed features against log-effort.
        self._gp.fit(self._transform(X, fit=True), np.log(effort))
        return self

    def predict_log(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Posterior mean and std in log space; std feeds the Gaussian quantiles
        # that CQR later corrects into calibrated hour-intervals.
        if self._gp is None:
            raise RuntimeError("fit() must be called before predict_log()")
        mean, std = self._gp.predict(self._transform(X, fit=False), return_std=True)
        return mean, std

    def predict(self, X: np.ndarray) -> np.ndarray:
        # Point estimate (median effort): exp of the log-space mean.
        mean, _ = self.predict_log(X)
        return np.exp(mean)
