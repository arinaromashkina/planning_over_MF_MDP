"""Second-order Factorization Machine for binary response (NumPy).

Log-loss optimization with minibatch SGD. The pairwise term uses the standard
factorization trick:

    sum_{i<j} x_i x_j <V_i, V_j>
      = 0.5 * sum_f [ (sum_i x_i V_if)^2 - sum_i x_i^2 V_if^2 ]

Interface matches models/response.py:LogisticResponseModel (fit / predict_proba)
so the FM plugs into the same FeatureEncoder normalization pipeline. Used by the
H4 cross-response experiments (docs/H4_CROSS_RESPONSE_MODEL_DESIGN.md).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


@dataclass
class FactorizationMachine:
    """Binary FM: p(y=1|x) = sigmoid(w0 + w'x + sum_{i<j} x_i x_j <V_i, V_j>)."""

    k: int = 8
    l2: float = 1e-4
    lr: float = 0.05
    n_epochs: int = 20
    batch_size: int = 8192
    seed: int = 0
    w0: float = field(default=0.0, init=False)
    w: np.ndarray | None = field(default=None, init=False)
    V: np.ndarray | None = field(default=None, init=False)

    def _pair_term(self, X: np.ndarray) -> np.ndarray:
        """Vectorized pairwise interaction term, shape (n,)."""
        assert self.V is not None
        E = X @ self.V  # (n, k)
        return 0.5 * ((E ** 2).sum(axis=1) - ((X ** 2) @ (self.V ** 2)).sum(axis=1))

    def _logit(self, X: np.ndarray) -> np.ndarray:
        assert self.w is not None
        return self.w0 + X @ self.w + self._pair_term(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.w is None or self.V is None:
            raise RuntimeError("model not fitted")
        return _sigmoid(self._logit(np.asarray(X, dtype=float)))

    def _grads(self, X: np.ndarray, y: np.ndarray):
        """Gradients of the mean log-loss at the current parameters."""
        assert self.w is not None and self.V is not None
        n = X.shape[0]
        E = X @ self.V
        p = _sigmoid(self.w0 + X @ self.w + 0.5 * ((E ** 2).sum(1) - ((X ** 2) @ (self.V ** 2)).sum(1)))
        g = (p - y) / n
        dw0 = float(g.sum())
        dw = X.T @ g + self.l2 * self.w
        gX = g[:, None] * X
        # dV_ij = sum_n g_n x_ni E_nj  -  (sum_n g_n x_ni^2) V_ij
        s2 = (g[:, None] * (X ** 2)).sum(axis=0)
        dV = gX.T @ E - s2[:, None] * self.V + self.l2 * self.V
        return dw0, dw, dV

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        self.w0 = 0.0
        self.w = np.zeros(d)
        self.V = rng.normal(0.0, 0.01, (d, self.k))
        for _ in range(self.n_epochs):
            order = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                idx = order[s:s + self.batch_size]
                dw0, dw, dV = self._grads(X[idx], y[idx])
                self.w0 -= self.lr * dw0
                self.w -= self.lr * dw
                self.V -= self.lr * dV
