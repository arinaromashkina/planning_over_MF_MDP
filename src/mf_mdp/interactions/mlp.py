"""Small MLP classifier for binary response (NumPy).

One hidden layer (tanh) + sigmoid output, log-loss with minibatch SGD. Interface
matches models/response.py:LogisticResponseModel. Used by the H4 cross-response
experiments (docs/H4_CROSS_RESPONSE_MODEL_DESIGN.md); the M3' arm additionally
appends raw u-v elementwise-product dims to the input (interaction #5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -30.0, 30.0)))


@dataclass
class MLPClassifier:
    """Binary MLP: x -> tanh(x W1 + b1) -> sigmoid(a1 w2 + b2)."""

    hidden: int = 32
    l2: float = 1e-4
    lr: float = 0.05
    n_epochs: int = 20
    batch_size: int = 8192
    seed: int = 0
    W1: np.ndarray | None = field(default=None, init=False)
    b1: np.ndarray | None = field(default=None, init=False)
    w2: np.ndarray | None = field(default=None, init=False)
    b2: float = field(default=0.0, init=False)

    def _forward(self, X: np.ndarray):
        assert self.W1 is not None and self.b1 is not None and self.w2 is not None
        z1 = X @ self.W1 + self.b1
        a1 = np.tanh(z1)
        p = _sigmoid(a1 @ self.w2 + self.b2)
        return z1, a1, p

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.W1 is None:
            raise RuntimeError("model not fitted")
        _, _, p = self._forward(np.asarray(X, dtype=float))
        return p

    def _grads(self, X: np.ndarray, y: np.ndarray):
        n = X.shape[0]
        z1, a1, p = self._forward(X)
        dz2 = (p - y) / n
        dw2 = a1.T @ dz2 + self.l2 * self.w2
        db2 = float(dz2.sum())
        dz1 = (dz2[:, None] * self.w2[None, :]) * (1.0 - a1 ** 2)
        dW1 = X.T @ dz1 + self.l2 * self.W1
        db1 = dz1.sum(axis=0)
        return dW1, db1, dw2, db2

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n, d = X.shape
        rng = np.random.default_rng(self.seed)
        self.W1 = rng.normal(0.0, np.sqrt(1.0 / max(d, 1)), (d, self.hidden))
        self.b1 = np.zeros(self.hidden)
        self.w2 = rng.normal(0.0, np.sqrt(1.0 / self.hidden), self.hidden)
        self.b2 = 0.0
        for _ in range(self.n_epochs):
            order = rng.permutation(n)
            for s in range(0, n, self.batch_size):
                idx = order[s:s + self.batch_size]
                dW1, db1, dw2, db2 = self._grads(X[idx], y[idx])
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
                self.w2 -= self.lr * dw2
                self.b2 -= self.lr * db2
