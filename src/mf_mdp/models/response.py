"""Response model: logistic regression with L2 (numpy implementation).

p_phi(y=1 | features) for binary feedback. Labels and their semantics are the caller's
responsibility (see docs/PHASE4_EXECUTION_PLAN.md §1 — ML-1M supports only consumed
rating-4/5 events; no exposure modeling).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LogisticResponseModel:
    l2: float = 1e-3
    lr: float = 0.5
    n_steps: int = 500
    weights: np.ndarray | None = field(default=None, init=False)
    bias: float = field(default=0.0, init=False)

    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        n, d = X.shape
        w = np.zeros(d)
        b = 0.0
        y = y.astype(float)
        for _ in range(self.n_steps):
            z = X @ w + b
            p = 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))
            g = (p - y) / n
            grad_w = X.T @ g + self.l2 * w
            grad_b = float(g.sum())
            w -= self.lr * grad_w
            b -= self.lr * grad_b
        self.weights = w
        self.bias = b

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("model not fitted")
        z = X @ self.weights + self.bias
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))


def log_loss(y: np.ndarray, p: np.ndarray, eps: float = 1e-12) -> float:
    p = np.clip(p, eps, 1 - eps)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def brier_score(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y) ** 2))


def roc_auc(y: np.ndarray, p: np.ndarray) -> float:
    """Rank-based AUC (Mann-Whitney), no sklearn dependency."""
    order = np.argsort(p)
    ranks = np.empty(len(p), dtype=float)
    ranks[order] = np.arange(1, len(p) + 1)
    # average ranks for ties
    sorted_p = p[order]
    i = 0
    while i < len(p):
        j = i
        while j + 1 < len(p) and sorted_p[j + 1] == sorted_p[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    pos = y == 1
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def expected_calibration_error(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (p >= lo) & (p < hi if i < n_bins - 1 else p <= hi)
        if m.sum() > 0:
            ece += float(m.mean()) * abs(float(y[m].mean()) - float(p[m].mean()))
    return ece
