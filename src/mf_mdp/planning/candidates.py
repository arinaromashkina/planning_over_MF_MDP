"""Candidate generation for planning.

The paper restricts MCTS expansion to the l items nearest the current user vector via an
ANN index (Wolpertinger-style action reduction). Our catalogs are <= ~5k items, so exact
top-l by cosine similarity via one matrix-vector product is used instead of a real ANN
index. Documented in docs/REPRODUCTION.md.
"""

from __future__ import annotations

import numpy as np


def exact_cosine_topl(
    V: np.ndarray,
    u: np.ndarray,
    consumed: tuple[int, ...],
    l: int,
) -> np.ndarray:
    """Indices of the l unconsumed items with highest cosine similarity to u."""
    norms = np.linalg.norm(V, axis=1)
    norms = np.where(norms == 0.0, 1.0, norms)
    u_norm = np.linalg.norm(u)
    if u_norm == 0.0:
        scores = np.zeros(V.shape[0])
    else:
        scores = (V @ u) / (norms * u_norm)
    if consumed:
        scores[np.asarray(consumed, dtype=np.int64)] = -np.inf
    l = min(l, V.shape[0])
    part = np.argpartition(-scores, l - 1)[:l]
    return part[np.argsort(-scores[part], kind="stable")]


def catalog_scores(
    V: np.ndarray,
    u: np.ndarray,
    consumed: tuple[int, ...],
    kind: str,
) -> np.ndarray:
    """Score every catalog item; consumed items get -inf."""
    if kind == "dot":
        scores = V @ u
    elif kind == "cos":
        norms = np.linalg.norm(V, axis=1)
        norms = np.where(norms == 0.0, 1.0, norms)
        u_norm = np.linalg.norm(u)
        scores = (V @ u) / norms if u_norm == 0.0 else (V @ u) / (norms * u_norm)
    else:
        raise ValueError(f"unknown similarity kind: {kind}")
    if consumed:
        scores = scores.copy()
        scores[np.asarray(consumed, dtype=np.int64)] = -np.inf
    return scores
