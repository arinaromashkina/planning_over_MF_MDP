"""Grounded cross-features for the H4 response-model ladder.

Owner's interaction list (docs/H4_CROSS_RESPONSE_MODEL_DESIGN.md §2.3) and their
operationalization per testbed. All builders are pure functions of decision-time
quantities; the leakage contract of models/features.py is unchanged.

Feature layout (BASE, 10 dims, features_for order):
    0 cos        1 dot        2 leverage  3 |delta_s|  4 ||v||
    5 log_pop    6 hist_len   7 tr(P)     8 ||P||_F    9 ||u||
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

BASE_NAMES = ["cos", "dot", "leverage", "|delta_s|", "||v||", "log_pop",
              "hist_len", "tr(P)", "||P||_F", "||u||"]
N_BASE = 10

ENVC_HIST_NAMES = ["mean_eng_hist", "repeat_cnt"]
KUAI_HIST_NAMES = ["mean_wr_hist", "log1p_recency", "repeat_cnt",
                   "author_repeat_cnt", "hour", "log_duration", "video_type",
                   "session_pos"]

ENVC_CROSS_NAMES = ["cos_x_hist_len", "mean_eng_x_cos", "leverage_x_log_pop",
                    "repeat_x_cos"]
KUAI_CROSS_NAMES = ["cos_x_hist_len", "cos_x_recency_decay", "mean_wr_x_cos",
                    "leverage_x_log_pop", "repeat_x_cos", "author_repeat_x_cos"]

RECENCY_TAU_MIN = 30.0  # decay constant for interaction #2 (documented)


def envc_crosses(base: np.ndarray, mean_eng_hist: float, repeat_cnt: float) -> np.ndarray:
    """Env-C grounded crosses (4). Interaction #2 (recency) is infeasible in
    discrete-step Env-C — documented in the design doc; #6 is structurally 0."""
    cos, leverage, log_pop, hist_len = base[0], base[2], base[5], base[6]
    return np.array([
        cos * hist_len,
        mean_eng_hist * cos,
        leverage * log_pop,
        repeat_cnt * cos,
    ])


def kuairand_crosses(base: np.ndarray, mean_wr_hist: float, recency_min: float,
                     repeat_cnt: float, author_repeat_cnt: float) -> np.ndarray:
    """KuaiRand grounded crosses (6) — the full owner list operationalized."""
    cos, leverage, log_pop, hist_len = base[0], base[2], base[5], base[6]
    return np.array([
        cos * hist_len,
        cos * float(np.exp(-recency_min / RECENCY_TAU_MIN)),
        mean_wr_hist * cos,
        leverage * log_pop,
        repeat_cnt * cos,
        author_repeat_cnt * cos,
    ])


@dataclass
class RandomCrosses:
    """Ungrounded control: K products of uniformly drawn distinct feature pairs.

    Pairs are frozen at fit time (fixed seed) and reused at transform time, so
    train/test see exactly the same random cross definitions.
    """

    k: int
    seed: int = 0
    pairs: list[tuple[int, int]] = field(default_factory=list, init=False)

    def fit(self, X: np.ndarray) -> "RandomCrosses":
        d = X.shape[1]
        rng = np.random.default_rng(self.seed)
        seen = set()
        while len(self.pairs) < self.k:
            i, j = int(rng.integers(d)), int(rng.integers(d))
            if i == j or (min(i, j), max(i, j)) in seen:
                continue
            seen.add((min(i, j), max(i, j)))
            self.pairs.append((i, j))
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        if not self.pairs:
            raise RuntimeError("RandomCrosses not fitted")
        X = np.asarray(X, dtype=float)
        return np.column_stack([X[:, i] * X[:, j] for i, j in self.pairs])

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        return self.fit(X).transform(X)
