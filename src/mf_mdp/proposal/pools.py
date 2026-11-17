"""Candidate pools C_t for pool-restricted planning (H3, docs/H3_LEARNED_ACTION_PROPOSALS_DESIGN.md §2.2).

All proposers exclude consumed items and return item-id arrays of length <= M
(`union` may return up to 2M by design). Pools are computed once per user at the
root; the MCTS tree searches inside C_t \\ consumed(path) (proposal/planning.py).
"""

from __future__ import annotations

import numpy as np

from mf_mdp.planning.candidates import exact_cosine_topl


def mf_pool(V: np.ndarray, u: np.ndarray, consumed: tuple[int, ...], m: int) -> np.ndarray:
    """Paper ANN step made exact (catalog ~3.4k): top-M unconsumed by cosine(u, e_i)."""
    return exact_cosine_topl(V, u, consumed, m)


def sasrec_pool(
    sasrec_logits: np.ndarray,
    user: int,
    consumed: tuple[int, ...],
    m: int,
) -> np.ndarray:
    """Top-M unconsumed items by the sequential model's next-item logits."""
    return sasrec_pool_from_scores(sasrec_logits[user], consumed, m)


def sasrec_pool_from_scores(
    scores_row: np.ndarray,
    consumed: tuple[int, ...],
    m: int,
) -> np.ndarray:
    """Row-wise variant (positional index-free): top-M unconsumed by next-item logits."""
    scores = np.asarray(scores_row, dtype=np.float64).copy()
    if consumed:
        scores[np.asarray(consumed, dtype=np.int64)] = -np.inf
    n_unconsumed = int(np.isfinite(scores).sum())
    if n_unconsumed == 0:
        return np.empty(0, dtype=np.int64)
    m = max(1, min(m, n_unconsumed))
    part = np.argpartition(-scores, m - 1)[:m]
    return part[np.argsort(-scores[part], kind="stable")].astype(np.int64)


def union_pool(pool_a: np.ndarray, pool_b: np.ndarray) -> np.ndarray:
    """Set union preserving order: all of pool_a (rank order), then pool_b-only items."""
    seen = set(int(a) for a in pool_a)
    extra = [int(b) for b in pool_b if int(b) not in seen]
    if extra:
        return np.concatenate([pool_a, np.asarray(extra, dtype=np.int64)])
    return np.asarray(pool_a, dtype=np.int64)


def mf_random_pool(
    V: np.ndarray,
    u: np.ndarray,
    consumed: tuple[int, ...],
    m: int,
    rho: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Top-ceil((1-rho)M) by MF cosine plus floor(rho*M) uniform over the remaining
    unconsumed catalog (owner's 'случайное расширение части пула для exploration')."""
    m_greedy = int(np.ceil((1.0 - rho) * m))
    m_rand = int(np.floor(rho * m))
    greedy = mf_pool(V, u, consumed, m_greedy)
    greedy_set = set(int(a) for a in greedy)
    consumed_arr = np.asarray(consumed, dtype=np.int64) if consumed else np.empty(0, dtype=np.int64)
    mask = np.ones(V.shape[0], dtype=bool)
    if consumed_arr.size:
        mask[consumed_arr] = False
    mask[np.asarray(sorted(greedy_set), dtype=np.int64)] = False
    rest = np.flatnonzero(mask)
    if m_rand > 0 and rest.size > 0:
        take = rng.choice(rest, size=min(m_rand, rest.size), replace=False)
        return np.concatenate([greedy, take.astype(np.int64)])
    return greedy


def coverage(pools: list[np.ndarray], targets: np.ndarray) -> float:
    """Fraction of users whose target item lies in their pool (Recall@M of the proposer)."""
    if len(pools) != len(targets):
        raise ValueError("pools/targets length mismatch")
    hit = sum(1 for pool, t in zip(pools, targets) if int(t) in set(int(a) for a in pool))
    return hit / max(len(targets), 1)


def oracle_coverage(pools: list[np.ndarray], oracle_actions: np.ndarray) -> float:
    """Fraction of users whose planner-oracle action (full-catalog Plan-K first action)
    survived the filter: P(oracle in C_t) — the candidate-bottleneck quantity."""
    return coverage(pools, oracle_actions)
