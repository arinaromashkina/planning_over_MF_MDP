"""Retrieval metrics: Recall@K and nDCG@K (binary relevance)."""

from __future__ import annotations


def _prepare(recs: list[int], relevant: set[int], k: int) -> list[int]:
    return recs[:k]


def recall_at_k(recs: list[int], relevant: set[int], k: int) -> float:
    """|top-K ∩ relevant| / min(|relevant|, K)."""
    if not relevant:
        return 0.0
    hits = sum(1 for x in recs[:k] if x in relevant)
    return hits / min(len(relevant), k)


def ndcg_at_k(recs: list[int], relevant: set[int], k: int) -> float:
    """Binary-gain DCG with log2 discount, normalized by the ideal DCG."""
    if not relevant:
        return 0.0
    dcg = 0.0
    for rank, item in enumerate(recs[:k]):
        if item in relevant:
            dcg += 1.0 / np_log2(rank + 2)
    ideal_len = min(len(relevant), k)
    idcg = sum(1.0 / np_log2(r + 2) for r in range(ideal_len))
    return dcg / idcg if idcg > 0 else 0.0


def np_log2(x: float) -> float:
    import math

    return math.log2(x)
