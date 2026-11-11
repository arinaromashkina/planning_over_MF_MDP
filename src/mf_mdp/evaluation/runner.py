"""Shared evaluation loop used by reproduction runs and HP sweeps."""

from __future__ import annotations

import time

import numpy as np

from mf_mdp.data.movielens import UserSplit
from mf_mdp.evaluation.metrics import ndcg_at_k, recall_at_k
from mf_mdp.mf import IALSConfig
from mf_mdp.mf import user_state
from mf_mdp.planning import PlannerConfig, plan


def holdout_items(sp: UserSplit, holdout: str) -> set[int]:
    items = sp.valid if holdout == "valid" else sp.test
    return set(int(i) for i in items)


def evaluate_method(
    V: np.ndarray,
    splits: list[UserSplit],
    k: int,
    pcfg: PlannerConfig,
    rng: np.random.Generator,
    ials_cfg: IALSConfig,
    holdout: str = "test",
) -> dict:
    recalls, ndcgs, latencies = [], [], []
    for sp in splits:
        state = user_state(V, sp.prefix, ials_cfg)
        t0 = time.perf_counter()
        recs = plan(V, state, k, pcfg, rng)
        latencies.append(time.perf_counter() - t0)
        relevant = holdout_items(sp, holdout)
        recalls.append(recall_at_k(recs, relevant, k))
        ndcgs.append(ndcg_at_k(recs, relevant, k))
    return {
        "recall_at_k": float(np.mean(recalls)),
        "ndcg_at_k": float(np.mean(ndcgs)),
        "recall_std": float(np.std(recalls)),
        "ndcg_std": float(np.std(ndcgs)),
        "latency_mean_s": float(np.mean(latencies)),
        "latency_p95_s": float(np.quantile(latencies, 0.95)),
        "n_users": len(splits),
    }
