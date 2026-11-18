"""H3 experiment runner: sequential model as action proposer (quality-compute trade-off).

Stages (docs/H3_LEARNED_ACTION_PROPOSALS_DESIGN.md §3):
    pools    gt/valid coverage of every pool x M on all test users (cheap)
    oracle   full-catalog Plan-K first action per user (subset; cached, reused)
    policy   restricted MCTS per arm x M: Recall@10, nDCG@10, J(pi), latency (subset)
    latency  quality vs n_sim per arm, fixed-latency iso-compute comparison (subset)

Usage:
    .venv/bin/python experiments/h3/run_proposals.py --stage pools
    .venv/bin/python experiments/h3/run_proposals.py --stage oracle --n-users 300
    .venv/bin/python experiments/h3/run_proposals.py --stage policy --n-users 1000 --ms 50,100
    .venv/bin/python experiments/h3/run_proposals.py --stage latency --n-users 300 --nsims 25,50,100,200

Writes JSON under MFMDP_OUTPUT_ROOT/h3/.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from mf_mdp.data.movielens import Dataset, build_lln
from mf_mdp.evaluation.metrics import ndcg_at_k, recall_at_k
from mf_mdp.mf import IALSConfig, train_ials, user_state
from mf_mdp.planning.planners import PlannerConfig, planK_topk
from mf_mdp.proposal.planning import planK_topk_pool, trajectory_return
from mf_mdp.proposal.pools import (
    coverage,
    mf_pool,
    mf_random_pool,
    oracle_coverage,
    sasrec_pool_from_scores,
    union_pool,
)

MF_CFG = IALSConfig(dim=64, alpha=40.0, lam=1.0, n_iter=20, init_scale=0.05)
PLANNER = PlannerConfig(
    method="plank", eta=1.0, gamma=0.9, c_uct=1.0,
    expansion_width=20, n_simulations=100, horizon=10,
)
IALS_CACHE = Path("checkpoints/ials_ml1m_d64a40l1i20_seed0.npz")


def out_dir() -> Path:
    d = Path(os.environ.get("MFMDP_OUTPUT_ROOT", "./outputs")) / "h3"
    d.mkdir(parents=True, exist_ok=True)
    return d


def load_data_and_V(seed: int = 0) -> tuple[Dataset, np.ndarray, IALSConfig]:
    data = build_lln(min_rating=4, n_test=1, n_valid=1)
    if IALS_CACHE.exists():
        z = np.load(IALS_CACHE)
        V = z["V"]
    else:
        _, V = train_ials(data.positives_train, data.n_users, data.n_items, MF_CFG, seed=seed)
        IALS_CACHE.parent.mkdir(exist_ok=True)
        np.savez(IALS_CACHE, V=V)
    return data, V, MF_CFG


def load_sasrec_logits(
    data: Dataset,
    splits,
    device: str,
    seed: int = 0,
    npz_path: str | None = None,
) -> np.ndarray:
    """Next-item logits aligned to `splits` (positional: runners use prefix slices).

    Two sources: (1) npz cache 'logits' computed on the FULL data.splits order —
    server-friendly, no torch needed; (2) the SASRec checkpoint scored locally.
    """
    if npz_path:
        arr = np.load(npz_path)["logits"]
        if arr.shape[0] < len(splits):
            raise ValueError(f"{npz_path}: {arr.shape[0]} rows < {len(splits)} splits")
        return arr[: len(splits)]
    from mf_mdp.proposal.sasrec import SASRec, SASRecConfig, score_histories, build_seq_array
    import torch

    ckpt = Path(f"checkpoints/sasrec_ml1m_seed{seed}.pt")
    if not ckpt.exists():
        raise SystemExit(f"missing {ckpt}; run experiments/h3/train_sasrec.py first")
    cfg = SASRecConfig(n_items=data.n_items)
    model = SASRec(cfg)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    seqs = build_seq_array([sp.prefix for sp in splits], data.n_items, cfg.maxlen)
    return score_histories(model, seqs, device=device)


def stage_export_logits(args) -> None:
    """Score the FULL splits list with the local checkpoint; save the npz cache."""
    data, _, _ = load_data_and_V()
    logits = load_sasrec_logits(data, data.splits, args.device, seed=args.seed)
    out = Path(args.logits_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, logits=logits.astype(np.float32), n_items=data.n_items)
    print(f"wrote {out} shape={logits.shape} dtype=float32")


def build_pool(arm: str, m: int, V, state, sas_row, rng, rho: float = 0.25) -> np.ndarray:
    """sas_row: this user's next-item logits (positional — split lists may skip user ids)."""
    if arm == "mf_ann":
        return mf_pool(V, state.u, state.consumed, m)
    if arm == "sasrec":
        return sasrec_pool_from_scores(sas_row, state.consumed, m)
    if arm == "union":
        return union_pool(mf_pool(V, state.u, state.consumed, m),
                          sasrec_pool_from_scores(sas_row, state.consumed, m))
    if arm == "mf_rand":
        return mf_random_pool(V, state.u, state.consumed, m, rho, rng)
    raise ValueError(arm)


def stage_pools(args) -> None:
    data, V, ials_cfg = load_data_and_V()
    splits = data.splits
    sas_logits = load_sasrec_logits(data, splits, args.device, npz_path=args.logits_npz)
    rng = np.random.default_rng(args.seed)
    ms = [int(x) for x in args.ms.split(",")]
    arms = ["mf_ann", "sasrec", "union", "mf_rand_rho0.10", "mf_rand_rho0.25"]
    gt = np.array([int(sp.test[0]) for sp in splits])
    va = np.array([int(sp.valid[0]) for sp in splits])
    results = {"ms": ms, "arms": {}}
    states = [user_state(V, sp.prefix, ials_cfg) for sp in splits]
    for arm in arms:
        per_m = {}
        for m in ms:
            pools, t0 = [], time.perf_counter()
            for idx, (sp, st) in enumerate(zip(splits, states)):
                if arm.startswith("mf_rand"):
                    rho = float(arm.rsplit("rho", 1)[1])
                    pool = mf_random_pool(V, st.u, st.consumed, m, rho, rng)
                elif arm == "mf_ann":
                    pool = mf_pool(V, st.u, st.consumed, m)
                elif arm == "sasrec":
                    pool = sasrec_pool_from_scores(sas_logits[idx], st.consumed, m)
                else:
                    pool = union_pool(mf_pool(V, st.u, st.consumed, m),
                                      sasrec_pool_from_scores(sas_logits[idx], st.consumed, m))
                pools.append(pool)
            per_m[m] = {
                "gt_coverage": coverage(pools, gt),
                "valid_coverage": coverage(pools, va),
                "mean_pool_size": float(np.mean([len(p) for p in pools])),
                "build_time_s": (time.perf_counter() - t0) / len(splits),
            }
            print(f"{arm} M={m}: gt={per_m[m]['gt_coverage']:.4f} "
                  f"valid={per_m[m]['valid_coverage']:.4f} size={per_m[m]['mean_pool_size']:.1f}")
        results["arms"][arm] = per_m
    _write("pools", results, args)


def stage_oracle(args) -> None:
    """Full-catalog Plan-K first action per user (the planner oracle). Cached."""
    cache = out_dir() / f"oracle_actions_n{args.n_users}_seed{args.seed}.json"
    if cache.exists() and not args.overwrite:
        print(f"reuse {cache}")
        return
    data, V, ials_cfg = load_data_and_V()
    rng = np.random.default_rng(args.seed)
    actions, lat = [], []
    for sp in data.splits[: args.n_users]:
        st = user_state(V, sp.prefix, ials_cfg)
        t0 = time.perf_counter()
        recs = planK_topk(V, st, 1, PLANNER, rng)
        lat.append(time.perf_counter() - t0)
        actions.append(int(recs[0]))
    payload = {"n_users": args.n_users, "oracle_actions": actions,
               "latency_mean_s": float(np.mean(lat)), "config": _cfg_dict()}
    with open(cache, "w") as f:
        json.dump(payload, f)
    print(f"oracle: {cache}; latency_mean_s={np.mean(lat):.3f}")


def _cfg_dict() -> dict:
    return {"eta": PLANNER.eta, "gamma": PLANNER.gamma, "c_uct": PLANNER.c_uct,
            "expansion_width": PLANNER.expansion_width, "n_simulations": PLANNER.n_simulations,
            "horizon": PLANNER.horizon, "mf": {"dim": 64, "alpha": 40.0, "lam": 1.0, "n_iter": 20}}


def stage_policy(args) -> None:
    data, V, ials_cfg = load_data_and_V()
    splits = data.splits[: args.n_users]
    sas_logits = load_sasrec_logits(data, data.splits, args.device, npz_path=args.logits_npz)[: args.n_users]
    rng = np.random.default_rng(args.seed)
    ms = [int(x) for x in args.ms.split(",")]
    arms = ["full"] + [a for a in args.arms.split(",") if a != "full"]
    results = {"n_users": len(splits), "ms": ms, "arms": {}}
    for arm in arms:
        for m in (ms if arm != "full" else [0]):
            recalls, ndcgs, js, lats = [], [], [], []
            t_start = time.perf_counter()
            for idx, sp in enumerate(splits):
                st = user_state(V, sp.prefix, ials_cfg)
                t0 = time.perf_counter()
                if arm == "full":
                    recs = planK_topk(V, st, args.k, PLANNER, rng)
                else:
                    pool = build_pool(arm, m, V, st, sas_logits[idx], rng)
                    recs = planK_topk_pool(V, st, args.k, PLANNER, rng, pool)
                lats.append(time.perf_counter() - t0)
                rel = {int(sp.test[0])}
                recalls.append(recall_at_k(recs, rel, args.k))
                ndcgs.append(ndcg_at_k(recs, rel, args.k))
                js.append(trajectory_return(V, st, recs, PLANNER.eta, PLANNER.sim, PLANNER.gamma))
            key = f"{arm}@M{m}" if arm != "full" else "full"
            results["arms"][key] = {
                "recall_at_10": float(np.mean(recalls)),
                "ndcg_at_10": float(np.mean(ndcgs)),
                "J_pi": float(np.mean(js)),
                "latency_mean_s": float(np.mean(lats)),
                "latency_p95_s": float(np.quantile(lats, 0.95)),
                "wall_time_s": time.perf_counter() - t_start,
            }
            print(key, json.dumps(results["arms"][key], indent=None))
    _write("policy", results, args)


def stage_latency(args) -> None:
    data, V, ials_cfg = load_data_and_V()
    splits = data.splits[: args.n_users]
    sas_logits = load_sasrec_logits(data, data.splits, args.device, npz_path=args.logits_npz)[: args.n_users]
    nsims = [int(x) for x in args.nsims.split(",")]
    arms = [a for a in args.arms.split(",")]
    m = args.m
    results = {"n_users": len(splits), "m": m, "nsims": nsims, "arms": {}}
    for arm in arms:
        per_n = {}
        for ns in nsims:
            cfg = PlannerConfig(**{**PLANNER.__dict__, "n_simulations": ns})
            rng = np.random.default_rng(args.seed)
            recalls, ndcgs, lats = [], [], []
            for idx, sp in enumerate(splits):
                st = user_state(V, sp.prefix, ials_cfg)
                t0 = time.perf_counter()
                if arm == "full":
                    recs = planK_topk(V, st, args.k, cfg, rng)
                else:
                    pool = build_pool(arm, m, V, st, sas_logits[idx], rng)
                    recs = planK_topk_pool(V, st, args.k, cfg, rng, pool)
                lats.append(time.perf_counter() - t0)
                recalls.append(recall_at_k(recs, {int(sp.test[0])}, args.k))
                ndcgs.append(ndcg_at_k(recs, {int(sp.test[0])}, args.k))
            per_n[ns] = {"recall_at_10": float(np.mean(recalls)),
                         "ndcg_at_10": float(np.mean(ndcgs)),
                         "latency_mean_s": float(np.mean(lats))}
            print(f"{arm} nsim={ns}: R@10={per_n[ns]['recall_at_10']:.4f} lat={per_n[ns]['latency_mean_s']:.3f}s")
        results["arms"][arm] = per_n
    _write("latency", results, args)


def _write(stage: str, results: dict, args) -> None:
    results["stage"] = stage
    results["config"] = {**vars(args), **_cfg_dict(), "seed": args.seed}
    path = out_dir() / f"h3_{stage}_seed{args.seed}.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["pools", "oracle", "policy", "latency", "export_logits"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps")
    ap.add_argument("--ms", default="25,50,100,200")
    ap.add_argument("--nsims", default="25,50,100,200")
    ap.add_argument("--arms", default="full,mf_ann,sasrec,union,mf_rand")
    ap.add_argument("--n-users", type=int, default=300)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--m", type=int, default=100, help="pool size for the latency stage")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--logits-npz", default=None, help="cached SASRec logits (full splits order)")
    ap.add_argument("--logits-out", default="checkpoints/sasrec_logits_ml1m_seed0.npz")
    args = ap.parse_args()
    {"pools": stage_pools, "oracle": stage_oracle,
     "policy": stage_policy, "latency": stage_latency,
     "export_logits": stage_export_logits}[args.stage](args)


if __name__ == "__main__":
    main()
