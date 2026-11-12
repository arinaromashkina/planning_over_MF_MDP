"""E4: MCTS leaf-evaluator ablation — original vs zero vs v_logging_ridge x horizon.

Diagnostic experiment (NOT proof of policy improvement; offline evaluation limits apply).
Subset of test users, one seed — labeled as such.

Usage: python experiments/phase4/run_leaf_ablation.py [--n-users 300] [--seed 0]
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
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mf_mdp.data.movielens import build_lln
from mf_mdp.evaluation.runner import evaluate_method
from mf_mdp.mf import IALSConfig, train_ials
from mf_mdp.models.value import VLoggingRidge
from mf_mdp.planning import PlannerConfig


def load_v_logging(ckpt: Path, gamma: float) -> VLoggingRidge:
    from mf_mdp.models.features import FeatureEncoder

    z = np.load(ckpt)
    vm = VLoggingRidge(lam=float(z["lam"]), encoder=FeatureEncoder(mean=z["mean"], std=z["std"]),
                       horizon_features=bool(z.get("horizon_features", False)))
    vm.weights = z["weights"]
    return vm


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-users", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-simulations", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--variant", default="base", choices=["base", "horizon"],
                        help="V_logging checkpoint variant to load")
    args = parser.parse_args()
    t_start = time.time()

    data = build_lln(min_rating=4, n_test=1, n_valid=1)
    cfg = IALSConfig(dim=64, alpha=40.0, lam=1.0, n_iter=20)
    U, V = train_ials(data.positives_train, data.n_users, data.n_items, cfg, seed=args.seed)

    ckpt_dir = Path(os.environ.get("MFMDP_CHECKPOINT_ROOT", "./checkpoints"))
    ckpt = ckpt_dir / f"v_logging_ridge_{args.variant}_seed{args.seed}.npz"
    if not ckpt.exists():
        raise SystemExit(f"V_logging checkpoint not found: {ckpt} — run run_value_model.py first")
    vm = load_v_logging(ckpt, args.gamma)
    print(f"loaded {ckpt}", flush=True)

    splits = data.splits[: args.n_users]
    results = {"n_users": args.n_users, "seed": args.seed, "n_simulations": args.n_simulations,
               "gamma": args.gamma, "runs": []}
    grid = [("original", 10), ("zero", 10), ("v_logging_ridge", 10),
            ("original", 5), ("v_logging_ridge", 5),
            ("original", 1), ("v_logging_ridge", 1)]
    for leaf, horizon in grid:
        pcfg = PlannerConfig(
            method="plank", eta=1.0, gamma=args.gamma, c_uct=1.0,
            expansion_width=20, n_simulations=args.n_simulations, horizon=horizon,
            leaf_evaluator=leaf, value_model=vm if leaf == "v_logging_ridge" else None,
        )
        res = evaluate_method(V, splits, 10, pcfg, np.random.default_rng(args.seed), cfg, holdout="test")
        row = {"leaf_evaluator": leaf, "horizon": horizon,
               "recall": res["recall_at_k"], "ndcg": res["ndcg_at_k"],
               "latency_mean_s": res["latency_mean_s"], "latency_p95_s": res["latency_p95_s"],
               "leaf_evals_per_user": args.n_simulations}
        results["runs"].append(row)
        print(f"  {leaf:16s} h={horizon:2d}  R@10={res['recall_at_k']:.4f} "
              f"lat={res['latency_mean_s']*1e3:.0f}ms", flush=True)

    results["runtime_s"] = time.time() - t_start
    out_dir = Path(os.environ.get("MFMDP_OUTPUT_ROOT", "./outputs")) / "phase4"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"leaf_ablation_{args.variant}_seed{args.seed}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_dir / f'leaf_ablation_{args.variant}_seed{args.seed}.json'} ({time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
