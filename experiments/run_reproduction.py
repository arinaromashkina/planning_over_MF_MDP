"""Run reproduction experiments for arXiv:2607.02115v2.

Usage:
    python experiments/run_reproduction.py --config configs/reproduction/movielens_lln.yaml

Writes metrics.json + environment.json under MFMDP_OUTPUT_ROOT.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mf_mdp.data.movielens import build_lln
from mf_mdp.evaluation.runner import evaluate_method
from mf_mdp.mf import IALSConfig, train_ials
from mf_mdp.planning import PlannerConfig


def load_config(path: str) -> dict:
    with open(path) as f:
        cfg = yaml.safe_load(f)
    if "extends" in cfg:
        parent_path = str(Path(path).parent / cfg.pop("extends"))
        parent = load_config(parent_path)
        merged = {**parent, **cfg}
        for key in ("dataset", "mf", "split", "eval", "planner", "run"):
            if key in parent or key in cfg:
                merged[key] = {**parent.get(key, {}), **cfg.get(key, {})}
        return merged
    return cfg


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=Path(__file__).resolve().parent.parent
        ).decode().strip()
    except Exception:
        return "unknown"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--methods", default=None, help="comma-separated override of run.methods")
    parser.add_argument("--holdout", default="test", choices=["test", "valid"])
    parser.add_argument("--limit-users", type=int, default=None, help="debug: cap number of users")
    parser.add_argument("--seed", type=int, default=None, help="override config seed")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
    seed = int(cfg.get("seed", 0))
    k = int(cfg["eval"]["k"])
    ials_cfg = IALSConfig(**cfg["mf"])

    run = cfg.get("run", {})
    methods = (args.methods or run.get("methods") or cfg["planner"].get("method") or "plan1")
    methods = [m.strip() for m in methods.split(",")] if isinstance(methods, str) else list(methods)

    protocol = cfg["split"]["protocol"]
    assert protocol == "lln", "GTS not implemented yet"
    data = build_lln(
        min_rating=int(cfg["dataset"]["min_rating"]),
        n_test=int(cfg["split"]["n_test"]),
        n_valid=int(cfg["split"]["n_valid"]),
    )
    splits = data.splits[: args.limit_users] if args.limit_users else data.splits
    print(f"dataset: {data.n_users} users, {data.n_items} items, {len(splits)} users, "
          f"{len(data.positives_train)} train positives, holdout={args.holdout}")

    U, V = train_ials(data.positives_train, data.n_users, data.n_items, ials_cfg, seed=seed)

    results = {}
    for method in methods:
        pcfg = PlannerConfig(**{**cfg["planner"], "method": method})
        rng = np.random.default_rng(seed)
        t0 = time.perf_counter()
        results[method] = evaluate_method(V, splits, k, pcfg, rng, ials_cfg, holdout=args.holdout)
        results[method]["wall_time_s"] = time.perf_counter() - t0
        print(f"{method:12s} recall@{k}={results[method]['recall_at_k']:.4f} "
              f"ndcg@{k}={results[method]['ndcg_at_k']:.4f} "
              f"latency_mean={results[method]['latency_mean_s']*1e3:.1f}ms")

    out_root = Path(os.environ.get("MFMDP_OUTPUT_ROOT", "./outputs"))
    base = cfg.get("run_name", f"{cfg['dataset']['name']}_{protocol}")
    base = re.sub(r"_seed\d+$", "", str(base))
    if args.limit_users:
        base = f"{base}_subset{args.limit_users}"
    run_name = f"{base}_seed{seed}"
    out_dir = out_root / "reproduction" / run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    env = {
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "git_commit": git_commit(),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "n_cpu": os.cpu_count(),
    }
    with open(out_dir / "metrics.json", "w") as f:
        json.dump({"config": cfg, "seed": seed, "holdout": args.holdout, "methods": results}, f, indent=2)
    with open(out_dir / "environment.json", "w") as f:
        json.dump(env, f, indent=2)
    print(f"wrote {out_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
