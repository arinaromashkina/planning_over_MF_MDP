"""H2-E2 grid runner: scale x dynamics boundary for the hybrid sequential state.

Cells (frozen protocol, docs/H2_HYBRID_SEQUENTIAL_STATE_DESIGN.md Part 7 follow-up):
    pilot_ref       n=400,  alpha_pos=0.35 (default), q est   (pilot parity)
    scale2000_a010  n=2000, alpha_pos=0.1,  q est
    scale2000_a035  n=2000, alpha_pos=0.35, q est
    scale2000_a060  n=2000, alpha_pos=0.6,  q est
    einit2000_a035  n=2000, alpha_pos=0.35, q est, + E-init-from-iALS arms
    ctrl2000_a035   n=2000, alpha_pos=0.35, q est, + negative controls
    scale5000_a035  n=5000, alpha_pos=0.35, q est
    qtrue_pilot     n=400,  alpha_pos=0.35, q TRUE (isolates the stoch-opt flip)

Each (cell, seed) job is an independent process (ProcessPoolExecutor, 6 workers);
per-job JSONs land in outputs/h2/grid/, then a summary across seeds is aggregated.

Usage: python experiments/h2/run_h2_grid.py [--workers 6] [--seeds 5]
          [--only cell1,cell2] [--n-eval 200] [--epochs 150] [--smoke]
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import json
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_h2_factorial import run_seed  # noqa: E402

CELLS = {
    "pilot_ref":      dict(n_train=400,  alpha_pos=None, q_mode="est",  controls=False, einit=False),
    "scale2000_a010": dict(n_train=2000, alpha_pos=0.1,  q_mode="est",  controls=False, einit=False),
    "scale2000_a035": dict(n_train=2000, alpha_pos=0.35, q_mode="est",  controls=False, einit=False),
    "scale2000_a060": dict(n_train=2000, alpha_pos=0.6,  q_mode="est",  controls=False, einit=False),
    "einit2000_a035": dict(n_train=2000, alpha_pos=0.35, q_mode="est",  controls=False, einit=True),
    "ctrl2000_a035":  dict(n_train=2000, alpha_pos=0.35, q_mode="est",  controls=True,  einit=False),
    "scale5000_a035": dict(n_train=5000, alpha_pos=0.35, q_mode="est",  controls=False, einit=False),
    "qtrue_pilot":    dict(n_train=400,  alpha_pos=None, q_mode="true", controls=False, einit=False),
}


def out_dir() -> Path:
    d = Path(os.environ.get("MFMDP_OUTPUT_ROOT", "./outputs")) / "h2" / "grid"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_job(cell_name: str, seed: int, cell: dict, n_eval: int, epochs: int) -> str:
    path = out_dir() / f"{cell_name}__seed{seed}.json"
    if path.exists():
        return f"skip {path.name}"
    t0 = time.time()
    try:
        res = run_seed(seed, cell["n_train"], n_eval, epochs, cell["controls"],
                       alpha_pos=cell["alpha_pos"], q_mode=cell["q_mode"],
                       einit=cell["einit"])
        payload = {"cell": cell_name, "seed": seed, "cell_cfg": cell,
                   "n_eval": n_eval, "epochs": epochs, "ok": True,
                   "runtime_s": time.time() - t0, "result": res}
    except Exception as e:  # keep the grid going; error is recorded
        payload = {"cell": cell_name, "seed": seed, "cell_cfg": cell, "ok": False,
                   "runtime_s": time.time() - t0,
                   "error": f"{type(e).__name__}: {e}", "trace": traceback.format_exc()}
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    return f"done {path.name} ({time.time() - t0:.0f}s)"


def bootstrap_ci(diff, rng, n=2000):
    boots = [np.mean(rng.choice(diff, len(diff))) for _ in range(n)]
    return [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]


def aggregate(cells, seeds):
    rng = np.random.default_rng(0)
    summary = {}
    for name, cell in cells.items():
        per_seed = []
        for s in seeds:
            p = out_dir() / f"{name}__seed{s}.json"
            if p.exists():
                d = json.load(open(p))
                if d.get("ok"):
                    per_seed.append(d["result"])
        if not per_seed:
            summary[name] = {"error": "no successful seeds"}
            continue
        arms = [k for k in per_seed[0] if k != "predictive"]
        agg = {}
        for a in arms:
            vals = [r[a]["J_mean"] for r in per_seed]
            agg[a] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                      "se": float(np.std(vals) / np.sqrt(len(vals))), "n_seeds": len(vals)}
        contrasts = {}
        for tr in ("optimistic", "stochastic"):
            if f"hybrid+{tr}" in arms and f"mf+{tr}" in arms:
                d = [r[f"hybrid+{tr}"]["J_mean"] - r[f"mf+{tr}"]["J_mean"] for r in per_seed]
                contrasts[f"hybrid-mf@{tr}"] = {"mean": float(np.mean(d)), "ci95": bootstrap_ci(d, rng)}
        for arm in ("mf", "seq", "hybrid"):
            if f"{arm}+stochastic" in arms and f"{arm}+optimistic" in arms:
                d = [r[f"{arm}+stochastic"]["J_mean"] - r[f"{arm}+optimistic"]["J_mean"] for r in per_seed]
                contrasts[f"stoch-opt@{arm}"] = {"mean": float(np.mean(d)), "ci95": bootstrap_ci(d, rng)}
        for arm in ("seq", "hybrid"):
            key = f"{arm}+stochastic+einit"
            if key in arms and f"{arm}+stochastic" in arms:
                d = [r[key]["J_mean"] - r[f"{arm}+stochastic"]["J_mean"] for r in per_seed]
                contrasts[f"einit-clean@{arm}"] = {"mean": float(np.mean(d)), "ci95": bootstrap_ci(d, rng)}
            for c in ("shuffle", "no_response", "random_action"):
                key = f"{arm}+stochastic+{c}"
                if key in arms and f"{arm}+stochastic" in arms:
                    d = [r[key]["J_mean"] - r[f"{arm}+stochastic"]["J_mean"] for r in per_seed]
                    contrasts[f"control-{c}@{arm}"] = {"mean": float(np.mean(d)), "ci95": bootstrap_ci(d, rng)}
        summary[name] = {"cell_cfg": cell, "aggregate": agg, "contrasts": contrasts,
                         "predictive_seed0": per_seed[0]["predictive"]}
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-eval", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--only", type=str, default=None, help="comma-separated cell names")
    ap.add_argument("--smoke", action="store_true", help="tiny: 1 seed, 200 users, 30 epochs")
    args = ap.parse_args()
    cells = CELLS
    if args.only:
        cells = {k: v for k, v in CELLS.items() if k in args.only.split(",")}
    if args.smoke:
        args.seeds, args.n_eval, args.epochs = 1, 60, 30
    seeds = list(range(args.seeds))

    jobs = [(name, s, cell) for name, cell in cells.items() for s in seeds]
    t0 = time.time()
    print(f"[grid] {len(jobs)} jobs, {args.workers} workers", flush=True)
    with cf.ProcessPoolExecutor(max_workers=args.workers) as pool:
        futs = [pool.submit(run_job, name, s, cell, args.n_eval, args.epochs)
                for (name, s, cell) in jobs]
        for fut in cf.as_completed(futs):
            print(f"[grid] {fut.result()}  ({time.time() - t0:.0f}s elapsed)", flush=True)

    summary = aggregate(cells, seeds)
    path = out_dir() / "summary.json"
    with open(path, "w") as f:
        json.dump({"config": {"seeds": seeds, "n_eval": args.n_eval, "epochs": args.epochs},
                   "cells": summary}, f, indent=2)
    print(f"[grid] summary -> {path}  (total {time.time() - t0:.0f}s)")
    for name in cells:
        c = summary.get(name, {})
        agg = c.get("aggregate", {})
        top = sorted(agg.items(), key=lambda kv: -kv[1]["mean"])[:3]
        print(f"  {name}: " + ", ".join(f"{k}={v['mean']:.3f}" for k, v in top))


if __name__ == "__main__":
    main()
