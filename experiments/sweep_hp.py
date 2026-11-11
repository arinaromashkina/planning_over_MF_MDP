"""Hyperparameter sweep for the MF-MDP reproduction on the validation split.

Selects MF hyperparameters (dim, alpha, lam) and planner eta on validation only,
per the paper's protocol. Prints a table and writes a CSV under MFMDP_OUTPUT_ROOT.

Usage:
    python experiments/sweep_hp.py --n-users 1500
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mf_mdp.data.movielens import build_lln
from mf_mdp.evaluation.runner import evaluate_method
from mf_mdp.mf import IALSConfig, train_ials
from mf_mdp.mdp import fold_in
from mf_mdp.planning import PlannerConfig
from mf_mdp.planning.candidates import exact_cosine_topl


def leverage_stats(V, splits, ials_cfg, n_users=100, pool=50):
    meds = []
    for sp in splits[:n_users]:
        from mf_mdp.mf import user_state

        st = user_state(V, sp.prefix, ials_cfg)
        cand = exact_cosine_topl(V, st.u, st.consumed, pool)
        ls = [float(V[a] @ (st.P @ V[a])) for a in cand]
        meds.append(np.median(ls))
    return float(np.median(meds))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-users", type=int, default=1500)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", default=None)
    parser.add_argument("--dims", default="32,64,128")
    parser.add_argument("--alphas", default="1,10")
    parser.add_argument("--lams", default="0.01,0.1,1.0")
    parser.add_argument("--etas", default="0.1,1,10")
    parser.add_argument("--n-iter", type=int, default=15)
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    dims = [int(x) for x in args.dims.split(",")]
    alphas = [float(x) for x in args.alphas.split(",")]
    lams = [float(x) for x in args.lams.split(",")]
    etas = [float(x) for x in args.etas.split(",")]
    n_iter = args.n_iter

    data = build_lln(min_rating=4, n_test=1, n_valid=1)
    splits = data.splits[: args.n_users]
    print(f"validation on {len(splits)} users")

    rows = []
    for dim, alpha, lam in itertools.product(dims, alphas, lams):
        ials_cfg = IALSConfig(dim=dim, alpha=alpha, lam=lam, n_iter=n_iter)
        t0 = time.perf_counter()
        U, V = train_ials(data.positives_train, data.n_users, data.n_items, ials_cfg, seed=args.seed)
        fit_s = time.perf_counter() - t0
        lev = leverage_stats(V, splits, ials_cfg)
        base = dict(dim=dim, alpha=alpha, lam=lam, n_iter=n_iter, fit_s=round(fit_s, 1), lev_med=lev)
        print(f"[mf] d={dim} a={alpha} l={lam} fit={fit_s:.0f}s lev={lev:.2e}", flush=True)
        methods = [("static_dot", None), ("static_cos", None)]
        methods += [(f"plan1_eta{e}", ("plan1", e)) for e in etas]
        for name, spec in methods:
            if spec is None:
                pcfg = PlannerConfig(method=name)
            else:
                pcfg = PlannerConfig(method="plan1", eta=spec[1], plan1_variant="post")
            res = evaluate_method(V, splits, args.k, pcfg, np.random.default_rng(args.seed), ials_cfg, holdout="valid")
            row = {**base, "method": name, **res}
            rows.append(row)
            print(f"  {name:16s} recall={res['recall_at_k']:.4f} ndcg={res['ndcg_at_k']:.4f}", flush=True)

    out_root = Path(os.environ.get("MFMDP_OUTPUT_ROOT", "./outputs"))
    out_dir = out_root / "sweeps"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = Path(args.out) if args.out else out_dir / f"mf_planer_sweep{args.tag}_seed{args.seed}.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    with open(out_path.with_suffix(".json"), "w") as f:
        json.dump({"grid": {"dims": dims, "alphas": alphas, "lams": lams, "etas": etas, "n_iter": n_iter}}, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
