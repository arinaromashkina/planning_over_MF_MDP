"""E3: V_logging ridge — training, selection, evaluation on held-out users.

V_logging = logging-policy value estimate (NEVER V_target; see PHASE4_DESIGN.md §4.2).
Returns: normalized R0 rewards, gamma=0.9, last-30 decision points per user.

Usage: python experiments/phase4/run_value_model.py [--seed 0] [--max-t 30]
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

from common import split_users
from mf_mdp.data.movielens import build_lln
from mf_mdp.mf import IALSConfig, train_ials
from mf_mdp.models.value import VLoggingRidge, build_logged_trajectories


def metrics(G, pred):
    err = pred - G
    return {
        "n": int(len(G)),
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(np.mean(err**2))),
        "r2": float(1.0 - np.sum(err**2) / np.sum((G - G.mean()) ** 2)),
        "baseline_mean_mae": float(np.mean(np.abs(G - G.mean()))),
    }


def reliability(G, pred, n_bins=10):
    bins = np.linspace(min(G.min(), pred.min()), max(G.max(), pred.max()), n_bins + 1)
    out = []
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (pred >= lo) & (pred < hi if i < n_bins - 1 else pred <= hi)
        if m.sum() > 0:
            out.append({"bin": [float(lo), float(hi)], "n": int(m.sum()),
                        "pred_mean": float(pred[m].mean()), "actual_mean": float(G[m].mean())})
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-t", type=int, default=30)
    parser.add_argument("--gamma", type=float, default=0.9)
    parser.add_argument("--horizon-features", action="store_true",
                        help="append [log1p(remaining), mean_past_reward, last_reward]")
    args = parser.parse_args()
    t_start = time.time()

    data = build_lln(min_rating=4, n_test=1, n_valid=1)
    cfg = IALSConfig(dim=64, alpha=40.0, lam=1.0, n_iter=20)
    U, V = train_ials(data.positives_train, data.n_users, data.n_items, cfg, seed=args.seed)

    print("building logged trajectories...", flush=True)
    per_user = {}
    for sp in data.splits:
        X_u, G_u = build_logged_trajectories([sp], V, cfg, eta=1.0, gamma=args.gamma,
                                             max_t=args.max_t, horizon_features=args.horizon_features)
        if len(G_u) > 0:
            per_user[sp.user] = (X_u, G_u)
    users = np.array(sorted(per_user.keys()))
    tr_u, va_u, te_u = split_users(users, seed=args.seed)
    Xtr = np.vstack([per_user[u][0] for u in tr_u])
    Gtr = np.concatenate([per_user[u][1] for u in tr_u])
    Xva = np.vstack([per_user[u][0] for u in va_u])
    Gva = np.concatenate([per_user[u][1] for u in va_u])
    Xte = np.vstack([per_user[u][0] for u in te_u])
    Gte = np.concatenate([per_user[u][1] for u in te_u])
    print(f"  train {len(Gtr)}, valid {len(Gva)}, test {len(Gte)} decision points", flush=True)

    results = {"gamma": args.gamma, "max_t": args.max_t, "horizon_features": args.horizon_features,
               "n_users": {"train": len(tr_u), "valid": len(va_u), "test": len(te_u)}}
    best = None
    for lam in (0.1, 1.0, 10.0, 100.0):
        vm = VLoggingRidge(lam=lam, horizon_features=args.horizon_features)
        vm.fit(Xtr, Gtr)
        pv = vm.predict_batch(Xva)
        m = metrics(Gva, pv)
        print(f"  lam={lam:6.1f}  valid RMSE={m['rmse']:.4f} MAE={m['mae']:.4f} R2={m['r2']:.4f}", flush=True)
        if best is None or m["rmse"] < best[1]["rmse"]:
            best = (lam, m, vm)
    lam, m_va, vm = best
    results["selected_lambda"] = lam
    results["valid"] = m_va
    results["test"] = metrics(Gte, vm.predict_batch(Xte))
    results["reliability_test"] = reliability(Gte, vm.predict_batch(Xte))
    print(f"selected lam={lam}; test: {json.dumps(results['test'])}", flush=True)

    out_dir = Path(os.environ.get("MFMDP_OUTPUT_ROOT", "./outputs")) / "phase4"
    ckpt_dir = Path(os.environ.get("MFMDP_CHECKPOINT_ROOT", "./checkpoints"))
    out_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    tag = "horizon" if args.horizon_features else "base"
    with open(out_dir / f"value_model_{tag}_seed{args.seed}.json", "w") as f:
        json.dump(results, f, indent=2)
    np.savez(
        ckpt_dir / f"v_logging_ridge_{tag}_seed{args.seed}.npz",
        weights=vm.weights, mean=vm.encoder.mean, std=vm.encoder.std,
        lam=lam, gamma=args.gamma, horizon_features=args.horizon_features,
    )
    print(f"wrote {out_dir / f'value_model_{tag}_seed{args.seed}.json'} and checkpoint ({time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
