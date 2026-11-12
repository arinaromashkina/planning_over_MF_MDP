"""E1+E2: response-model quality and reward-variant ablation (R0/R1/R2) on ML-1M LLN.

Protocol: user-disjoint 70/15/15 split (response-train / HP-valid / final-test).
Response model: logistic regression on decision-time features, y = 1[rating == 5].
R1 = p5*U5 + (1-p5)*U4 (+ optional alignment eta); R2 = direct ridge on utility.
Plan-1 ranking uses the learned reward via reward_fn. All HP choices on the
validation split only.

Usage: python experiments/phase4/run_reward_ablation.py [--seed 0]
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

from common import build_decision_dataset, split_users
from mf_mdp.data.movielens import build_lln
from mf_mdp.evaluation.runner import evaluate_method
from mf_mdp.mf import IALSConfig, train_ials
from mf_mdp.models.features import FeatureEncoder
from mf_mdp.models.response import (
    LogisticResponseModel,
    brier_score,
    expected_calibration_error,
    log_loss,
    roc_auc,
)
from mf_mdp.models.reward import R1ExpectedReward, R2DirectExpectedReward, UtilityMapping, fit_ridge
from mf_mdp.planning import PlannerConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-users-eval", type=int, default=500)
    args = parser.parse_args()
    t_start = time.time()

    data = build_lln(min_rating=4, n_test=1, n_valid=1)
    cfg = IALSConfig(dim=64, alpha=40.0, lam=1.0, n_iter=20)
    U, V = train_ials(data.positives_train, data.n_users, data.n_items, cfg, seed=args.seed)
    log_pop = np.log1p(np.bincount(data.positives_train[:, 1], minlength=data.n_items).astype(float))

    print("building decision dataset...", flush=True)
    X, y, uid, tpos = build_decision_dataset(data, data.splits, V, cfg, max_t=40)
    print(f"  {len(X)} decision examples, {y.mean():.3f} rating-5 rate", flush=True)

    tr_u, va_u, te_u = split_users(uid, seed=args.seed)
    tr_m, va_m, te_m = np.isin(uid, tr_u), np.isin(uid, va_u), np.isin(uid, te_u)

    # ---- E2: response model quality on held-out users
    enc = FeatureEncoder()
    Xtr = enc.fit_transform(X[tr_m])
    resp = LogisticResponseModel(l2=1e-3, lr=0.5, n_steps=600)
    resp.fit(Xtr, y[tr_m])
    metrics = {}
    for name, m in (("valid", va_m), ("test", te_m)):
        p = resp.predict_proba(enc.transform(X[m]))
        metrics[name] = {
            "n": int(m.sum()), "base_rate": float(y[m].mean()),
            "log_loss": log_loss(y[m], p), "brier": brier_score(y[m], p),
            "roc_auc": roc_auc(y[m], p), "ece": expected_calibration_error(y[m], p),
        }
    print(json.dumps({"response_model": metrics}, indent=2), flush=True)

    # ---- E1: reward-variant ablation with Plan-1 on validation users
    va_splits = [sp for sp in data.splits if sp.user in set(va_u.tolist())][: args.n_users_eval]
    te_splits = [sp for sp in data.splits if sp.user in set(te_u.tolist())][: args.n_users_eval]
    print(f"eval users: {len(va_splits)} valid-split, {len(te_splits)} test-split", flush=True)

    def eval_reward(reward_fn, splits, tag):
        pcfg = PlannerConfig(method="plan1", reward_fn=reward_fn, plan1_pool=100)
        res = evaluate_method(V, splits, 10, pcfg, np.random.default_rng(args.seed), cfg, holdout="valid")
        print(f"  {tag:24s} R@10={res['recall_at_k']:.4f} nDCG={res['ndcg_at_k']:.4f}", flush=True)
        return res["recall_at_k"]

    results = {"response_model": metrics, "ablation": {}}
    # R0 baseline (paper heuristic) — no model
    res_r0 = evaluate_method(V, va_splits, 10, PlannerConfig(method="plan1", plan1_pool=100),
                             np.random.default_rng(args.seed), cfg, holdout="valid")
    print(f"  R0 paper (valid)         R@10={res_r0['recall_at_k']:.4f}")
    results["ablation"]["R0_paper_valid"] = res_r0["recall_at_k"]

    # R1 grid
    r1_results = {}
    for u4 in (0.0, 0.25, 0.5):
        for eta in (0.0, 1.0):
            r1 = R1ExpectedReward(lambda f: resp.predict_proba(f), UtilityMapping(1.0, u4), enc, eta=eta)
            r1_results[(u4, eta)] = eval_reward(
                lambda st, a: r1.predict_reward(st, V, a, log_pop), va_splits, f"R1 u4={u4} eta={eta}")
    results["ablation"]["R1_grid_valid"] = {f"u4={k[0]},eta={k[1]}": v for k, v in r1_results.items()}

    # R2 direct expected utility (u4 fixed 0.25)
    util = UtilityMapping(1.0, 0.25)
    y_util = np.where(y >= 5, 1.0, 0.25)
    w_r2 = fit_ridge(Xtr, y_util[tr_m], lam=1.0)
    r2 = R2DirectExpectedReward(w_r2, enc, eta=0.0)
    results["ablation"]["R2_valid"] = eval_reward(
        lambda st, a: r2.predict_reward(st, V, a, log_pop), va_splits, "R2 direct (u4=0.25)")

    # choose best variant on validation, run on test split
    all_valid = {"R0_paper": results["ablation"]["R0_paper_valid"], **{f"R1_u4={k[0]}_eta={k[1]}": v for k, v in r1_results.items()}, "R2": results["ablation"]["R2_valid"]}
    best = max(all_valid, key=all_valid.get)
    print(f"best on validation: {best} ({all_valid[best]:.4f})", flush=True)
    results["selected"] = {"variant": best, "valid_recall": all_valid[best]}
    # final test evaluation of the selected variant
    if best.startswith("R1"):
        u4 = float(best.split("u4=")[1].split("_")[0])
        eta = float(best.split("eta=")[1])
        r1 = R1ExpectedReward(lambda f: resp.predict_proba(f), UtilityMapping(1.0, u4), enc, eta=eta)
        fn = lambda st, a: r1.predict_reward(st, V, a, log_pop)
    elif best == "R2":
        fn = lambda st, a: r2.predict_reward(st, V, a, log_pop)
    else:
        fn = None
    if fn is None:
        res_te = evaluate_method(V, te_splits, 10, PlannerConfig(method="plan1", plan1_pool=100),
                                 np.random.default_rng(args.seed), cfg, holdout="test")
    else:
        res_te = evaluate_method(V, te_splits, 10, PlannerConfig(method="plan1", reward_fn=fn, plan1_pool=100),
                                 np.random.default_rng(args.seed), cfg, holdout="test")
    results["selected"]["test_recall"] = res_te["recall_at_k"]
    results["selected"]["test_ndcg"] = res_te["ndcg_at_k"]
    # R0 on test split for reference
    res_te_r0 = evaluate_method(V, te_splits, 10, PlannerConfig(method="plan1", plan1_pool=100),
                                np.random.default_rng(args.seed), cfg, holdout="test")
    results["R0_paper_test"] = {"recall": res_te_r0["recall_at_k"], "ndcg": res_te_r0["ndcg_at_k"]}
    results["runtime_s"] = time.time() - t_start

    out_dir = Path(os.environ.get("MFMDP_OUTPUT_ROOT", "./outputs")) / "phase4"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / f"reward_ablation_seed{args.seed}.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_dir / f'reward_ablation_seed{args.seed}.json'} ({time.time()-t_start:.0f}s)")


if __name__ == "__main__":
    main()
