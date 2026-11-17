"""H4 decision-level experiment: cross-feature response models inside the Env-C planner.

Protocol: H2-E1 pipeline with the state arm FIXED to `mf` and transition FIXED to
`stochastic` (the only mode where p_hat drives planning — under `optimistic`,
_p == 1 and the response model is unused; verified in planning/hybrid.py:_p).
Only the response model moves across arms; the hazard model stays the shared LR
on base10 features in every arm. True J(pi) is measured in closed loop in the
TRUE environment; per-arm predictive holdout metrics are recorded for the
transmission check (docs/H4_CROSS_RESPONSE_MODEL_DESIGN.md §4).

Usage:
    python experiments/h4/run_env_c_cross.py [--seeds 5] [--n-train-users 400]
        [--n-eval-users 200] [--quick]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mf_mdp.envs.continuous_env import EnvC, EnvCConfig, gen_theta0s
from mf_mdp.interactions.cross import RandomCrosses
from mf_mdp.interactions.fm import FactorizationMachine
from mf_mdp.interactions.mlp import MLPClassifier
from mf_mdp.interactions.planner import CrossPlanner, CrossPlannerConfig
from mf_mdp.mf import IALSConfig, train_ials, user_state
from mf_mdp.models.features import FeatureEncoder
from mf_mdp.models.response import (LogisticResponseModel, brier_score,
                                    expected_calibration_error, log_loss, roc_auc)
from mf_mdp.planning.hybrid import HybridState

from common import assemble_envc, envc_feature_rows, estimate_q_hat, log_trajectories

# arm -> (feature mode, model family); hazard is always shared LR on base10
ARMS = {
    "lr": ("base", "lr"),
    "lr_x": ("cross", "lr"),
    "lr_randx": ("randcross", "lr"),
    "fm": ("baseh", "fm"),
    "mlp": ("baseh", "mlp"),
    "mlp_uv": ("uv", "mlp"),
}
FM_K_GRID = (4, 8, 16)
MLP_HIDDEN_GRID = (16, 32)


def fit_lr(X, y):
    enc = FeatureEncoder()
    m = LogisticResponseModel(l2=1e-3, lr=0.5, n_steps=800)
    m.fit(enc.fit_transform(X), y)
    return enc, m


def fit_fm(X, y, k):
    enc = FeatureEncoder()
    m = FactorizationMachine(k=k, l2=1e-4, lr=0.1, n_epochs=100, seed=0)
    m.fit(enc.fit_transform(X), y)
    return enc, m


def fit_mlp(X, y, hidden):
    enc = FeatureEncoder()
    m = MLPClassifier(hidden=hidden, l2=1e-4, lr=0.1, n_epochs=100, seed=0)
    m.fit(enc.fit_transform(X), y)
    return enc, m


def fit_arm(arm, X, y, hp):
    kind = ARMS[arm][1]
    if kind == "lr":
        return fit_lr(X, y)
    if kind == "fm":
        return fit_fm(X, y, hp["fm_k"])
    return fit_mlp(X, y, hp["mlp_hidden"])


def select_hp(tr_rows, va_rows):
    """Small validation grid for FM k / MLP hidden (valid users = last 20%)."""
    hp = {"fm_k": 8, "mlp_hidden": 32}
    best = {}
    for k in FM_K_GRID:
        enc, m = fit_fm(assemble_envc(tr_rows, "baseh"), tr_rows["y"], k)
        ll = log_loss(va_rows["y"], m.predict_proba(enc.transform(assemble_envc(va_rows, "baseh"))))
        if "fm" not in best or ll < best["fm"][1]:
            best["fm"] = (k, ll)
    for h in MLP_HIDDEN_GRID:
        enc, m = fit_mlp(assemble_envc(tr_rows, "baseh"), tr_rows["y"], h)
        ll = log_loss(va_rows["y"], m.predict_proba(enc.transform(assemble_envc(va_rows, "baseh"))))
        if "mlp" not in best or ll < best["mlp"][1]:
            best["mlp"] = (h, ll)
    hp["fm_k"], hp["fm_valid_ll"] = best["fm"]
    hp["mlp_hidden"], hp["mlp_valid_ll"] = best["mlp"]
    return hp


def build_planner(env, V, ials_cfg, log_pop, q_hat, arm, rows, hp, rand_crosses):
    mode, kind = ARMS[arm]
    X = assemble_envc(rows, mode, rand_crosses)
    enc_resp, resp = fit_arm(arm, X, rows["y"], hp)
    enc_haz, haz = fit_lr(rows["base"], rows["y_term"])
    cfg = CrossPlannerConfig(state_arm="mf", transition="stochastic", feature_mode=mode)
    pl = CrossPlanner(V, ials_cfg, None, resp, haz, enc_resp, log_pop, q_hat,
                      env.cfg.kappa_sat, cfg=cfg, enc_resp=enc_resp, enc_haz=enc_haz,
                      rand_crosses=rand_crosses)
    return pl, enc_resp, resp


def evaluate_closed_loop(planner, env, eval_thetas, rng):
    rets = []
    for theta in eval_thetas:
        th, f, hist, events = theta.copy(), 0, [], []
        total = 0.0
        for t in range(env.cfg.H):
            st = user_state(planner.V, np.asarray(hist, dtype=np.int64), planner.ials_cfg)
            hstate = HybridState(P=st.P, u=st.u, h=None, consumed=tuple(sorted(hist)))
            i = planner.plan(hstate, f, None, events)
            r, term, th, nf, engaged = env.step(th, f, i, rng)
            total += (env.cfg.gamma ** t) * r
            hist.append(i)
            events.append((i, float(engaged)))
            if term:
                break
            f = nf
        rets.append(total)
    return float(np.mean(rets)), float(np.std(rets))


def true_greedy_reference(env, eval_thetas, seed):
    rets = []
    for theta in eval_thetas:
        rng2 = np.random.default_rng(seed + 77)
        th, f, consumed = theta.copy(), 0, set()
        total = 0.0
        for t in range(env.cfg.H):
            avail = [i for i in range(env.n_items) if i not in consumed] or list(range(env.n_items))
            i = max(avail, key=lambda j: env.q[j] * float(th @ env.V[j]))
            r, term, th, nf, engaged = env.step(th, f, i, rng2)
            total += (env.cfg.gamma ** t) * r
            consumed.add(i)
            if term:
                break
            f = nf
        rets.append(total)
    return float(np.mean(rets))


def holdout_metrics(planner, enc_resp, resp, rows, mode, rand_crosses):
    X = assemble_envc(rows, mode, rand_crosses)
    p = resp.predict_proba(enc_resp.transform(X))
    return {"log_loss": log_loss(rows["y"], p), "brier": brier_score(rows["y"], p),
            "roc_auc": roc_auc(rows["y"], p), "ece": expected_calibration_error(rows["y"], p)}


def run_seed(seed, n_train, n_eval):
    env = EnvC(EnvCConfig(seed=seed))
    rng = np.random.default_rng(seed)
    ials_cfg = IALSConfig(dim=16, alpha=10.0, lam=0.1, n_iter=15)
    train_thetas = gen_theta0s(n_train, env, seed=1000 + seed)
    trajs = log_trajectories(env, train_thetas, rng)
    positives = np.array([(u, t[0]) for u, tr in enumerate(trajs) for t in tr], dtype=np.int64)
    _, V = train_ials(positives, n_train, env.n_items, ials_cfg, seed=seed)
    log_pop = np.log1p(np.bincount(positives[:, 1], minlength=env.n_items).astype(float))
    q_hat = estimate_q_hat(trajs, env.n_items)

    n_tr = max(1, int(0.8 * len(trajs)))
    tr_rows = envc_feature_rows(trajs[:n_tr], V, ials_cfg, log_pop)
    va_rows = envc_feature_rows(trajs[n_tr:], V, ials_cfg, log_pop)
    hp = select_hp(tr_rows, va_rows)
    rows = envc_feature_rows(trajs, V, ials_cfg, log_pop)

    rand_crosses = RandomCrosses(k=4, seed=seed).fit(assemble_envc(rows, "baseh"))
    planners, fitted = {}, {}
    for arm in ARMS:
        pl, enc_resp, resp = build_planner(env, V, ials_cfg, log_pop, q_hat, arm, rows, hp, rand_crosses)
        planners[arm] = pl
        fitted[arm] = (enc_resp, resp)

    # references: optimistic + lr (Phase-5/H2 baseline arm), true greedy oracle
    enc_o, resp_o = fit_lr(rows["base"], rows["y"])
    enc_haz, haz = fit_lr(rows["base"], rows["y_term"])
    cfg_opt = CrossPlannerConfig(state_arm="mf", transition="optimistic", feature_mode="base")
    planners["optimistic_lr"] = CrossPlanner(V, ials_cfg, None, resp_o, haz, enc_o, log_pop,
                                             q_hat, env.cfg.kappa_sat, cfg=cfg_opt,
                                             enc_resp=enc_o, enc_haz=enc_haz)

    holdout = log_trajectories(env, gen_theta0s(n_eval, env, seed=6000 + seed), rng)
    ho_rows = envc_feature_rows(holdout, V, ials_cfg, log_pop)
    predictive = {arm: holdout_metrics(planners[arm], *fitted[arm], ho_rows, ARMS[arm][0], rand_crosses)
                  for arm in ARMS}

    eval_thetas = gen_theta0s(n_eval, env, seed=5000 + seed)
    out = {"predictive": predictive, "hp": hp}
    for name, pl in planners.items():
        j, jsd = evaluate_closed_loop(pl, env, eval_thetas, rng)
        out[name] = {"J_mean": j, "J_std": jsd}
    out["true_greedy"] = {"J_mean": true_greedy_reference(env, eval_thetas, seed)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-train-users", type=int, default=400)
    ap.add_argument("--n-eval-users", type=int, default=200)
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.n_train_users, args.n_eval_users = 1, 100, 60

    t0 = time.time()
    per_seed = [run_seed(s, args.n_train_users, args.n_eval_users) for s in range(args.seeds)]

    names = [n for n in per_seed[0] if n != "predictive" and n != "hp"]
    agg = {}
    for name in names:
        vals = [r[name]["J_mean"] for r in per_seed]
        agg[name] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                     "se": float(np.std(vals) / np.sqrt(len(vals)))}
    rng = np.random.default_rng(0)

    def ci(diff):
        boots = [np.mean(rng.choice(diff, len(diff))) for _ in range(2000)]
        return [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]

    contrasts = {}
    for arm in ARMS:
        d = [r[arm]["J_mean"] - r["lr"]["J_mean"] for r in per_seed]
        contrasts[f"{arm}-lr"] = {"mean": float(np.mean(d)), "ci95": ci(d)}
    d = [r["lr"]["J_mean"] - r["optimistic_lr"]["J_mean"] for r in per_seed]
    contrasts["lr-optimistic_lr"] = {"mean": float(np.mean(d)), "ci95": ci(d)}

    pred_agg = {}
    for arm in ARMS:
        for k in ("log_loss", "roc_auc", "ece"):
            vals = [r["predictive"][arm][k] for r in per_seed]
            pred_agg.setdefault(arm, {})[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         cwd=Path(__file__).resolve().parent.parent.parent).decode().strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                         cwd=Path(__file__).resolve().parent.parent.parent).decode().strip()
    except Exception:
        commit, branch = None, None
    out = {"config": vars(args), "git": {"commit": commit, "branch": branch},
           "arms": {a: {"mode": m, "model": k} for a, (m, k) in ARMS.items()},
           "aggregate": agg, "contrasts": contrasts, "predictive_holdout": pred_agg,
           "per_seed": per_seed, "runtime_s": time.time() - t0}
    print(json.dumps({"aggregate": agg, "contrasts": contrasts, "predictive_holdout": pred_agg}, indent=2))
    out_dir = Path(os.environ.get("MFMDP_OUTPUT_ROOT", "./outputs")) / "h4"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "env_c_cross.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
