"""H2-E1 factorial experiment: state (mf | seq | hybrid) x transition (optimistic |
stochastic) on Env-C, plus the three mandatory negative controls.

Protocol (docs/H2_HYBRID_SEQUENTIAL_STATE_DESIGN.md): behavior-logged trajectories only
(planners never see theta); iALS factors, MF response/hazard logistics, and the GRU world
model trained per seed; closed-loop J in the TRUE env from held-out initial states;
true-theta greedy as reference oracle; paired bootstrap CIs over seeds.

Usage:
    python experiments/h2/run_h2_factorial.py [--seeds 5] [--n-train-users 400]
        [--n-eval-users 200] [--epochs 150] [--controls] [--quick]
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
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from mf_mdp.envs.continuous_env import EnvC, EnvCConfig, gen_theta0s
from mf_mdp.mf import IALSConfig, train_ials, user_state
from mf_mdp.models.features import FeatureEncoder, features_for
from mf_mdp.models.response import (LogisticResponseModel, brier_score,
                                    expected_calibration_error, log_loss, roc_auc)
from mf_mdp.planning.hybrid import HybridPlanner, HybridPlannerConfig, HybridState
from mf_mdp.sequential.world import CONTROL_MODES, SeqWorldConfig, SeqWorldModel

ARMS = [(arm, tr) for arm in ("mf", "seq", "hybrid") for tr in ("optimistic", "stochastic")]


def behavior_policy(env, theta, f, consumed, rng):
    avail = [i for i in range(env.n_items) if i not in consumed] or list(range(env.n_items))
    if rng.random() < 0.3:
        return int(rng.choice(avail))
    return max(avail, key=lambda i: env.q[i] * float(theta @ env.V[i]))


def log_trajectories(env, thetas, rng):
    trajs = []
    for theta in thetas:
        th, f, consumed = theta.copy(), 0, set()
        traj = []
        for _ in range(env.cfg.H):
            i = behavior_policy(env, th, f, consumed, rng)
            r, term, th, nf, engaged = env.step(th, f, i, rng)
            traj.append((int(i), int(engaged), int(term)))
            consumed.add(i)
            if term:
                break
            f = nf
        trajs.append(traj)
    return trajs


def estimate_q_hat(trajs, n_items, shrink=5.0):
    """Engagement-shrunk empirical quality per item (shared across arms, no env leakage)."""
    exp = np.zeros(n_items)
    eng = np.zeros(n_items)
    for traj in trajs:
        for (i, e, _t) in traj:
            exp[i] += 1
            eng[i] += e
    global_mean = eng.sum() / max(exp.sum(), 1.0)
    return (eng + shrink * global_mean) / (exp + shrink)


def train_mf_models(trajs, V, ials_cfg, log_pop):
    X, y, y_term = [], [], []
    for traj in trajs:
        hist = []
        st = user_state(V, np.asarray(hist, dtype=np.int64), ials_cfg)
        for (i, e, term) in traj:
            X.append(features_for(st, V, i, log_pop))
            y.append(float(e))
            y_term.append(float(term))
            hist.append(i)
            st = user_state(V, np.asarray(hist, dtype=np.int64), ials_cfg)
    X, y, y_term = np.vstack(X), np.asarray(y), np.asarray(y_term)
    enc = FeatureEncoder()
    resp = LogisticResponseModel(l2=1e-3, lr=0.5, n_steps=800)
    resp.fit(enc.fit_transform(X), y)
    hresp = LogisticResponseModel(l2=1e-3, lr=0.5, n_steps=800)
    hresp.fit(enc.transform(X), y_term)
    return enc, resp, hresp


def make_planners(V, ials_cfg, log_pop, q_hat, kappa, world, world_by_control, enc, resp, hresp):
    planners = {}
    for arm, tr in ARMS:
        cfg = HybridPlannerConfig(state_arm=arm, transition=tr)
        w = None if arm == "mf" else world
        planners[f"{arm}+{tr}"] = HybridPlanner(
            V, ials_cfg, w, resp, hresp, enc, log_pop, q_hat, kappa, cfg)
    for c, w in world_by_control.items():
        for arm in ("seq", "hybrid"):
            cfg = HybridPlannerConfig(state_arm=arm, transition="stochastic")
            planners[f"{arm}+stochastic+{c}"] = HybridPlanner(
                V, ials_cfg, w, resp, hresp, enc, log_pop, q_hat, kappa, cfg)
    return planners


def build_state(pl, arm, hist_items, events):
    P = u = h = None
    consumed = tuple(sorted(hist_items))
    if arm in ("mf", "hybrid"):
        st = user_state(pl.V, np.asarray(hist_items, dtype=np.int64), pl.ials_cfg)
        P, u = st.P, st.u
    if arm in ("seq", "hybrid"):
        h = pl.world.state_of(events)
    return HybridState(P=P, u=u, h=h, consumed=consumed)


def evaluate_closed_loop(planner, arm, env, eval_thetas, rng):
    rets = []
    for theta in eval_thetas:
        th, f, hist, events = theta.copy(), 0, [], []
        total = 0.0
        for t in range(env.cfg.H):
            st = build_state(planner, arm, hist, events)
            i = planner.plan(st, f, rng)
            r, term, th, nf, engaged = env.step(th, f, i, rng)
            total += (env.cfg.gamma ** t) * r
            hist.append(i)
            events.append((i, int(engaged)))
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


def mf_response_holdout(holdout, V, ials_cfg, enc, resp, log_pop, k=10):
    probs, ys, next_hits, n_pos = [], [], [], 0
    for traj in holdout:
        hist = []
        st = user_state(V, np.asarray(hist, dtype=np.int64), ials_cfg)
        for j, (i, e, _t) in enumerate(traj):
            x = enc.transform(features_for(st, V, i, log_pop)[None, :])
            probs.append(float(resp.predict_proba(x)[0]))
            ys.append(float(e))
            if j + 1 < len(traj):
                nxt = traj[j + 1][0]
                u = st.u
                scores = V @ u
                top = np.argsort(-scores)[:k]
                next_hits.append(int(nxt in top))
                n_pos += 1
            hist.append(i)
            st = user_state(V, np.asarray(hist, dtype=np.int64), ials_cfg)
    probs, ys = np.asarray(probs), np.asarray(ys)
    return {"log_loss": log_loss(ys, probs), "brier": brier_score(ys, probs),
            "roc_auc": roc_auc(ys, probs), "ece": expected_calibration_error(ys, probs),
            "next_item_recall10": float(np.mean(next_hits)) if next_hits else None}


def seq_response_holdout(world, holdout, k=10):
    probs, ys, next_hits = [], [], []
    for traj in holdout:
        events = [(i, e) for (i, e, _t) in traj]
        p, scores, y = world.predict_trajectory(events)
        probs.extend(p.tolist())
        ys.extend(y.tolist())
        for j in range(len(events) - 1):
            top = np.argsort(-scores[j])[:k]
            next_hits.append(int(events[j + 1][0] in top))
    probs, ys = np.asarray(probs), np.asarray(ys)
    return {"log_loss": log_loss(ys, probs), "brier": brier_score(ys, probs),
            "roc_auc": roc_auc(ys, probs), "ece": expected_calibration_error(ys, probs),
            "next_item_recall10": float(np.mean(next_hits)) if next_hits else None}


def run_seed(seed, n_train, n_eval, epochs, run_controls):
    env = EnvC(EnvCConfig(seed=seed))
    rng = np.random.default_rng(seed)
    ials_cfg = IALSConfig(dim=16, alpha=10.0, lam=0.1, n_iter=15)
    train_thetas = gen_theta0s(n_train, env, seed=1000 + seed)
    trajs = log_trajectories(env, train_thetas, rng)
    positives = np.array([(u, t[0]) for u, tr in enumerate(trajs) for t in tr], dtype=np.int64)
    _, V = train_ials(positives, n_train, env.n_items, ials_cfg, seed=seed)
    log_pop = np.log1p(np.bincount(positives[:, 1], minlength=env.n_items).astype(float))
    q_hat = estimate_q_hat(trajs, env.n_items)
    enc, resp, hresp = train_mf_models(trajs, V, ials_cfg, log_pop)

    wcfg = SeqWorldConfig(n_epochs=epochs, seed=seed)
    world = SeqWorldModel(env.n_items, wcfg)
    world.fit(trajs)
    world_by_control = {}
    if run_controls:
        for c in CONTROL_MODES[1:]:
            w = SeqWorldModel(env.n_items, SeqWorldConfig(n_epochs=epochs, seed=seed))
            w.fit(trajs, control=c)
            world_by_control[c] = w

    planners = make_planners(V, ials_cfg, log_pop, q_hat, env.cfg.kappa_sat,
                             world, world_by_control, enc, resp, hresp)

    holdout = log_trajectories(env, gen_theta0s(n_eval, env, seed=6000 + seed), rng)
    predictive = {"mf": mf_response_holdout(holdout, V, ials_cfg, enc, resp, log_pop),
                  "seq": seq_response_holdout(world, holdout)}
    for c, w in world_by_control.items():
        predictive[f"seq+{c}"] = seq_response_holdout(w, holdout)

    eval_thetas = gen_theta0s(n_eval, env, seed=5000 + seed)
    out = {"predictive": predictive}
    for name, pl in planners.items():
        arm = pl.cfg.state_arm
        j, jsd = evaluate_closed_loop(pl, arm, env, eval_thetas, rng)
        out[name] = {"J_mean": j, "J_std": jsd}
    out["true_greedy"] = {"J_mean": true_greedy_reference(env, eval_thetas, seed)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--n-train-users", type=int, default=400)
    ap.add_argument("--n-eval-users", type=int, default=200)
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--controls", action="store_true", default=True)
    ap.add_argument("--no-controls", dest="controls", action="store_false")
    ap.add_argument("--quick", action="store_true", help="tiny smoke run")
    args = ap.parse_args()
    if args.quick:
        args.seeds, args.n_train_users, args.n_eval_users, args.epochs = 1, 100, 60, 30

    t0 = time.time()
    per_seed = [run_seed(s, args.n_train_users, args.n_eval_users, args.epochs, args.controls)
                for s in range(args.seeds)]

    names = list(per_seed[0].keys())
    agg = {}
    for name in names:
        if name == "predictive":
            continue
        vals = [r[name]["J_mean"] for r in per_seed]
        agg[name] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)),
                     "se": float(np.std(vals) / np.sqrt(len(vals)))}
    rng = np.random.default_rng(0)

    def ci(diff):
        boots = [np.mean(rng.choice(diff, len(diff))) for _ in range(2000)]
        return [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]

    contrasts = {}
    for tr in ("optimistic", "stochastic"):
        d = [r[f"hybrid+{tr}"]["J_mean"] - r[f"mf+{tr}"]["J_mean"] for r in per_seed]
        contrasts[f"hybrid-mf@{tr}"] = {"mean": float(np.mean(d)), "ci95": ci(d)}
    for arm in ("mf", "seq", "hybrid"):
        d = [r[f"{arm}+stochastic"]["J_mean"] - r[f"{arm}+optimistic"]["J_mean"] for r in per_seed]
        contrasts[f"stoch-opt@{arm}"] = {"mean": float(np.mean(d)), "ci95": ci(d)}
    if args.controls:
        for c in CONTROL_MODES[1:]:
            for arm in ("seq", "hybrid"):
                key = f"{arm}+stochastic+{c}"
                d = [r[key]["J_mean"] - r[f"{arm}+stochastic"]["J_mean"] for r in per_seed]
                contrasts[f"control-{c}@{arm}(minus clean)"] = {"mean": float(np.mean(d)), "ci95": ci(d)}

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         cwd=Path(__file__).resolve().parent.parent).decode().strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                         cwd=Path(__file__).resolve().parent.parent).decode().strip()
    except Exception:
        commit, branch = None, None
    out = {"config": vars(args), "git": {"commit": commit, "branch": branch},
           "aggregate": agg, "contrasts": contrasts,
           "predictive_seed0": per_seed[0]["predictive"],
           "per_seed": per_seed, "runtime_s": time.time() - t0}
    print(json.dumps({"aggregate": agg, "contrasts": contrasts,
                      "predictive_seed0": out["predictive_seed0"]}, indent=2))
    out_dir = Path(os.environ.get("MFMDP_OUTPUT_ROOT", "./outputs")) / "h2"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "h2_env_c_factorial.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
