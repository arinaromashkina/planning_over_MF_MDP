"""Experiment A1/A2/A3 — Env-S controlled environment checks and learned-model ablations.

Modes:
  --checks     A1: planning correctness (greedy vs DP, MCTS convergence)
  --learn      A2: learn response model + transition params; held-out predictive metrics
  --ablation   A3: optimistic vs stochastic planners at matched budgets, TRUE-env J

Usage: python experiments/phase5/run_env_a.py [--checks|--learn|--ablation] [--seeds 5]
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

from mf_mdp.envs import DPPolicy, EnvS, EnvSConfig, GreedyPolicy, MCTSPolicy
from mf_mdp.envs.learned import LogisticResponse, ModelDP, estimate_rho, gen_trajectories
from mf_mdp.models.response import brier_score, expected_calibration_error, log_loss, roc_auc


def evaluate(policy, env, s0, f0, n_rollouts, seed):
    rng = np.random.default_rng(seed)
    return np.array([env.rollout(policy, s0, f0, rng) for _ in range(n_rollouts)])


def paired_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 0):
    rng = np.random.default_rng(seed)
    diff = a - b
    boots = [np.mean(rng.choice(diff, len(diff))) for _ in range(n_boot)]
    return {"mean": float(diff.mean()),
            "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}


def run_checks(args):
    env = EnvS(EnvSConfig(seed=0))
    dp, greedy = DPPolicy(env), GreedyPolicy(env)
    out = {"env": "EnvS", "H": env.cfg.H, "gamma": env.cfg.gamma, "n_rollouts": args.n_rollouts}

    disagree = [(s, f, int(dp.pi[0][s][f]), greedy(s, f, 0))
                for s in range(env.K) for f in range(3)
                if dp.pi[0][s][f] != greedy(s, f, 0)]
    out["greedy_dp_disagreements"] = {"count": len(disagree), "states": disagree}

    rng = np.random.default_rng(0)
    J_dp = evaluate(dp, env, 0, 0, args.n_rollouts, seed=100)
    J_greedy = evaluate(greedy, env, 0, 0, args.n_rollouts, seed=100)
    out["J_dp"] = {"mean": float(J_dp.mean()), "std": float(J_dp.std())}
    out["J_greedy"] = {"mean": float(J_greedy.mean()), "std": float(J_greedy.std())}
    out["J_gap"] = paired_ci(J_dp, J_greedy, seed=0)

    budgets = [200, 1000, 5000, 20000]
    agreement = {}
    for B in budgets:
        mcts = MCTSPolicy(env, n_sims=B, c_uct=1.4, rollout_greedy=True)
        rng = np.random.default_rng(3)
        agree = sum(
            mcts.action(s, f, rng) == dp.pi[0][s][f]
            for s in range(env.K) for f in range(3)
        )
        agreement[B] = f"{agree}/12"
    out["mcts_dp_agreement_by_budget"] = agreement
    return out


def run_learn(args):
    env = EnvS(EnvSConfig(seed=0))
    trajs = gen_trajectories(env, "eps_greedy", 1500, seed=1) + gen_trajectories(env, "uniform", 1500, seed=2)
    test_trajs = gen_trajectories(env, "eps_greedy", 500, seed=101) + gen_trajectories(env, "uniform", 500, seed=102)

    resp = LogisticResponse()
    resp.fit(env, trajs)
    rho_hat = estimate_rho(env, trajs)

    X, y, p = [], [], []
    for traj in test_trajs:
        for s, f, i, e, _ in traj:
            X.append(resp._features(env, s, f, i))
            y.append(float(e))
            p.append(resp.predict(env, s, f, i))
    y, p = np.asarray(y), np.asarray(p)
    out = {
        "n_train_traj": len(trajs), "n_test_events": len(y),
        "base_rate": float(y.mean()),
        "log_loss": log_loss(y, p), "brier": brier_score(y, p),
        "roc_auc": roc_auc(y, p), "ece": expected_calibration_error(y, p),
        "rho_hat": float(rho_hat), "rho_true": env.cfg.rho,
    }
    return out


def run_ablation(args):
    env = EnvS(EnvSConfig(seed=0))
    oracle = DPPolicy(env)
    rows = []
    for seed in range(args.seeds):
        trajs = gen_trajectories(env, "eps_greedy", 1500, seed=10 + seed) + gen_trajectories(env, "uniform", 1500, seed=20 + seed)
        resp = LogisticResponse()
        resp.fit(env, trajs)
        rho_hat = estimate_rho(env, trajs)
        policies = {
            "A0_optimistic_depthH": ModelDP(env, resp, rho_hat, optimistic=True),
            "A1_stochastic_depthH": ModelDP(env, resp, rho_hat),
            "A2_stochastic_depth3": ModelDP(env, resp, rho_hat, horizon=3),
            "A3_stochastic_depth2": ModelDP(env, resp, rho_hat, horizon=2),
            "greedy": GreedyPolicy(env),
            "oracle_dp": oracle,
        }
        Js = {name: evaluate(pol, env, 0, 0, args.n_rollouts, seed=1000 + seed)
              for name, pol in policies.items()}
        rows.append({name: float(J.mean()) for name, J in Js.items()})
    out = {"seeds": args.seeds, "n_rollouts": args.n_rollouts, "per_seed": rows}
    agg = {name: {"mean": float(np.mean([r[name] for r in rows])),
                  "std": float(np.std([r[name] for r in rows]))}
           for name in rows[0]}
    out["aggregate"] = agg
    out["stochastic_minus_optimistic"] = paired_ci(
        np.array([r["A1_stochastic_depthH"] for r in rows]),
        np.array([r["A0_optimistic_depthH"] for r in rows]), seed=0)
    out["oracle_regret_stochastic"] = float(agg["oracle_dp"]["mean"] - agg["A1_stochastic_depthH"]["mean"])
    return out


def run_depth(args):
    """H3: regret vs (planner depth, response-model accuracy) at matched budgets."""
    env = EnvS(EnvSConfig(seed=0))
    oracle = DPPolicy(env)
    J_oracle = evaluate(oracle, env, 0, 0, args.n_rollouts, seed=999).mean()
    rows = []
    for train_size in (250, 750, 3000):
        # response accuracy is controlled by the training-set size (documented)
        for depth in (1, 2, 3, 6):
            Js = []
            for seed in range(args.seeds):
                trajs = gen_trajectories(env, "eps_greedy", train_size // 2, seed=30 + seed) + \
                    gen_trajectories(env, "uniform", train_size // 2, seed=40 + seed)
                resp = LogisticResponse()
                resp.fit(env, trajs)
                rho_hat = estimate_rho(env, trajs)
                pol = ModelDP(env, resp, rho_hat, horizon=depth)
                Js.append(evaluate(pol, env, 0, 0, args.n_rollouts, seed=2000 + seed).mean())
            rows.append({"train_size": train_size, "depth": depth,
                         "J_mean": float(np.mean(Js)), "J_std": float(np.std(Js)),
                         "regret_vs_oracle": float(J_oracle - np.mean(Js))})
    return {"J_oracle": float(J_oracle), "rows": rows}


def run_noise(args):
    """H3: regret vs (response-model error, planning depth) with latent logit noise.

    The response model's irreducible+estimation error is controlled by noise_scale and
    training-set size; the oracle uses noise-averaged probabilities.
    """
    rows = []
    for noise in (0.5, 1.5, 3.0):
        env = EnvS(EnvSConfig(seed=0, noise_scale=noise))
        oracle = DPPolicy(env)
        J_oracle = evaluate(oracle, env, 0, 0, args.n_rollouts, seed=999).mean()
        for train_size in (250, 1000, 3000):
            for depth in (2, 6):
                Js, briers = [], []
                for seed in range(args.seeds):
                    trajs = gen_trajectories(env, "eps_greedy", train_size // 2, seed=60 + seed) + \
                        gen_trajectories(env, "uniform", train_size // 2, seed=70 + seed)
                    resp = LogisticResponse()
                    resp.fit(env, trajs)
                    rho_hat = estimate_rho(env, trajs)
                    # held-out Brier of the response model (model-quality measure)
                    test_trajs = gen_trajectories(env, "eps_greedy", 200, seed=800 + seed)
                    yy = np.array([e for tr in test_trajs for (_, _, _, e, _) in tr], dtype=float)
                    pp = np.array([resp.predict(env, s, f, i) for tr in test_trajs for (s, f, i, _, _) in tr])
                    briers.append(brier_score(yy, pp))
                    pol = ModelDP(env, resp, rho_hat, horizon=depth)
                    Js.append(evaluate(pol, env, 0, 0, args.n_rollouts, seed=3000 + seed).mean())
                rows.append({"noise": noise, "train_size": train_size, "depth": depth,
                             "J_mean": float(np.mean(Js)), "J_std": float(np.std(Js)),
                             "regret_vs_oracle": float(J_oracle - np.mean(Js)),
                             "model_brier": float(np.mean(briers))})
    return {"rows": rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checks", action="store_true")
    parser.add_argument("--learn", action="store_true")
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--depth", action="store_true")
    parser.add_argument("--noise", action="store_true")
    parser.add_argument("--n-rollouts", type=int, default=4000)
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()
    t0 = time.time()

    out = {}
    if args.checks or not (args.learn or args.ablation or args.depth or args.noise):
        out["a1_checks"] = run_checks(args)
    if args.learn:
        out["a2_learn"] = run_learn(args)
    if args.ablation:
        out["a3_ablation"] = run_ablation(args)
    if args.depth:
        out["a4_depth"] = run_depth(args)
    if args.noise:
        out["a4_noise"] = run_noise(args)
    out["runtime_s"] = time.time() - t0
    print(json.dumps(out, indent=2))

    out_dir = Path(os.environ.get("MFMDP_OUTPUT_ROOT", "./outputs")) / "phase5"
    out_dir.mkdir(parents=True, exist_ok=True)
    name = "env_a_" + ("checks" if args.checks else "") + ("learn" if args.learn else "") + ("ablation" if args.ablation else "") + ("depth" if args.depth else "") + ("noise" if args.noise else "")
    with open(out_dir / f"{name}.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_dir / (name + '.json')} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
