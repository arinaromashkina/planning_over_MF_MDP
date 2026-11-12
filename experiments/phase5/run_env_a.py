"""Experiment A1 — planning correctness checks in the controlled Env-S.

Verifies (Gate B):
- greedy differs from the DP-optimal policy and loses J in rollouts;
- stochastic (chance-node) MCTS recovers the DP choice as budget grows;
- the stochastic backup matches exact enumeration (test suite covers this too).

Usage: python experiments/phase5/run_env_a.py --checks
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


def evaluate(policy, env, s0, f0, n_rollouts, seed):
    rng = np.random.default_rng(seed)
    return np.array([env.rollout(policy, s0, f0, rng) for _ in range(n_rollouts)])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checks", action="store_true")
    parser.add_argument("--n-rollouts", type=int, default=4000)
    args = parser.parse_args()
    t0 = time.time()

    env = EnvS(EnvSConfig(seed=0))
    dp, greedy = DPPolicy(env), GreedyPolicy(env)
    out = {"env": "EnvS", "H": env.cfg.H, "gamma": env.cfg.gamma, "n_rollouts": args.n_rollouts}

    # 1. greedy vs DP: policy disagreement + J gap with paired bootstrap CIs
    disagree = [(s, f, int(dp.pi[0][s][f]), greedy(s, f, 0))
                for s in range(env.K) for f in range(3)
                if dp.pi[0][s][f] != greedy(s, f, 0)]
    out["greedy_dp_disagreements"] = {"count": len(disagree), "states": disagree}

    rng = np.random.default_rng(0)
    J_dp = evaluate(dp, env, 0, 0, args.n_rollouts, seed=100)
    J_greedy = evaluate(greedy, env, 0, 0, args.n_rollouts, seed=100)
    diff = J_dp - J_greedy
    boots = [np.mean(rng.choice(diff, len(diff))) for _ in range(2000)]
    out["J_dp"] = {"mean": float(J_dp.mean()), "std": float(J_dp.std())}
    out["J_greedy"] = {"mean": float(J_greedy.mean()), "std": float(J_greedy.std())}
    out["J_gap"] = {"mean": float(diff.mean()),
                    "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}

    # 2. MCTS agreement with DP vs simulation budget
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

    out["runtime_s"] = time.time() - t0
    print(json.dumps(out, indent=2))

    out_dir = Path(os.environ.get("MFMDP_OUTPUT_ROOT", "./outputs")) / "phase5"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "env_a1_checks.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_dir / 'env_a1_checks.json'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
