"""H1-E1 — isolation experiment (design doc §3): does the belief/push conflation bind?

Env-L (learning-only) vs Env-P (preference-shaping-only), exact response model by
construction. Question: is B2 (belief-only, T̂=δ) regret ≈ 0 in Env-L and > 0 in Env-P
while B4 (joint) ≈ 0 in both?

Also runs the three pilot sign-off checks from the design doc's open items:
  pilot 1: greedy(myopic on true theta) != OracleState in Env-P (steering tension);
  pilot 2: JointMFBelief tracks theta better than MFPosteriorBelief in Env-P;
  pilot 3: OracleHistory == OracleState in Env-P (degenerate point-mass belief).

Usage: python experiments/h1/run_e1_isolation.py [--users 150] [--seeds 5] [--sims 60] [--pilot-only]
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

from mf_mdp.envs.h1_envs import H1Env, make_J, make_L, make_P
from mf_mdp.planning.stochastic_belief import (
    BeliefMCTSConfig,
    BeliefMCTSPolicy,
    MyopicPolicy,
    OracleHistoryPolicy,
    OracleStatePolicy,
)


def make_methods(env: H1Env, sims: int):
    mcfg = BeliefMCTSConfig(n_sims=sims, c_uct=1.4, depth=8, expansion=10)
    return {
        "B0_myopic": MyopicPolicy(env, env.initial_belief),
        "B1_paper_optimistic": BeliefMCTSPolicy(env, env.initial_belief,
                                                BeliefMCTSConfig(**{**mcfg.__dict__, "observe": "optimistic", "update": "foldin"})),
        "B2_belief_only": BeliefMCTSPolicy(env, env.initial_belief,
                                           BeliefMCTSConfig(**{**mcfg.__dict__, "observe": "sample", "update": "foldin"})),
        "B4_joint": BeliefMCTSPolicy(env, env.initial_belief_joint,
                                     BeliefMCTSConfig(**{**mcfg.__dict__, "observe": "sample", "update": "joint"})),
        "B4_joint_particles": OracleHistoryPolicy(env, BeliefMCTSConfig(**{**mcfg.__dict__, "observe": "sample", "update": "joint"}),
                                                  n_particles=200),
        "OracleState": OracleStatePolicy(env, n_sims=sims),
    }


def paired_ci(a, b, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    diff = a - b
    boots = [np.mean(rng.choice(diff, len(diff))) for _ in range(n_boot)]
    return {"mean": float(diff.mean()),
            "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]}


def run_pilots() -> dict:
    out = {}
    env = make_P(seed=0)
    rng = np.random.default_rng(1)
    thetas = [env.sample_theta0(np.random.default_rng(100 + s)) for s in range(50)]

    # pilot 1: greedy(myopic on true theta) vs OracleState J
    class GreedyTheta:
        is_oracle = True
        def reset(self, theta0, rng): return {"theta": theta0.copy(), "anchor": theta0.copy()}
        def act(self, state, f, t, rng):
            th = state["theta"]
            return int(max(range(env.n_items), key=lambda i: env.q[i] * float(th @ env.V[i])))
        def update(self, state, item, engaged): return state

    J_greedy = np.array([env.rollout(GreedyTheta(), th, np.random.default_rng(s)) for s, th in enumerate(thetas)])
    J_oracle = np.array([env.rollout(OracleStatePolicy(env, n_sims=200), th, np.random.default_rng(s)) for s, th in enumerate(thetas)])
    out["pilot1_steering_tension"] = paired_ci(J_oracle, J_greedy, seed=0)
    out["pilot1_pass"] = bool(out["pilot1_steering_tension"]["ci95"][0] > 0)

    # pilot 2: belief tracking in Env-P (open-loop fixed belief vs joint belief)
    dev_static, dev_joint = [], []
    for th in thetas[:30]:
        rng_i = np.random.default_rng(7)
        b_static = env.initial_belief(th, rng_i)
        b_joint = env.initial_belief_joint(th, rng_i)
        theta = th.copy()
        f = 0
        for t in range(8):
            i = int(np.argmax([env.q[j] * float(theta @ env.V[j]) for j in range(env.n_items)]))
            r, term, theta, nf, eng = env.step(theta, f, i, rng_i, th)
            b_static = b_static.update(env, i, True)
            b_joint = b_joint.update(env, i, True)
            f = nf
            if term:
                break
        dev_static.append(float(np.linalg.norm(b_static.mean() - theta)))
        dev_joint.append(float(np.linalg.norm(b_joint.mean() - theta)))
    out["pilot2_tracking_dev_static"] = float(np.mean(dev_static))
    out["pilot2_tracking_dev_joint"] = float(np.mean(dev_joint))
    out["pilot2_pass"] = out["pilot2_tracking_dev_joint"] < out["pilot2_tracking_dev_static"]

    # pilot 3: OracleHistory == OracleState in Env-P
    J_hist = np.array([env.rollout(OracleHistoryPolicy(env, BeliefMCTSConfig(n_sims=100), n_particles=150), th,
                                   np.random.default_rng(s)) for s, th in enumerate(thetas[:30])])
    out["pilot3_oracle_gap"] = paired_ci(J_oracle[:30], J_hist, seed=0)
    out["pilot3_pass"] = abs(out["pilot3_oracle_gap"]["mean"]) < 0.05
    return out


def run_e1(env_fn, users: int, seeds: int, sims: int, tag: str):
    rows = {}
    traces = {}
    for seed in range(seeds):
        env = env_fn(seed=seed)
        rng = np.random.default_rng(500 + seed)
        thetas = [env.sample_theta0(rng) for _ in range(users)]
        methods = make_methods(env, sims)
        for name, pol in methods.items():
            rets = np.array([env.rollout(pol, th, np.random.default_rng(1000 * seed + u)) for u, th in enumerate(thetas)])
            rows.setdefault(name, []).append(rets.mean())
            # belief tracking trace (B2 vs B4) on a few users
            if name in ("B2_belief_only", "B4_joint") and seed == 0:
                devs = []
                for th in thetas[:40]:
                    rng_i = np.random.default_rng(9)
                    state = pol.reset(th, rng_i)
                    theta, f = th.copy(), 0
                    for t in range(env.cfg.H):
                        i = pol.act(state, f, t, rng_i)
                        r, term, theta, nf, eng = env.step(theta, f, i, rng_i, th)
                        devs.append(float(np.linalg.norm(state.mean() - theta)))
                        state = pol.update(state, i, eng)
                        f = nf
                        if term:
                            break
                traces[name] = float(np.mean(devs))
    agg = {name: {"mean": float(np.mean(v)), "std": float(np.std(v))} for name, v in rows.items()}
    J_oracle = np.array(rows["OracleState"])
    regrets = {name: {"mean": float(J_oracle.mean() - v["mean"]),
                      "ci95": paired_ci(J_oracle, np.array(rows[name]), seed=0)["ci95"]}
               for name in agg}
    return {"tag": tag, "aggregate": agg, "regret_vs_oraclestate": regrets, "mean_deviation_theta": traces}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--users", type=int, default=150)
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--sims", type=int, default=60)
    parser.add_argument("--pilot-only", action="store_true")
    args = parser.parse_args()
    t0 = time.time()

    out = {"config": vars(args)}
    pilots = run_pilots()
    out["pilots"] = pilots
    print(json.dumps({"pilots": {k: v for k, v in pilots.items()}}, indent=2), flush=True)
    if not all(v for k, v in pilots.items() if k.endswith("_pass")):
        print("PILOT CHECKS FAILED — freezing config per design doc before E1.", flush=True)

    if not args.pilot_only:
        for tag, env_fn in (("Env-L", make_L), ("Env-P", make_P), ("Env-J", make_J)):
            res = run_e1(env_fn, args.users, args.seeds, args.sims, tag)
            out[tag] = res
            print(f"== {tag}: " + ", ".join(f"{k}={v['mean']:.3f}" for k, v in res["aggregate"].items()), flush=True)

    out["runtime_s"] = time.time() - t0
    out_dir = Path(os.environ.get("MFMDP_OUTPUT_ROOT", "./outputs")) / "h1"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "e1_isolation.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_dir / 'e1_isolation.json'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
