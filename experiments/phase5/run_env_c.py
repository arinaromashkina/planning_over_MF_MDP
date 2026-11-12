"""Experiment A on Env-C: optimistic vs stochastic planning over the MF posterior,
evaluated in the TRUE continuous environment.

Pipeline per seed:
1. Log trajectories from a behavior policy (eps-greedy on true quality-adjusted match).
2. Train iALS factors on the logged interactions.
3. Learn response model p(y | MF decision features).
4. Plan with (a) optimistic 2-step lookahead (paper T0 analog: assume engagement) and
   (b) stochastic 2-step lookahead (sample y ~ p̂, exact weighted fold-in transitions).
5. Evaluate J in the TRUE Env-C from held-out initial states.

Usage: python experiments/phase5/run_env_c.py [--seeds 5] [--n-users 200]
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

from mf_mdp.envs.continuous_env import EnvC, EnvCConfig, gen_theta0s
from mf_mdp.mdp import MFState, fold_in_general
from mf_mdp.mf import IALSConfig, train_ials, user_state
from mf_mdp.models.features import FeatureEncoder, features_for
from mf_mdp.models.response import LogisticResponseModel, brier_score, expected_calibration_error, log_loss, roc_auc
from mf_mdp.planning.candidates import exact_cosine_topl


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
            traj.append((i, int(engaged), int(term)))
            consumed.add(i)
            if term:
                break
            f = nf
        trajs.append(traj)
    return trajs


def mf_state_of(history, V, cfg):
    return user_state(V, np.asarray(history, dtype=np.int64), cfg)


def resp_features(st, V, i, log_pop):
    return features_for(st, V, i, log_pop)


class StochasticPlanner:
    """2-step lookahead over the MF posterior with learned response + termination heads
    and exact weighted fold-in transitions.

    Bellman form (prompt §6): Q(s,a) = E_y[ r(s,a,y) + gamma * (1 - h(s,a)) * V(s') ].
    optimistic=True sets p(y=engage)=1 and h=0 (the paper's optimistic analog).
    """

    def __init__(self, V, ials_cfg, resp, enc, log_pop, pool_size=8, optimistic=False, hresp=None):
        self.V, self.cfg, self.resp, self.enc, self.log_pop = V, ials_cfg, resp, enc, log_pop
        self.pool = pool_size
        self.optimistic = optimistic
        self.hresp = hresp

    def _p_eng(self, st, i):
        if self.optimistic:
            return 1.0
        x = self.enc.transform(resp_features(st, self.V, i, self.log_pop)[None, :])
        return float(self.resp.predict_proba(x)[0])

    def _p_term(self, st, i):
        if self.optimistic or self.hresp is None:
            return 0.0
        x = self.enc.transform(resp_features(st, self.V, i, self.log_pop)[None, :])
        return float(self.hresp.predict_proba(x)[0])

    def _rhat(self, st, i, f_streak):
        v = self.V[i]
        cos = float(st.u @ v) / (np.linalg.norm(st.u) * np.linalg.norm(v) + 1e-12)
        return float(self.envq[i] * max(cos, 0.0) * (1.0 - self.kappa * f_streak))

    def plan(self, st, f_streak, k, rng):
        if np.linalg.norm(st.u) == 0.0:
            # cold start (empty history): popularity pool (documented rule)
            pool = np.argsort(-self.envq)[: self.pool]
        else:
            pool = exact_cosine_topl(self.V, st.u, st.consumed, self.pool)
        best_val, best_i = -np.inf, int(pool[0])
        for a in pool:
            va = self.V[a]
            p = self._p_eng(st, a)
            # outcome branches (Bellman: gamma * (1 - termination) * continuation)
            h_est = self._p_term(st, a)
            q_val = 0.0
            for engaged, prob in ((1, p), (0, 1.0 - p)):
                if prob == 0.0:
                    continue
                if engaged:
                    P_p, u_p, _ = fold_in_general(st, va, self.cfg.alpha, 1.0 + self.cfg.alpha)
                else:
                    P_p, u_p, _ = fold_in_general(st, va, -0.5 * self.cfg.alpha, 0.0)
                st1 = MFState(P=P_p, u=u_p, consumed=tuple(sorted(st.consumed + (a,))))
                r1 = self._rhat(st, a, f_streak) if engaged else 0.0
                # depth-2: best next-step reward under the same response model
                pool2 = exact_cosine_topl(self.V, st1.u, st1.consumed, self.pool)
                best2 = 0.0
                for b in pool2:
                    pb = self._p_eng(st1, b)
                    best2 = max(best2, pb * self._rhat(st1, b, f_streak + engaged))
                q_val += prob * (r1 + (1.0 - h_est) * self.cfg_gamma * best2)
            if q_val > best_val:
                best_val, best_i = q_val, int(a)
        return best_i


def run_seed(seed, n_train_users, n_eval_users):
    env = EnvC(EnvCConfig(seed=seed))
    rng = np.random.default_rng(seed)
    ials_cfg = IALSConfig(dim=16, alpha=10.0, lam=0.1, n_iter=15)
    train_thetas = gen_theta0s(n_train_users, env, seed=1000 + seed)
    trajs = log_trajectories(env, train_thetas, rng)
    positives = np.array([(u, t[0]) for u, tr in enumerate(trajs) for t in tr], dtype=np.int64)
    _, V = train_ials(positives, n_train_users, env.n_items, ials_cfg, seed=seed)

    # response model on MF features
    log_pop = np.log1p(np.bincount(positives[:, 1], minlength=env.n_items).astype(float))
    X, y, y_term = [], [], []
    for u, tr in enumerate(trajs):
        hist = []
        st = mf_state_of(hist, V, ials_cfg)
        for (i, e, term) in tr:
            X.append(resp_features(st, V, i, log_pop))
            y.append(float(e))
            y_term.append(float(term))
            hist.append(i)
            st = mf_state_of(hist, V, ials_cfg)
    X, y, y_term = np.vstack(X), np.asarray(y), np.asarray(y_term)
    enc = FeatureEncoder()
    resp = LogisticResponseModel(l2=1e-3, lr=0.5, n_steps=800)
    resp.fit(enc.fit_transform(X), y)
    hresp = LogisticResponseModel(l2=1e-3, lr=0.5, n_steps=800)
    hresp.fit(enc.transform(X), y_term)
    p = resp.predict_proba(enc.transform(X))
    ph = hresp.predict_proba(enc.transform(X))
    resp_metrics = {"log_loss": log_loss(y, p), "brier": brier_score(y, p),
                    "roc_auc": roc_auc(y, p), "ece": expected_calibration_error(y, p),
                    "base_rate": float(y.mean()),
                    "term_log_loss": log_loss(y_term, ph), "term_brier": brier_score(y_term, ph),
                    "term_roc_auc": roc_auc(y_term, ph), "term_ece": expected_calibration_error(y_term, ph),
                    "term_base_rate": float(y_term.mean())}

    # planners
    planners = {}
    for name, optimistic, hmod in (("optimistic", True, None), ("stochastic", False, hresp)):
        pl = StochasticPlanner(V, ials_cfg, resp, enc, log_pop, pool_size=8, optimistic=optimistic, hresp=hmod)
        pl.envq = env.q
        pl.kappa = env.cfg.kappa_sat
        pl.cfg_gamma = env.cfg.gamma
        planners[name] = pl

    class GreedyMF:
        """No planning: rank by immediate expected utility q*cos(u,v)*(1-kappa f)."""

        def __init__(self, pl):
            self.pl = pl

        def plan(self, st, f_streak, k, rng):
            import numpy as _np
            if _np.linalg.norm(st.u) == 0.0:
                pool = _np.argsort(-self.pl.envq)[: self.pl.pool]
            else:
                pool = exact_cosine_topl(self.pl.V, st.u, st.consumed, self.pl.pool)
            vals = [self.pl._p_eng(st, int(a)) * self.pl._rhat(st, int(a), f_streak) for a in pool]
            return int(pool[int(_np.argmax(vals))])

    planners["greedy_mf"] = GreedyMF(planners["stochastic"])

    def greedy_policy(theta, f, t, consumed=frozenset()):
        avail = [i for i in range(env.n_items) if i not in consumed] or list(range(env.n_items))
        return max(avail, key=lambda i: env.q[i] * float(theta @ env.V[i]))

    # evaluation in the TRUE env; f tracked from OBSERVED outcomes (engagements)
    eval_thetas = gen_theta0s(n_eval_users, env, seed=5000 + seed)
    results = {"response_model": resp_metrics}
    for name, pl in planners.items():
        rets = []
        for theta in eval_thetas:
            th, f_obs, hist = theta.copy(), 0, []
            total = 0.0
            for t in range(env.cfg.H):
                st = mf_state_of(hist, V, ials_cfg)
                i = pl.plan(st, f_obs, 1, rng)
                r, term, th, nf, engaged = env.step(th, f_obs, i, rng)
                total += (env.cfg.gamma ** t) * r
                hist.append(i)
                f_obs = min(2, f_obs + int(engaged))  # fatigue is observable from outcomes
                if term:
                    break
            rets.append(total)
        results[name] = {"J_mean": float(np.mean(rets)), "J_std": float(np.std(rets))}
    # true-greedy reference (consumption excluded, same as real recommenders)
    rets = []
    for theta in eval_thetas:
        rng2 = np.random.default_rng(seed + 77)
        th, f, consumed = theta.copy(), 0, set()
        total = 0.0
        for t in range(env.cfg.H):
            i = greedy_policy(th, f, t, consumed)
            r, term, th, nf, engaged = env.step(th, f, i, rng2)
            total += (env.cfg.gamma ** t) * r
            consumed.add(i)
            if term:
                break
            f = nf
        rets.append(total)
    results["true_greedy"] = {"J_mean": float(np.mean(rets)), "J_std": float(np.std(rets))}
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5)
    parser.add_argument("--n-train-users", type=int, default=400)
    parser.add_argument("--n-eval-users", type=int, default=200)
    args = parser.parse_args()
    t0 = time.time()
    per_seed = [run_seed(s, args.n_train_users, args.n_eval_users) for s in range(args.seeds)]
    agg = {}
    for name in ("optimistic", "stochastic", "greedy_mf", "true_greedy"):
        vals = [r[name]["J_mean"] for r in per_seed]
        agg[name] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
    resp_m = per_seed[0]["response_model"]
    diff = [r["stochastic"]["J_mean"] - r["optimistic"]["J_mean"] for r in per_seed]
    rng = np.random.default_rng(0)
    boots = [np.mean(rng.choice(diff, len(diff))) for _ in range(2000)]
    diff_g = [r["stochastic"]["J_mean"] - r["greedy_mf"]["J_mean"] for r in per_seed]
    boots_g = [np.mean(rng.choice(diff_g, len(diff_g))) for _ in range(2000)]
    out = {
        "config": {"seeds": args.seeds, "n_train_users": args.n_train_users, "n_eval_users": args.n_eval_users},
        "aggregate": agg,
        "stochastic_minus_optimistic": {"mean": float(np.mean(diff)),
                                        "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))]},
        "stochastic_minus_greedy_mf": {"mean": float(np.mean(diff_g)),
                                       "ci95": [float(np.quantile(boots_g, 0.025)), float(np.quantile(boots_g, 0.975))]},
        "response_model_seed0": resp_m,
        "per_seed": per_seed,
        "runtime_s": time.time() - t0,
    }
    print(json.dumps(out, indent=2))
    out_dir = Path(os.environ.get("MFMDP_OUTPUT_ROOT", "./outputs")) / "phase5"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "env_c_ablation.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {out_dir / 'env_c_ablation.json'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
