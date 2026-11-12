"""Learned world-model components for Env-S (Experiment A2/A3).

All learned components are trained on logged trajectories from behavior policies and
evaluated on the TRUE environment. The optimistic baseline assumes engagement with
probability 1 (the paper's optimistic fold-in analog).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mf_mdp.envs.discrete_env import EnvS


# ---------------------------------------------------------------- logging

def gen_trajectories(env: EnvS, policy_kind: str, n_traj: int, seed: int) -> list[list[tuple]]:
    """Log trajectories (s, f, i, engaged, terminated) under a behavior policy."""
    rng = np.random.default_rng(seed)
    trajs = []
    for _ in range(n_traj):
        s, f = 0, 0
        traj = []
        for t in range(env.cfg.H):
            if policy_kind == "eps_greedy":
                i = max(range(env.n_items), key=lambda i: env.expected_immediate_reward(s, f, i))
                if rng.random() < 0.3:
                    i = int(rng.integers(env.n_items))
            elif policy_kind == "uniform":
                i = int(rng.integers(env.n_items))
            else:
                raise ValueError(policy_kind)
            engaged = rng.random() < env.engagement_prob(s, i)
            r = env.reward(s, f, i, engaged)
            term = rng.random() < env.termination_prob(r)
            traj.append((s, f, i, int(engaged), int(term)))
            if term:
                break
            s = env.next_mode(s, i, engaged, rng)
            f = env.next_fatigue(s, f, i) if engaged else f
        trajs.append(traj)
    return trajs


# ---------------------------------------------------------------- response model

@dataclass
class LogisticResponse:
    """p(y=1 | s, f, i) logistic regression on [onehot(s,f), m_i[s], q_i]."""

    l2: float = 1e-4
    lr: float = 0.5
    n_steps: int = 800
    weights: np.ndarray | None = None
    bias: float = 0.0
    n_states: int = 12

    def _features(self, env: EnvS, s: int, f: int, i: int) -> np.ndarray:
        oh = np.zeros(self.n_states)
        oh[s * 3 + f] = 1.0
        return np.concatenate([oh, [env.M[i][s], env.q[i]]])

    def fit(self, env: EnvS, trajs: list[list[tuple]]) -> None:
        X, y = [], []
        for traj in trajs:
            for s, f, i, e, _ in traj:
                X.append(self._features(env, s, f, i))
                y.append(float(e))
        X, y = np.vstack(X), np.asarray(y)
        n, d = X.shape
        w, b = np.zeros(d), 0.0
        for _ in range(self.n_steps):
            p = 1.0 / (1.0 + np.exp(-np.clip(X @ w + b, -30, 30)))
            g = (p - y) / n
            w -= self.lr * (X.T @ g + self.l2 * w)
            b -= self.lr * float(g.sum())
        self.weights, self.bias = w, b

    def predict(self, env: EnvS, s: int, f: int, i: int) -> float:
        x = self._features(env, s, f, i)
        z = float(x @ self.weights + self.bias)
        return float(1.0 / (1.0 + np.exp(-np.clip(z, -30, 30))))


def estimate_rho(env: EnvS, trajs: list[list[tuple]]) -> float:
    """MLE of the assimilation probability from engagement transitions."""
    moved, tot = 0, 0
    for traj in trajs:
        for t, (s, f, i, e, term) in enumerate(traj):
            if e and t + 1 < len(traj):
                dominant = int(np.argmax(env.M[i]))
                if dominant != s:
                    tot += 1
                    moved += int(traj[t + 1][0] == dominant)
    return moved / tot if tot > 0 else env.cfg.rho


# ---------------------------------------------------------------- model-based DP

class ModelDP:
    """Backward induction under a LEARNED response model p̂ and transition estimate.

    optimistic=True reproduces the paper's assumption: engagement with probability 1
    and assimilation with probability 1 (the deterministic positive fold-in analog).
    """

    def __init__(self, env: EnvS, resp: LogisticResponse | None, rho_hat: float,
                 horizon: int | None = None, optimistic: bool = False):
        self.env = env
        self.optimistic = optimistic
        self.resp = resp
        self.rho_hat = rho_hat
        c = env.cfg
        H = horizon if horizon is not None else c.H
        self.H = H
        K, N = env.K, env.n_items
        self.V = np.zeros((H + 1, K, 3))
        self.pi = np.zeros((H, K, 3), dtype=int)
        for h in range(H - 1, -1, -1):
            for s in range(K):
                for f in range(3):
                    best_q, best_i = -np.inf, 0
                    for i in range(N):
                        p = 1.0 if optimistic else resp.predict(env, s, f, i)
                        rho = 1.0 if optimistic else self.rho_hat
                        r_eng = env.reward(s, f, i, True)
                        te = env.termination_prob(r_eng)
                        dominant = int(np.argmax(env.M[i]))
                        if dominant == s:
                            eng_next = [(s, env.next_fatigue(s, f, i), 1.0)]
                        else:
                            eng_next = [(dominant, 0, rho), (s, 0, 1.0 - rho)]
                        q = sum(prob * (r_eng + c.gamma * (1.0 - te) * self.V[h + 1][ns][nf])
                                for ns, nf, prob in eng_next) * p
                        ts = env.termination_prob(0.0)
                        q += (1.0 - p) * (c.gamma * (1.0 - ts) * self.V[h + 1][s][f])
                        if q > best_q:
                            best_q, best_i = q, i
                    self.V[h][s][f] = best_q
                    self.pi[h][s][f] = best_i

    def __call__(self, s: int, f: int, t: int) -> int:
        return int(self.pi[min(t, self.H - 1)][s][f])
