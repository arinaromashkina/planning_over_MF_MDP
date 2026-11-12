"""Env-S: small discrete stochastic recommendation environment with exact DP oracle.

Generative model (frozen, docs/PHASE5_EXPERIMENT_PLAN.md):

- K=4 interest modes, N=10 items. Item i has affinity m_i in [0,1]^K (row-normalized)
  and quality q_i in [0,1]. User state (s, f): mode s in {0..K-1}, fatigue f in {0,1,2}.
- Engagement probability: p_i(s) = sigmoid(a * (m_i[s] - theta0) + b * q_i).
- Reward on engagement: r = q_i * m_i[s] * (1 - kappa_sat * f); on skip r = 0.
- Termination after each step: h(r) = sigmoid(c0 - c1 * r); terminated -> no future reward.
- Mode transition on engagement: s' = argmax_k m_i[k] w.p. rho (assimilation), else s.
  On skip: s' = s. Fatigue: f' = min(2, f+1) if argmax_k m_i[k] == s else 0.
- Horizon H, discount gamma.

Handcrafted item set (deterministic, documented) creates the greedy-vs-optimal tension:
high-quality items exist off the initial mode, so steering (assimilation) pays off over
the horizon while repetition builds fatigue.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnvSConfig:
    K: int = 4
    n_items: int = 10
    H: int = 6
    gamma: float = 0.9
    rho: float = 0.6  # assimilation probability
    a: float = 4.0  # engagement sharpness on affinity margin
    b: float = 1.0  # engagement bias from quality
    theta0: float = 0.5
    kappa_sat: float = 0.35  # fatigue penalty per level
    c0: float = -1.5  # termination hazard intercept
    c1: float = 4.0  # termination hazard slope on reward
    noise_scale: float = 0.0  # per-step latent logit noise on engagement (H3 regime)
    seed: int = 0


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# Handcrafted catalog: rows are (m[0..3], q). Two high-quality items are anchored
# off-mode (items 6, 7 dominate mode 2; items 8, 9 dominate mode 3) so that steering
# away from the initial mode can beat myopic play. Documented in the experiment plan.
_ITEMS = [
    ([0.70, 0.10, 0.10, 0.10], 0.45),  # 0: mode-0 filler
    ([0.55, 0.25, 0.10, 0.10], 0.55),  # 1: mode-0 decent
    ([0.10, 0.70, 0.10, 0.10], 0.50),  # 2: mode-1 filler
    ([0.15, 0.60, 0.15, 0.10], 0.60),  # 3: mode-1 good
    ([0.10, 0.10, 0.75, 0.05], 0.80),  # 4: mode-2 HIGH quality (off-mode anchor)
    ([0.10, 0.15, 0.65, 0.10], 0.70),  # 5: mode-2 good
    ([0.05, 0.05, 0.80, 0.10], 0.95),  # 6: mode-2 BEST (off-mode anchor)
    ([0.10, 0.10, 0.20, 0.60], 0.75),  # 7: mode-3 good
    ([0.05, 0.10, 0.10, 0.75], 0.90),  # 8: mode-3 BEST
    ([0.25, 0.25, 0.25, 0.25], 0.35),  # 9: generic low-quality
]


class EnvS:
    """Discrete stochastic environment. All randomness through the supplied rng."""

    def __init__(self, cfg: EnvSConfig = EnvSConfig()):
        self.cfg = cfg
        M = np.array([x[0] for x in _ITEMS], dtype=float)
        self.M = M / M.sum(axis=1, keepdims=True)
        self.q = np.array([x[1] for x in _ITEMS], dtype=float)
        self.n_items = cfg.n_items
        self.K = cfg.K

    # ---- model primitives -------------------------------------------------

    def engagement_prob(self, s: int, i: int, eps: float = 0.0) -> float:
        c = self.cfg
        return float(_sigmoid(c.a * (self.M[i][s] - c.theta0) + c.b * self.q[i] + eps))

    def engagement_prob_avg(self, s: int, i: int, n_mc: int = 4000, seed: int = 0) -> float:
        """Noise-averaged engagement probability E_eps[sigmoid(logit + eps)].

        With per-step i.i.d. logit noise this is the correct planning quantity
        (the noise only adds Bernoulli variance). Used by the oracle in noisy regimes.
        """
        if self.cfg.noise_scale == 0.0:
            return self.engagement_prob(s, i)
        rng = np.random.default_rng(seed + s * 131 + i * 17)
        eps = rng.normal(0.0, self.cfg.noise_scale, n_mc)
        return float(np.mean(_sigmoid(self.cfg.a * (self.M[i][s] - self.cfg.theta0) + self.cfg.b * self.q[i] + eps)))

    def reward(self, s: int, f: int, i: int, engaged: bool) -> float:
        if not engaged:
            return 0.0
        return float(self.q[i] * self.M[i][s] * (1.0 - self.cfg.kappa_sat * f))

    def termination_prob(self, r: float) -> float:
        c = self.cfg
        return float(_sigmoid(c.c0 - c.c1 * r))

    def next_mode(self, s: int, i: int, engaged: bool, rng: np.random.Generator) -> int:
        if not engaged:
            return s
        dominant = int(np.argmax(self.M[i]))
        if dominant == s:
            return s
        return dominant if rng.random() < self.cfg.rho else s

    def next_fatigue(self, s: int, f: int, i: int) -> int:
        return min(2, f + 1) if int(np.argmax(self.M[i])) == s else 0

    def step(self, s: int, f: int, i: int, rng: np.random.Generator):
        """One environment step. Returns (reward, terminated, next_s, next_f)."""
        eps = rng.normal(0.0, self.cfg.noise_scale) if self.cfg.noise_scale > 0 else 0.0
        engaged = rng.random() < self.engagement_prob(s, i, eps)
        r = self.reward(s, f, i, engaged)
        terminated = rng.random() < self.termination_prob(r)
        ns = self.next_mode(s, i, engaged, rng)
        nf = self.next_fatigue(s, f, i) if engaged else f
        return r, terminated, ns, nf

    def expected_immediate_reward(self, s: int, f: int, i: int) -> float:
        p = self.engagement_prob_avg(s, i)
        return p * self.reward(s, f, i, True)

    # ---- evaluation ---------------------------------------------------------

    def rollout(self, policy, s0: int, f0: int, rng: np.random.Generator) -> float:
        """Undiscounted-by-caller discounted return of `policy` over the horizon."""
        g, total = 0.0, 0.0
        s, f = s0, f0
        for t in range(self.cfg.H):
            i = policy(s, f, t)
            r, term, ns, nf = self.step(s, f, i, rng)
            total += (self.cfg.gamma ** t) * r
            if term:
                break
            s, f = ns, nf
        return total


class GreedyPolicy:
    """Myopic: maximize expected immediate reward."""

    def __init__(self, env: EnvS):
        self.env = env

    def __call__(self, s: int, f: int, t: int) -> int:
        return max(range(self.env.n_items),
                   key=lambda i: self.env.expected_immediate_reward(s, f, i))


class DPPolicy:
    """Exact backward-induction oracle over the 12-state environment.

    The engaged branch must average over assimilation failure: engagement moves the
    user to the item's dominant mode w.p. rho, else keeps the mode (fatigue rules
    unchanged). Found via Monte-Carlo cross-check against an MCTS implementation
    (the DP initially ignored assimilation failure; the MCTS was right).
    """

    def __init__(self, env: EnvS):
        self.env = env
        c = env.cfg
        H, K, N = c.H, env.K, env.n_items
        self.V = np.zeros((H + 1, K, 3))  # V[h][s][f], V[H]=0 (end of horizon)
        self.pi = np.zeros((H, K, 3), dtype=int)
        for h in range(H - 1, -1, -1):
            for s in range(K):
                for f in range(3):
                    best_q, best_i = -np.inf, 0
                    for i in range(N):
                        p = env.engagement_prob_avg(s, i)
                        r_eng = env.reward(s, f, i, True)
                        term_eng = env.termination_prob(r_eng)
                        dominant = int(np.argmax(env.M[i]))
                        # engaged branch: expectation over assimilation outcomes
                        if dominant == s:
                            eng_next = [(s, env.next_fatigue(s, f, i), 1.0)]
                        else:
                            eng_next = [(dominant, 0, c.rho), (s, 0, 1.0 - c.rho)]
                        q = 0.0
                        for ns, nf, prob in eng_next:
                            q += prob * (r_eng + c.gamma * (1.0 - term_eng) * self.V[h + 1][ns][nf])
                        q *= p
                        term_skip = env.termination_prob(0.0)
                        q += (1.0 - p) * (c.gamma * (1.0 - term_skip) * self.V[h + 1][s][f])
                        if q > best_q:
                            best_q, best_i = q, i
                    self.V[h][s][f] = best_q
                    self.pi[h][s][f] = best_i

    def __call__(self, s: int, f: int, t: int) -> int:
        return int(self.pi[t][s][f])


class MCTSPolicy:
    """Chance-node UCT using the TRUE environment as simulator (verification tool).

    This is a search-verification instrument, not the learned-model planner.
    Nodes are (h, s, f); action values are estimated by sampling outcomes
    (engagement + termination), with random rollouts at the frontier.
    """

    def __init__(self, env: EnvS, n_sims: int = 2000, c_uct: float = 1.4, rollout_greedy: bool = False):
        self.env = env
        self.n_sims = n_sims
        self.c_uct = c_uct
        self.gamma = env.cfg.gamma
        self.rollout_greedy = rollout_greedy

    def _rollout(self, s: int, f: int, h: int, rng: np.random.Generator) -> float:
        env = self.env
        if h >= env.cfg.H:
            return 0.0
        if self.rollout_greedy:
            i = max(range(env.n_items), key=lambda i: env.expected_immediate_reward(s, f, i))
        else:
            i = int(rng.integers(env.n_items))
        r, term, ns, nf = env.step(s, f, i, rng)
        if term:
            return r
        return r + self.gamma * self._rollout(ns, nf, h + 1, rng)

    def _uct(self, node) -> int:
        log_n = np.log(max(node["N"], 2))
        best_a, best_v = 0, -np.inf
        for a, child in node["children"].items():
            v = child["Q"] + self.c_uct * np.sqrt(log_n / max(child["N"], 1))
            if v > best_v:
                best_v, best_a = v, a
        return best_a

    def _sim(self, node: dict, s: int, f: int, h: int, rng: np.random.Generator) -> float:
        """Simulate one pass; returns the discounted return from this node onward.

        Chance-node structure: each action has statistics (N, Q over sampled returns)
        and a dict of per-outcome child nodes keyed by (ns, nf), so continuations are
        never mixed across different next states.
        """
        env = self.env
        node.setdefault("N", 0)
        node.setdefault("children", {})
        node["N"] += 1
        if not node["children"]:
            for i in range(env.n_items):
                node["children"][i] = {"N": 0, "Q": 0.0, "nodes": {}}
            return self._rollout(s, f, h, rng)
        i = self._uct(node)
        child = node["children"][i]
        r, term, ns, nf = env.step(s, f, i, rng)
        if term or h + 1 >= env.cfg.H:
            G = r
        else:
            next_node = child["nodes"].setdefault((ns, nf), {"N": 0, "children": {}})
            G = r + self.gamma * self._sim(next_node, ns, nf, h + 1, rng)
        child["N"] += 1
        child["Q"] += (G - child["Q"]) / child["N"]
        return G

    def action(self, s: int, f: int, rng: np.random.Generator) -> int:
        root = {"N": 0, "children": {}}
        for _ in range(self.n_sims):
            self._sim(root, s, f, 0, rng)
        return max(root["children"], key=lambda a: root["children"][a]["Q"])

    def __call__(self, s: int, f: int, t: int) -> int:
        rng = np.random.default_rng(abs(hash((s, f, t, self.n_sims))) % (2**31))
        return self.action(s, f, rng)
