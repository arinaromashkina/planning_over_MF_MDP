"""Chance-node planning over beliefs + policy variants B0-B4 and oracles (design doc §3).

Harness interface (used by H1Env.rollout):
    state = policy.reset(theta0, rng)
    i     = policy.act(state, f, t, rng)
    state = policy.update(state, item, engaged)

Belief policies store only a Belief in `state`; oracles store the true theta (legitimate
oracle access, flagged). Budgets are matched by construction (same n_sims / depth / UCT);
particle variants cost more per node — reported by the runner.

Variants:
  B0 Myopic:        act greedily on belief expected reward; fold-in updates.
  B1 MF-MDP paper:  belief MF; observation = optimistic (y=+1); fold-in updates.
  B2 Belief-only:   belief MF; observation sampled ~ predictive; fold-in updates.
  B3 Dyn-only:      belief point-mass; open-loop push by expected response (no observation use).
  B4 Joint:         belief JointMF (or particles); observation sampled; joint update.
  OracleState:      MCTS in the true env from theta.
  OracleHistory:    particles + true kernels + BeliefMCTS.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mf_mdp.belief import Belief, JointMFBelief, MFPosteriorBelief, ParticleBelief
from mf_mdp.envs.h1_envs import H1Env
from mf_mdp.planning.candidates import exact_cosine_topl


# --------------------------------------------------------------------------- B0

class MyopicPolicy:
    is_oracle = False

    def __init__(self, env: H1Env, belief_fn):
        self.env = env
        self.belief_fn = belief_fn

    def reset(self, theta0, rng):
        return self.belief_fn(theta0, rng)

    def act(self, state, f, t, rng):
        return int(max(range(self.env.n_items), key=lambda i: state.expected_reward(self.env, i, f)))

    def update(self, state, item, engaged):
        return state.update(self.env, item, engaged)


# --------------------------------------------------------------------------- MCTS

@dataclass
class BeliefMCTSConfig:
    n_sims: int = 100
    c_uct: float = 1.4
    depth: int = 8
    expansion: int = 10
    observe: str = "sample"      # 'sample' | 'optimistic'
    update: str = "foldin"       # 'foldin' | 'joint' | 'openloop'


class BeliefMCTSPolicy:
    is_oracle = False

    def __init__(self, env: H1Env, belief_fn, cfg: BeliefMCTSConfig):
        self.env = env
        self.belief_fn = belief_fn
        self.cfg = cfg

    def reset(self, theta0, rng):
        return self.belief_fn(theta0, rng)

    # -- observation/update variants --
    def _outcome(self, state: Belief, item: int, rng) -> bool:
        if self.cfg.observe == "optimistic":
            return True
        return rng.random() < state.predictive_prob(self.env, item)

    def _update(self, state: Belief, item: int, engaged: bool) -> Belief:
        if self.cfg.update == "openloop":
            return self._openloop_push(state, item)
        return state.update(self.env, item, engaged)

    def _openloop_push(self, state: Belief, item: int) -> Belief:
        """B3: point-mass-style push by the expected response; no fold-in."""
        p = state.predictive_prob(self.env, item)
        c = self.env.cfg
        a_exp = p * c.alpha_pos + (1 - p) * c.alpha_neg
        u = state.mean()
        anchor = u  # open-loop tracker anchors at its own mean
        new = (1 - c.lam) * u + a_exp * self.env.V[item] + c.lam * anchor
        nrm = np.linalg.norm(new)
        new = new / nrm if nrm > 0 else u
        if isinstance(state, JointMFBelief):
            return JointMFBelief(new, state.P, state.lam, state.theta_bar, state.Q)
        return MFPosteriorBelief(new, getattr(state, "P", np.zeros((self.env.d, self.env.d))))

    # -- one simulation pass --
    def _rollout_value(self, state: Belief, f: int, depth: int, rng) -> float:
        """Random rollout from `state` for `depth` steps (uniform policy)."""
        env, g = self.env, self.env.cfg.gamma
        total, d = 0.0, 0
        while d < depth:
            i = int(rng.integers(env.n_items))
            engaged = self._outcome(state, i, rng)
            r = state.expected_reward(env, i, f) if engaged else 0.0
            total += (g ** d) * r
            state = self._update(state, i, engaged)
            f = min(2, f + 1) if engaged else f
            d += 1
        return total

    def _sim(self, node, state: Belief, f: int, depth_left: int, t: int, rng) -> float:
        """Chance-node UCT recursion. node: dict(children, N, edge stats)."""
        env = self.env
        if depth_left <= 0 or t >= env.cfg.H:
            return self._rollout_value(state, f, self.cfg.depth, rng) if depth_left > 0 else 0.0
        node["N"] = node.get("N", 0) + 1
        if not node.get("children"):
            node["children"] = {}
            for i in exact_cosine_topl(env.V, state.mean(), (), self.cfg.expansion):
                node["children"][int(i)] = {"N": 0, "Q": 0.0}
            return self._rollout_value(state, f, depth_left, rng)
        i = self._uct(node)
        child = node["children"][i]
        engaged = self._outcome(state, i, rng)
        r = state.expected_reward(env, i, f) if engaged else 0.0
        term = rng.random() < (1.0 / (1.0 + np.exp(-(env.cfg.c0 - env.cfg.c1 * float(state.mean() @ env.V[i])))))
        if term:
            G = r
        else:
            sub = child.setdefault("node", {"children": {}, "N": 0})
            G = r + env.cfg.gamma * self._sim(sub, self._update(state, i, engaged),
                                              min(2, f + 1) if engaged else f,
                                              depth_left - 1, t + 1, rng)
        child["N"] += 1
        child["Q"] += (G - child["Q"]) / child["N"]
        return G

    def _uct(self, node) -> int:
        log_n = np.log(max(node["N"], 2))
        best_a, best_v = 0, -np.inf
        for a, child in node["children"].items():
            v = child["Q"] + self.cfg.c_uct * np.sqrt(log_n / max(child["N"], 1))
            if v > best_v:
                best_v, best_a = v, a
        return best_a

    def act(self, state, f, t, rng):
        root = {"children": {}, "N": 0}
        depth_cap = min(self.cfg.depth, self.env.cfg.H - t)
        for _ in range(self.cfg.n_sims):
            self._sim(root, state, f, depth_cap, t, rng)
        return max(root["children"], key=lambda a: root["children"][a]["Q"])

    def update(self, state, item, engaged):
        if self.cfg.observe == "optimistic":
            engaged = True  # B1: the paper's observation model (y=+ assumed)
        return self._update(state, item, engaged)


# --------------------------------------------------------------------------- oracles

class OracleStatePolicy:
    """MCTS directly in the true environment (upper bound; accesses theta)."""
    is_oracle = True

    def __init__(self, env: H1Env, n_sims: int = 300, c_uct: float = 1.4, depth: int = 8, expansion: int = 10):
        self.env = env
        self.n_sims = n_sims
        self.c_uct = c_uct
        self.depth = depth
        self.expansion = expansion

    def reset(self, theta0, rng):
        return {"theta": theta0.copy(), "anchor": theta0.copy()}

    def _rollout(self, theta, anchor, f, depth, rng) -> float:
        env, g = self.env, self.env.cfg.gamma
        total, d = 0.0, 0
        while d < depth:
            i = int(rng.integers(env.n_items))
            r, term, theta, nf, eng = env.step(theta, f, i, rng, anchor)
            total += (g ** d) * r
            if term:
                break
            f = nf
            d += 1
        return total

    def _sim(self, node, theta, anchor, f, depth_left, rng) -> float:
        env = self.env
        if depth_left <= 0:
            return self._rollout(theta, anchor, f, self.depth, rng)
        node["N"] = node.get("N", 0) + 1
        if not node.get("children"):
            node["children"] = {}
            for i in exact_cosine_topl(env.V, theta, (), self.expansion):
                node["children"][int(i)] = {"N": 0, "Q": 0.0, "node": {}}
            return self._rollout(theta, anchor, f, depth_left, rng)
        i = self._uct(node)
        child = node["children"][i]
        r, term, ntheta, nf, eng = env.step(theta, f, i, rng, anchor)
        if term:
            G = r
        else:
            G = r + env.cfg.gamma * self._sim(child["node"], ntheta, anchor, nf, depth_left - 1, rng)
        child["N"] += 1
        child["Q"] += (G - child["Q"]) / child["N"]
        return G

    def _uct(self, node):
        log_n = np.log(max(node["N"], 2))
        return max(node["children"], key=lambda a: node["children"][a]["Q"] + self.c_uct * np.sqrt(log_n / max(node["children"][a]["N"], 1)))

    def act(self, state, f, t, rng):
        root = {"children": {}, "N": 0}
        depth_cap = min(self.depth, self.env.cfg.H - t)
        theta = state["theta"]
        anchor = state["anchor"]
        for _ in range(self.n_sims):
            self._sim(root, theta, anchor, f, depth_cap, rng)
        return max(root["children"], key=lambda a: root["children"][a]["Q"])

    def update(self, state, item, engaged):
        return state


class OracleHistoryPolicy(BeliefMCTSPolicy):
    """Particles + true kernels (exact filter) + belief MCTS (upper bound for B4 class)."""
    is_oracle = False  # uses only history-consistent information; belief is exact

    def __init__(self, env: H1Env, cfg: BeliefMCTSConfig, n_particles: int = 400):
        super().__init__(env, lambda th, rng: env.initial_particles(th, rng, n_particles), cfg)
