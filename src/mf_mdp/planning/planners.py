"""Planners over the MF-MDP: Static, Plan-1, and Plan-K (MCTS).

Paper ground truth (arXiv:2607.02115v2):
- Static: top-K by score at the root state.
- Plan-1: "evaluates each candidate after a single fold-in and re-ranks by the one-step
  relevance/alignment value" (Sec. 3.1). Our primary reading: value(a) = r(s+, a) with
  s+ the folded-in state. Alternatives selectable via `plan1_variant` (documented in
  docs/REPRODUCTION.md).
- Plan-K: MCTS with (i) expansion restricted to the l catalog items nearest the current
  user vector, (ii) cosine-prior UCT eq. (10): Q + c_uct * sqrt(logN(s)*(2+cos)/(N(s,a)+1)),
  (iii) closed-form leaf value, (iv) discounted return-to-go backprop G <- r + gamma*G,
  (v) readout of the best trajectory by backed-up Q (Fig. 1b).

Implementation assumptions (paper silent, see docs/DECISIONS.md):
- Closed-form leaf value rho(s) = max over the top-l unconsumed candidates of r(s, a).
- Rewards are affinely normalized to [0, 1] inside MCTS (Appendix B assumes samples
  in [0, 1]).
- If the best trajectory is shorter than K (tree too shallow), the tail is padded by
  repeatedly applying the Plan-1 step at the leaf state.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mf_mdp.mdp import MFState, fold_in, normalize_reward, reward, similarity, transitioned
from mf_mdp.planning.candidates import catalog_scores, exact_cosine_topl


@dataclass(frozen=True)
class PlannerConfig:
    method: str = "plan1"  # static_dot | static_cos | plan1 | plank
    sim: str = "cos"  # similarity used by planners (paper: cosine)
    eta: float = 1.0
    gamma: float = 0.9
    c_uct: float = 1.0
    expansion_width: int = 20  # l
    n_simulations: int = 100
    horizon: int = 10  # K
    plan1_variant: str = "paper"  # paper (exact eq.8 at current state) | post_transition | bellman
    plan1_pool: int = 100  # candidate pool size for Plan-1 re-ranking
    # Phase 4 extensions (all opt-in; defaults reproduce pre-Phase-4 behavior):
    reward_fn: object | None = None  # (state, item) -> float; Plan-1 value override (R1/R2)
    reward_bounds_override: tuple[float, float] | None = None  # for MCTS normalization w/ custom rewards
    leaf_evaluator: str = "original"  # original | zero | heuristic | v_logging_ridge
    value_model: object | None = None  # VLoggingRidge for leaf_evaluator=v_logging_ridge


def static_topk(V: np.ndarray, state: MFState, k: int, kind: str) -> list[int]:
    """Top-K unconsumed items by `kind` similarity at the root state."""
    scores = catalog_scores(V, state.u, state.consumed, kind)
    k = min(k, V.shape[0])
    part = np.argpartition(-scores, k - 1)[:k]
    return list(part[np.argsort(-scores[part], kind="stable")])


def _plan1_value_paper(V, state, a, eta, sim):
    """Paper-exact one-step value: r(s, a) = sim(u, v_a) + eta/(1+|Delta_s(a)|), eq. (8).

    Relevance is evaluated at the CURRENT state u (paper text); Delta_s comes from the
    hypothetical fold-in of a.
    """
    v = V[a]
    _, _, delta_s = fold_in(state, v)
    return reward(state.u, v, delta_s, eta, sim)


def _plan1_value_post_transition(V, state, a, eta, sim):
    """VARIANT (not the paper reward): post-transition relevance, value = r(s+, a)."""
    v = V[a]
    _, u_plus, delta_s = fold_in(state, v)
    return reward(u_plus, v, delta_s, eta, sim)


def _plan1_value_bellman(V, state, a, eta, gamma, sim, pool):
    """value(a) = r(s0,a) + gamma * max_b r(s1, b): one-step lookahead with greedy leaf."""
    v = V[a]
    P_plus, u_plus, delta_s = fold_in(state, v)
    r0 = reward(state.u, v, delta_s, eta, sim)
    consumed1 = tuple(sorted(state.consumed + (a,)))
    cand = exact_cosine_topl(V, u_plus, consumed1, pool)
    best = 0.0
    for b in cand:
        vb = V[b]
        _, _, d_b = fold_in(MFState(P=P_plus, u=u_plus, consumed=consumed1), vb)
        best = max(best, reward(u_plus, vb, d_b, eta, sim))
    return r0 + gamma * best


def plan1_topk(V: np.ndarray, state: MFState, k: int, cfg: PlannerConfig) -> list[int]:
    """Plan-1: re-rank ANN-pool candidates by the one-step dynamics-aware value."""
    pool = exact_cosine_topl(V, state.u, state.consumed, cfg.plan1_pool)
    values = np.empty(len(pool))
    for i, a in enumerate(pool):
        if cfg.reward_fn is not None:
            values[i] = cfg.reward_fn(state, a)
            continue
        variant = {"root": "paper", "post": "post_transition"}.get(cfg.plan1_variant, cfg.plan1_variant)
        if variant == "paper":
            values[i] = _plan1_value_paper(V, state, a, cfg.eta, cfg.sim)
        elif variant == "post_transition":
            values[i] = _plan1_value_post_transition(V, state, a, cfg.eta, cfg.sim)
        elif variant == "bellman":
            values[i] = _plan1_value_bellman(
                V, state, a, cfg.eta, cfg.gamma, cfg.sim, cfg.expansion_width
            )
        else:
            raise ValueError(f"unknown plan1_variant: {cfg.plan1_variant}")
    k = min(k, len(pool))
    part = np.argpartition(-values, k - 1)[:k]
    top = pool[part[np.argsort(-values[part], kind="stable")]]
    recs = list(int(a) for a in top)
    if len(recs) < k:  # pool smaller than k: fall back to static tail
        tail = [a for a in static_topk(V, state, k, cfg.sim) if a not in recs]
        recs.extend(tail[: k - len(recs)])
    return recs


class _Node:
    __slots__ = ("P", "u", "consumed", "depth", "children", "edge_r", "edge_n", "edge_q", "N")

    def __init__(self, state: MFState, depth: int):
        self.P = state.P
        self.u = state.u
        self.consumed = state.consumed
        self.depth = depth
        self.children: dict[int, _Node] = {}
        self.edge_r: dict[int, float] = {}  # normalized r(s, a) per child
        self.edge_n: dict[int, int] = {}  # N(s, a)
        self.edge_q: dict[int, float] = {}  # Q(s, a), mean backed-up return
        self.N = 0  # N(s)

    def state(self) -> MFState:
        return MFState(P=self.P, u=self.u, consumed=self.consumed)

    def uct(self, c_uct: float) -> int:
        log_n = np.log(max(self.N, 2))
        best_a, best_v = -1, -np.inf
        for a, child in self.children.items():
            cos = similarity(self.u, _V[a], "cos")
            bonus = c_uct * np.sqrt(log_n * (2.0 + cos) / (self.edge_n[a] + 1))
            v = self.edge_q[a] + bonus
            if v > best_v:
                best_v, best_a = v, a
        return best_a


_V: np.ndarray | None = None  # module-level cache for UCT cosine prior


def _norm_reward(r: float, cfg: PlannerConfig) -> float:
    if cfg.reward_bounds_override is not None:
        lo, hi = cfg.reward_bounds_override
        return (r - lo) / (hi - lo)
    return normalize_reward(r, cfg.eta)


def _greedy_rho(V: np.ndarray, node: _Node, cfg: PlannerConfig) -> float:
    """Closed-form leaf value: max normalized reward over the node's top-l candidates."""
    cand = exact_cosine_topl(V, node.u, node.consumed, cfg.expansion_width)
    best = 0.0
    for b in cand:
        v = V[b]
        _, _, d_b = fold_in(node.state(), v)
        r_b = reward(node.u, v, d_b, cfg.eta, cfg.sim)
        best = max(best, _norm_reward(r_b, cfg))
    return best


def _horizon_extra(node: _Node, cfg: PlannerConfig, path_rs: list[float]) -> np.ndarray:
    """Horizon/recency features for a V_logging model with horizon_features=True.

    Mapping caveat (documented): remaining = horizon - depth (planning horizon, NOT the
    user's real remaining session); mean/last reward from the current search path.
    """
    rem = max(0, cfg.horizon - node.depth)
    mean_r = float(np.mean(path_rs)) if path_rs else 0.0
    last_r = float(path_rs[-1]) if path_rs else 0.0
    return np.array([np.log1p(rem), mean_r, last_r])


def _leaf_value(V: np.ndarray, node: _Node, cfg: PlannerConfig, path_rs: list[float] | None = None) -> float:
    """Frontier leaf value G at a node, by configured leaf_evaluator.

    v_logging_ridge returns the LOGGING-POLICY value V_logging(s_h) in normalized
    reward units — it is NOT target-policy value (docs/PHASE4_DESIGN.md §4.2).
    """
    path_rs = path_rs or []
    if cfg.leaf_evaluator == "v_logging_ridge":
        if cfg.value_model is None:
            raise ValueError("leaf_evaluator=v_logging_ridge requires value_model")
        extra = _horizon_extra(node, cfg, path_rs) if getattr(cfg.value_model, "horizon_features", False) else None
        return float(np.clip(cfg.value_model.predict_value(node.state(), extra=extra), 0.0, 1.0 / (1.0 - cfg.gamma)))
    if cfg.leaf_evaluator == "zero":
        return 0.0
    if cfg.leaf_evaluator in ("original", "heuristic"):
        return _greedy_rho(V, node, cfg)
    raise ValueError(f"unknown leaf_evaluator: {cfg.leaf_evaluator}")


def _expand(V: np.ndarray, node: _Node, cfg: PlannerConfig, path_rs: list[float] | None = None) -> None:
    path_rs = path_rs or []
    cand = exact_cosine_topl(V, node.u, node.consumed, cfg.expansion_width)
    for a in cand:
        v = V[a]
        P_plus, u_plus, delta_s = fold_in(node.state(), v)
        r_norm = _norm_reward(reward(node.u, v, delta_s, cfg.eta, cfg.sim), cfg)
        child = _Node(MFState(P=P_plus, u=u_plus, consumed=tuple(sorted(node.consumed + (a,)))), node.depth + 1)
        child.N = 1
        node.children[a] = child
        node.edge_r[a] = r_norm
        node.edge_n[a] = 1
        node.edge_q[a] = r_norm + cfg.gamma * _leaf_value(V, child, cfg, path_rs + [r_norm])


def planK_topk(
    V: np.ndarray, state: MFState, k: int, cfg: PlannerConfig, rng: np.random.Generator
) -> list[int]:
    """Plan-K: MCTS over the MF-MDP, readout of the best trajectory by backed-up Q."""
    global _V
    _V = V
    root = _Node(state, 0)
    for _ in range(cfg.n_simulations):
        node = root
        path: list[tuple[_Node, int, _Node]] = []
        path_rs: list[float] = []  # normalized rewards along the current path
        # 1-2. Selection + expansion.
        while node.depth < cfg.horizon:
            if not node.children:
                _expand(V, node, cfg, path_rs)
                node.N += 1
                break
            a = node.uct(cfg.c_uct)
            child = node.children[a]
            path.append((node, a, child))
            path_rs.append(node.edge_r[a])
            node.N += 1
            node = child
        # 3. Leaf value at the frontier node.
        G = _leaf_value(V, node, cfg, path_rs)
        # 4. Backprop discounted return-to-go.
        for parent, a, child in reversed(path):
            G = parent.edge_r[a] + cfg.gamma * G
            n = parent.edge_n[a]
            parent.edge_q[a] += (G - parent.edge_q[a]) / (n + 1)
            parent.edge_n[a] = n + 1
    # Readout: descend by best backed-up Q, then pad the tail with Plan-1 steps.
    recs: list[int] = []
    node = root
    while node.children and len(recs) < k:
        a = max(node.children, key=lambda x: (node.edge_q[x], node.edge_n[x]))
        recs.append(a)
        node = node.children[a]
    leaf_state = node.state()
    while len(recs) < k:
        pool = exact_cosine_topl(V, leaf_state.u, leaf_state.consumed, cfg.expansion_width)
        if len(pool) == 0:
            break
        values = [_plan1_value_paper(V, leaf_state, a, cfg.eta, cfg.sim) for a in pool]
        a = int(pool[int(np.argmax(values))])
        recs.append(a)
        leaf_state, _ = transitioned(leaf_state, V[a], a)
    return recs[:k]


def plan(V: np.ndarray, state: MFState, k: int, cfg: PlannerConfig, rng=None) -> list[int]:
    """Dispatch to the configured planner."""
    if cfg.method == "static_dot":
        return static_topk(V, state, k, "dot")
    if cfg.method == "static_cos":
        return static_topk(V, state, k, "cos")
    if cfg.method == "plan1":
        return plan1_topk(V, state, k, cfg)
    if cfg.method == "plan1_dot":  # Table-2 ablation: same planner, dot similarity
        return plan1_topk(V, state, k, PlannerConfig(**{**cfg.__dict__, "sim": "dot"}))
    if cfg.method == "plank":
        if rng is None:
            rng = np.random.default_rng(0)
        return planK_topk(V, state, k, cfg, rng)
    raise ValueError(f"unknown planner method: {cfg.method}")
