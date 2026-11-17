"""Pool-restricted Plan-K MCTS (H3): the search tree stays inside a root candidate
pool C_t provided by an external proposer (sequential model / MF ANN / union / random
expansion). Mirrors mf_mdp.planning.planners.planK_topk (same UCT eq. (10), same
closed-form leaf value, same discounted return-to-go backprop, same readout) with the
expansion source swapped from the catalog-wide exact top-l to `C_t \\ consumed(path)`.
Existing planning modules stay untouched (docs/H3_LEARNED_ACTION_PROPOSALS_DESIGN.md §4).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mf_mdp.mdp import MFState, fold_in, normalize_reward, reward, similarity, transitioned
from mf_mdp.planning.planners import PlannerConfig, _plan1_value_paper

# NOTE: _plan1_value_paper is imported from the canonical module (it is the paper-exact
# one-step value, stateless) to avoid drift; the MCTS mechanics below are duplicated
# here only to inject the pool into expansion/leaf/readout-tail.


@dataclass
class _Node:
    P: np.ndarray
    u: np.ndarray
    consumed: tuple[int, ...]
    depth: int
    children: dict
    edge_r: dict
    edge_n: dict
    edge_q: dict
    N: int

    def __init__(self, state: MFState, depth: int):
        self.P = state.P
        self.u = state.u
        self.consumed = state.consumed
        self.depth = depth
        self.children = {}
        self.edge_r = {}
        self.edge_n = {}
        self.edge_q = {}
        self.N = 0

    def state(self) -> MFState:
        return MFState(P=self.P, u=self.u, consumed=self.consumed)

    def uct(self, c_uct: float, V: np.ndarray) -> int:
        log_n = np.log(max(self.N, 2))
        best_a, best_v = -1, -np.inf
        for a, child in self.children.items():
            cos = similarity(self.u, V[a], "cos")
            bonus = c_uct * np.sqrt(log_n * (2.0 + cos) / (self.edge_n[a] + 1))
            v = self.edge_q[a] + bonus
            if v > best_v:
                best_v, best_a = v, a
        return best_a


def _pool_at_node(pool: np.ndarray, node: _Node) -> np.ndarray:
    """C_t minus the node's consumed set (the expansion SOURCE in this module)."""
    consumed = set(node.consumed)
    return pool[[i for i, a in enumerate(pool) if int(a) not in consumed]]


def _topl_within_pool(
    V: np.ndarray, pool: np.ndarray, node: _Node, l: int
) -> np.ndarray:
    """The paper's ANN step (exact top-l by cosine to the CURRENT u) restricted to C_t.

    Branching factor matches the unrestricted planner (expansion_width l), so the
    compute comparison pool-vs-full is fair: identical per-sim fold-in work, only the
    candidate-scoring domain shrinks from the catalog to C_t.
    """
    cand = _pool_at_node(pool, node)
    if len(cand) <= l:
        return cand
    items = V[cand]
    norms = np.linalg.norm(items, axis=1)
    norms = np.where(norms == 0.0, 1.0, norms)
    u_norm = np.linalg.norm(node.u)
    scores = items @ node.u / (norms * u_norm) if u_norm > 0.0 else np.zeros(len(cand))
    part = np.argpartition(-scores, l - 1)[:l]
    return cand[part[np.argsort(-scores[part], kind="stable")]]


def _greedy_rho_pool(V: np.ndarray, node: _Node, cfg: PlannerConfig, pool: np.ndarray) -> float:
    """Closed-form leaf value inside C_t: max normalized reward over the node's
    top-expansion_width pool candidates by cosine (same restriction as expansion)."""
    cand = _topl_within_pool(V, pool, node, cfg.expansion_width)
    best = 0.0
    for b in cand:
        v = V[b]
        _, _, d_b = fold_in(node.state(), v)
        r_b = reward(node.u, v, d_b, cfg.eta, cfg.sim)
        best = max(best, normalize_reward(r_b, cfg.eta))
    return best


def _expand_pool(V: np.ndarray, node: _Node, cfg: PlannerConfig, pool: np.ndarray) -> None:
    cand = _topl_within_pool(V, pool, node, cfg.expansion_width)
    for a in cand:
        v = V[a]
        P_plus, u_plus, delta_s = fold_in(node.state(), v)
        r_norm = normalize_reward(reward(node.u, v, delta_s, cfg.eta, cfg.sim), cfg.eta)
        child = _Node(
            MFState(P=P_plus, u=u_plus, consumed=tuple(sorted(node.consumed + (int(a),)))),
            node.depth + 1,
        )
        child.N = 1
        node.children[int(a)] = child
        node.edge_r[int(a)] = r_norm
        node.edge_n[int(a)] = 1
        node.edge_q[int(a)] = r_norm + cfg.gamma * _greedy_rho_pool(V, child, cfg, pool)


def planK_topk_pool(
    V: np.ndarray,
    state: MFState,
    k: int,
    cfg: PlannerConfig,
    rng: np.random.Generator,
    pool: np.ndarray,
) -> list[int]:
    """Plan-K MCTS restricted to the root pool C_t (array of item ids, unconsumed at root).

    Selection/expansion/leaf/readout all stay inside C_t \\ consumed(path).
    """
    pool = np.unique(np.asarray(pool, dtype=np.int64))  # dedup (order irrelevant: expansion picks top-l by cosine)
    if pool.size == 0:  # degenerate: fall back to unrestricted catalog search
        from mf_mdp.planning.planners import planK_topk

        return planK_topk(V, state, k, cfg, rng)
    root = _Node(state, 0)
    for _ in range(cfg.n_simulations):
        node = root
        path: list[tuple[_Node, int, _Node]] = []
        while node.depth < cfg.horizon:
            if not node.children:
                _expand_pool(V, node, cfg, pool)
                node.N += 1
                break
            a = node.uct(cfg.c_uct, V)
            child = node.children[a]
            path.append((node, a, child))
            node.N += 1
            node = child
        G = _greedy_rho_pool(V, node, cfg, pool)
        for parent, a, child in reversed(path):
            G = parent.edge_r[a] + cfg.gamma * G
            n = parent.edge_n[a]
            parent.edge_q[a] += (G - parent.edge_q[a]) / (n + 1)
            parent.edge_n[a] = n + 1
    # Readout: descend by best backed-up Q, pad the tail with Plan-1 steps inside C_t.
    recs: list[int] = []
    node = root
    while node.children and len(recs) < k:
        a = max(node.children, key=lambda x: (node.edge_q[x], node.edge_n[x]))
        recs.append(a)
        node = node.children[a]
    leaf_state = node.state()
    while len(recs) < k:
        cand = _topl_within_pool(V, pool, _Node(leaf_state, 0), cfg.expansion_width)
        if len(cand) == 0:
            break
        values = [_plan1_value_paper(V, leaf_state, int(a), cfg.eta, cfg.sim) for a in cand]
        a = int(cand[int(np.argmax(values))])
        recs.append(a)
        leaf_state, _ = transitioned(leaf_state, V[a], a)
    return recs[:k]


def trajectory_return(
    V: np.ndarray,
    state: MFState,
    items: list[int],
    eta: float,
    sim: str = "cos",
    gamma: float = 0.9,
) -> float:
    """J(pi) of a recommended slate rolled out in the MF-MDP model:
    R(tau) = sum_k gamma^k r(s_k, a_k), eq. (8) RAW (unnormalized) rewards."""
    total, cur, disc = 0.0, state, 1.0
    for a in items:
        v = V[a]
        P_plus, u_plus, delta_s = fold_in(cur, v)
        total += disc * reward(cur.u, v, delta_s, eta, sim)
        cur = MFState(P=P_plus, u=u_plus, consumed=tuple(sorted(cur.consumed + (int(a),))))
        disc *= gamma
    return total
