"""MF-MDP state, closed-form fold-in transition, and reward.

Ground truth: arXiv:2607.02115v2, eqs. (2)-(8), Appendix A.

State s = (P, u) with P = A^{-1}, A = lam*I + sum_i c_i v_i v_i^T (all catalog items,
unobserved contribute baseline confidence c=1, p=0), b = sum_{i in H} c_i v_i, u = A^{-1} b.

Optimistic transition (unit-confidence positive fold-in), v = v_a:
    z = P v,  l = v' P v,  denom = 1 + l
    P+  = P - z z' / denom
    u+  = u + z (1 - u' v) / denom
    Delta_s(a) = (u+)' v - u' v = l / denom * (1 - u' v)

Reward, eq. (8):  r(s, a) = sim(u, v_a) + eta * 1 / (1 + |Delta_s(a)|),
trajectory reward R(tau) = sum_k gamma^k r(s_k, a_k).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class MFState:
    """MF-MDP state s = (P, u) plus the consumed-item set for candidate exclusion."""

    P: np.ndarray  # (d, d) inverse precision A^{-1}
    u: np.ndarray  # (d,) user embedding
    consumed: tuple[int, ...]  # item indices already in history (sorted)

    @property
    def dim(self) -> int:
        return int(self.u.shape[0])


def fold_in(state: MFState, v: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Closed-form optimistic fold-in (Appendix A, eqs. (15)-(17)).

    Args:
        state: current MFState.
        v: embedding (d,) of the action item.

    Returns:
        (P_plus, u_plus, delta_s): updated inverse precision, updated user vector,
        and the self-score shift Delta_s(a) of the folded-in item.
    """
    z = state.P @ v
    l = float(v @ z)
    denom = 1.0 + l
    u_dot_v = float(state.u @ v)
    delta_s = (l / denom) * (1.0 - u_dot_v)
    P_plus = state.P - np.outer(z, z) / denom
    u_plus = state.u + z * (1.0 - u_dot_v) / denom
    return P_plus, u_plus, delta_s


def transitioned(state: MFState, v: np.ndarray, item: int) -> tuple[MFState, float]:
    """fold_in + build the successor state, appending `item` to the consumed set."""
    P_plus, u_plus, delta_s = fold_in(state, v)
    return MFState(P=P_plus, u=u_plus, consumed=tuple(sorted(state.consumed + (item,)))), delta_s


def similarity(u: np.ndarray, v: np.ndarray, kind: str = "cos") -> np.ndarray | float:
    """Similarity between user vector(s) u and item vector(s) v.

    kind="dot": u'v. kind="cos": u'v / (||u|| ||v||), with 0 for zero vectors.
    """
    if kind == "dot":
        return u @ v
    if kind == "cos":
        u_norm = np.linalg.norm(u)
        v_norm = np.linalg.norm(v)
        if u_norm == 0.0 or v_norm == 0.0:
            return 0.0
        return float(u @ v) / (u_norm * v_norm)
    raise ValueError(f"unknown similarity kind: {kind}")


def reward(u: np.ndarray, v: np.ndarray, delta_s: float, eta: float, kind: str = "cos") -> float:
    """Reward r(s, a) = sim(u, v_a) + eta / (1 + |Delta_s(a)|), eq. (8)."""
    return float(similarity(u, v, kind) + eta / (1.0 + abs(delta_s)))


def reward_bounds(eta: float) -> tuple[float, float]:
    """Affine bounds of r for cosine similarity: sim in [-1, 1], alignment in (0, eta]."""
    return -1.0, 1.0 + eta


def normalize_reward(r: float, eta: float) -> float:
    """Map r from [r_lo, r_hi] to [0, 1] (UCT assumes samples in [0, 1], Appendix B)."""
    r_lo, r_hi = reward_bounds(eta)
    return (r - r_lo) / (r_hi - r_lo)
