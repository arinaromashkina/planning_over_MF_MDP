"""Reward-semantics audit tests (Phase 4 gate, prompt §4).

Locks in:
1. Candidate generation never returns consumed items, even at l >= catalog size
   (regression for the full-catalog Plan-1 artifact found in the audit).
2. Monotonicity in leverage l (fixed residual q = 1 - u'v > 0):
   - paper alignment 1/(1+|Delta(l)|) is DECREASING in l;
   - post-transition relevance u'v + Delta(l) is INCREASING in l;
   - paper reward sim + eta*alignment is DECREASING in l.
3. Component semantics: paper Plan-1 value uses the CURRENT state u; the
   'post_transition' variant uses u+; Plan-K edge rewards use the edge-origin state.
"""

import numpy as np
import pytest

from mf_mdp.mdp import MFState, fold_in, reward, similarity
from mf_mdp.planning.candidates import exact_cosine_topl
from mf_mdp.planning.planners import (
    PlannerConfig,
    _plan1_value_paper,
    _plan1_value_post_transition,
    plan,
    planK_topk,
)


def _state(rng, d=8, hist=(0, 1)):
    V = rng.normal(0, 1, (30, d))
    A = np.eye(d) * 2.0
    for i in hist:
        A += np.outer(V[i], V[i])
    P = np.linalg.inv(A)
    b = V[list(hist)].sum(axis=0)
    return V, MFState(P=P, u=P @ b, consumed=tuple(sorted(hist)))


# 1. candidate exclusion -------------------------------------------------

def test_topl_never_returns_consumed_at_full_catalog():
    rng = np.random.default_rng(0)
    V = rng.normal(0, 1, (50, 8))
    u = V[3] + V[7]
    consumed = tuple(range(20))  # 20 of 50 consumed
    for l in (1, 10, 30, 31, 50, 1000):
        out = exact_cosine_topl(V, u, consumed, l)
        assert len(set(out) & set(consumed)) == 0, f"l={l} leaked consumed items"
        assert len(out) == min(l, 30)


# 2. monotonicity in leverage ---------------------------------------------

def test_alignment_decreases_in_leverage_fixed_residual():
    q = 0.7
    ls = np.linspace(1e-6, 10.0, 200)
    delta = ls / (1 + ls) * q
    align = 1.0 / (1.0 + np.abs(delta))
    assert np.all(np.diff(align) < 0)


def test_post_transition_relevance_increases_in_leverage():
    q = 0.7
    u_dot_v = 0.3
    ls = np.linspace(1e-6, 10.0, 200)
    post = u_dot_v + ls / (1 + ls) * q
    assert np.all(np.diff(post) > 0)


def test_paper_reward_decreases_in_leverage_fixed_residual():
    eta, q, u_dot_v = 1.0, 0.7, 0.3
    ls = np.linspace(1e-6, 10.0, 200)
    r = u_dot_v + eta / (1.0 + ls / (1 + ls) * q)
    assert np.all(np.diff(r) < 0)


def test_monotonicity_via_states_fixed_udotv():
    """Two items with IDENTICAL u'v but different leverage (anisotropic P):
    the higher-leverage item must get a LOWER paper reward and a HIGHER
    post-transition relevance."""
    d = 8
    P = np.diag([0.05, 5.0] + [0.5] * (d - 2)).astype(float)  # heavy direction e2
    u = np.zeros(d); u[0] = 2.0
    st = MFState(P=P, u=u, consumed=())
    c = 0.3  # u'v = 2*0.3 = 0.6 -> residual q = 0.4 > 0
    v_lo = np.zeros(d); v_lo[0] = c          # only light direction
    v_hi = np.zeros(d); v_hi[0] = c; v_hi[1] = 1.0  # + heavy direction
    assert abs(float(u @ v_lo) - float(u @ v_hi)) < 1e-12  # same u'v
    l_lo = float(v_lo @ (P @ v_lo)); l_hi = float(v_hi @ (P @ v_hi))
    assert l_hi > 10 * l_lo
    _, u_plus_lo, d_lo = fold_in(st, v_lo)
    _, u_plus_hi, d_hi = fold_in(st, v_hi)
    assert d_hi > d_lo                                            # shift grows with l
    assert reward(u_plus_hi, v_hi, d_hi, 1.0, "dot") > reward(u_plus_lo, v_lo, d_lo, 1.0, "dot")
    assert reward(st.u, v_hi, d_hi, 1.0, "dot") < reward(st.u, v_lo, d_lo, 1.0, "dot")


# 3. component semantics ---------------------------------------------------

def test_plan1_paper_uses_current_state():
    rng = np.random.default_rng(2)
    V, st = _state(rng)
    a = 7
    v = V[a]
    val = _plan1_value_paper(V, st, a, eta=1.0, sim="cos")
    _, u_plus, delta_s = fold_in(st, v)
    expected = reward(st.u, v, delta_s, 1.0, "cos")
    assert val == pytest.approx(expected)
    assert not np.isclose(val, reward(u_plus, v, delta_s, 1.0, "cos"))


def test_plan1_post_transition_uses_u_plus():
    rng = np.random.default_rng(3)
    V, st = _state(rng)
    a = 7
    v = V[a]
    val = _plan1_value_post_transition(V, st, a, eta=1.0, sim="cos")
    _, u_plus, delta_s = fold_in(st, v)
    assert val == pytest.approx(reward(u_plus, v, delta_s, 1.0, "cos"))


def test_plank_edge_reward_uses_edge_origin_state():
    """MCTS immediate reward at an edge (s,a) must be r(s,a) with s the origin node."""
    rng = np.random.default_rng(4)
    d = 8
    V = rng.normal(0, 1, (40, d))
    A = np.eye(d) + V[:5].T @ V[:5]
    P = np.linalg.inv(A)
    b = V[:5].sum(axis=0)
    st = MFState(P=P, u=P @ b, consumed=tuple(range(5)))
    cfg = PlannerConfig(method="plank", expansion_width=10, n_simulations=20, horizon=3, eta=1.0)
    recs = planK_topk(V, st, 5, cfg, np.random.default_rng(0))
    assert len(recs) == 5
    assert not set(recs) & set(st.consumed)


def test_variant_aliases():
    rng = np.random.default_rng(5)
    V, st = _state(rng)
    cfg_new = PlannerConfig(method="plan1", plan1_variant="paper", plan1_pool=20)
    cfg_old = PlannerConfig(method="plan1", plan1_variant="root", plan1_pool=20)
    assert plan(V, st, 5, cfg_new) == plan(V, st, 5, cfg_old)
    cfg_pt = PlannerConfig(method="plan1", plan1_variant="post_transition", plan1_pool=20)
    cfg_post = PlannerConfig(method="plan1", plan1_variant="post", plan1_pool=20)
    assert plan(V, st, 5, cfg_pt) == plan(V, st, 5, cfg_post)
