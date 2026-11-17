"""Tests for the H2 hybrid sequential state (GRU gradients, world model, planner)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mf_mdp.mdp import fold_in_general, MFState
from mf_mdp.planning.hybrid import HybridPlanner, HybridPlannerConfig, HybridState
from mf_mdp.sequential.gru import GRU, GRUConfig
from mf_mdp.sequential.world import SeqWorldModel, SeqWorldConfig


def _toy_planner(arm="hybrid", transition="stochastic", seed=0):
    rng = np.random.default_rng(seed)
    n_items, d = 6, 4
    V = rng.normal(0, 1, (n_items, d))
    V /= np.linalg.norm(V, axis=1, keepdims=True)

    class _IALS:
        alpha = 10.0

    from mf_mdp.models.features import FeatureEncoder
    from mf_mdp.models.response import LogisticResponseModel

    world = SeqWorldModel(n_items, SeqWorldConfig(emb_dim=8, hid_dim=8, n_epochs=2, seed=seed))
    enc = FeatureEncoder()
    enc.fit(np.zeros((2, 10)))
    resp = LogisticResponseModel(n_steps=2)
    resp.fit(np.zeros((2, 10)), np.array([0.0, 1.0]))
    q_hat = np.full(n_items, 0.5)
    cfg = HybridPlannerConfig(state_arm=arm, transition=transition)
    pl = HybridPlanner(V, _IALS(), world if arm != "mf" else None,
                       resp, None, enc, np.zeros(n_items), q_hat, 0.3, cfg)
    return pl


# ---------------- GRU ----------------

def test_gru_gradients_match_finite_differences():
    rng = np.random.default_rng(1)
    cfg = GRUConfig(n_in=3, n_hidden=5, seed=1)
    gru = GRU(cfg, rng)
    X = rng.normal(0, 1, (4, 3))
    h0 = rng.normal(0, 1, 5)
    w2 = rng.normal(0, 1, 5)

    def loss():
        cache = gru.forward(X, h0)
        return float(sum(h["h2"] @ w2 for h in cache))

    cache = gru.forward(X, h0)
    dH = np.tile(w2, (4, 1))
    grads = gru.backward(cache, dH)
    eps = 1e-6
    for name, mat in gru.params.items():
        num = np.zeros_like(mat)
        flat = mat.ravel()
        for i in range(len(flat)):
            orig = flat[i]
            flat[i] = orig + eps
            lp = loss()
            flat[i] = orig - eps
            lm = loss()
            flat[i] = orig
            num.ravel()[i] = (lp - lm) / (2 * eps)
        assert np.max(np.abs(num - grads[name])) < 1e-5, name


# ---------------- world model ----------------

def test_world_model_probs_and_determinism():
    world = SeqWorldModel(10, SeqWorldConfig(emb_dim=8, hid_dim=8, n_epochs=1, seed=2))
    h = world.state_of([(1, 1), (2, 0), (3, 1)])
    h2 = world.state_of([(1, 1), (2, 0), (3, 1)])
    assert np.allclose(h, h2)
    p = world.response_prob(h, 4)
    assert 0.0 < p < 1.0
    assert world.score(h, 4) == float(world.E[4] @ world.readout(h))


def test_step_state_is_feedback_conditioned():
    world = SeqWorldModel(10, SeqWorldConfig(emb_dim=8, hid_dim=8, n_epochs=1, seed=3))
    h0 = np.zeros(8)
    h_pos = world.step_state(h0, 5, 1.0)
    h_neg = world.step_state(h0, 5, 0.0)
    assert not np.allclose(h_pos, h_neg)  # g(h, a, y) depends on y


def test_world_model_training_reduces_loss():
    rng = np.random.default_rng(4)
    trajs = []
    for _ in range(30):
        traj = []
        prev = 0
        for _t in range(5):
            a = int(rng.integers(12))
            y = 1.0 if (a + prev) % 2 == 0 else 0.0
            traj.append((a, int(y), 0))
            prev = a
        trajs.append(traj)
    world = SeqWorldModel(12, SeqWorldConfig(emb_dim=8, hid_dim=8, n_epochs=40, seed=4))
    world.fit(trajs)
    assert world.loss_history[-1] < world.loss_history[0]


def test_control_modes_corrupt_inputs_only():
    rng = np.random.default_rng(5)
    world = SeqWorldModel(20, SeqWorldConfig(emb_dim=8, hid_dim=8, n_epochs=1, seed=5))
    events = [(3, 1), (5, 1), (7, 1), (9, 0)]
    clean = world.encode_events(events, "clean", rng)
    assert clean[1]["x"][-2] == 1.0  # y_prev enters the encoder input
    no_resp = world.encode_events(events, "no_response", rng)
    assert all(st["x"][-2] == 0.0 for st in no_resp)
    assert [st["a"] for st in no_resp] == [a for a, _ in events]  # head keeps true action
    shuffled = world.encode_events(events, "shuffle", rng)
    assert sorted(st["a"] for st in shuffled) == sorted(a for a, _ in events)
    rand_act = world.encode_events(events, "random_action", rng)
    assert all(0 <= st["a_in"] < 20 for st in rand_act)
    assert [st["a"] for st in rand_act] == [a for a, _ in events]


# ---------------- planner ----------------

def test_planner_mf_transition_matches_fold_in():
    pl = _toy_planner(arm="mf")
    rng = np.random.default_rng(0)
    V = pl.V
    st = HybridState(P=np.eye(4), u=rng.normal(0, 1, 4), h=None, consumed=(0,))
    st2 = pl._transition(st, 2, True)
    P_exp, u_exp, _ = fold_in_general(MFState(P=st.P, u=st.u, consumed=st.consumed),
                                      V[2], pl.ials_cfg.alpha, 1.0 + pl.ials_cfg.alpha)
    assert np.allclose(st2.P, P_exp) and np.allclose(st2.u, u_exp)
    assert st2.consumed == (0, 2)


def test_optimistic_arm_ignores_response_model():
    pl = _toy_planner(arm="hybrid", transition="optimistic")
    st = HybridState(P=np.eye(4), u=np.ones(4), h=np.zeros(8), consumed=())

    def _boom(*_a, **_k):
        raise AssertionError("optimistic planner must not call the response model")

    pl.p_resp = _boom  # type: ignore[assignment]
    a = pl.plan(st, 0)
    assert isinstance(a, int)


def test_plan_is_deterministic():
    pl1 = _toy_planner(seed=7)
    pl2 = _toy_planner(seed=7)
    st = HybridState(P=np.eye(4), u=np.ones(4), h=np.zeros(8), consumed=(1,))
    rng = np.random.default_rng(0)
    assert pl1.plan(st, 0, rng) == pl2.plan(st, 0, rng)


def test_hybrid_user_vec_concatenates_geometries():
    pl = _toy_planner(arm="hybrid")
    st = HybridState(P=np.eye(4), u=np.ones(4), h=np.zeros(8), consumed=())
    uv = pl.user_vec(st)
    assert uv.shape == (4 + 8,)
    e = pl._emb(0)
    assert e.shape == (4 + 8,)
    assert -1.0 - 1e-9 <= pl.relevance(st, 0) <= 1.0 + 1e-9
