"""Tests for the H4 interaction response models (FM, MLP, crosses, CrossPlanner)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from mf_mdp.interactions.cross import (
    ENVC_CROSS_NAMES,
    KUAI_CROSS_NAMES,
    RandomCrosses,
    envc_crosses,
    kuairand_crosses,
)
from mf_mdp.interactions.fm import FactorizationMachine
from mf_mdp.interactions.mlp import MLPClassifier
from mf_mdp.interactions.planner import CrossPlanner, CrossPlannerConfig
from mf_mdp.mdp import MFState
from mf_mdp.models.features import FeatureEncoder
from mf_mdp.models.response import LogisticResponseModel
from mf_mdp.planning.hybrid import HybridPlanner, HybridPlannerConfig, HybridState


def _brute_pair_term(X: np.ndarray, V: np.ndarray) -> np.ndarray:
    n, d = X.shape
    out = np.zeros(n)
    for row in range(n):
        for i in range(d):
            for j in range(i + 1, d):
                out[row] += X[row, i] * X[row, j] * float(V[i] @ V[j])
    return out


def test_fm_pair_term_matches_brute_force():
    rng = np.random.default_rng(0)
    X = rng.normal(0, 1, (7, 5))
    fm = FactorizationMachine(k=3)
    fm.w = np.zeros(5)
    fm.V = rng.normal(0, 1, (5, 3))
    np.testing.assert_allclose(fm._pair_term(X), _brute_pair_term(X, fm.V), rtol=1e-10)


def _finite_diff_grad(fn, params: np.ndarray, eps=1e-6) -> np.ndarray:
    grad = np.zeros_like(params, dtype=float)
    it = np.nditer(params, flags=["multi_index"])
    while not it.finished:
        idx = it.multi_index
        old = params[idx]
        params[idx] = old + eps
        hi = fn()
        params[idx] = old - eps
        lo = fn()
        params[idx] = old
        grad[idx] = (hi - lo) / (2 * eps)
        it.iternext()
    return grad


def test_fm_gradients_match_finite_differences():
    rng = np.random.default_rng(1)
    X = rng.normal(0, 1, (6, 4))
    y = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    fm = FactorizationMachine(k=2, l2=1e-3)
    fm.w = rng.normal(0, 0.1, 4)
    fm.V = rng.normal(0, 0.2, (4, 2))
    fm.w0 = 0.1

    def loss():
        from mf_mdp.models.response import log_loss
        reg = 0.5 * fm.l2 * (float(fm.w @ fm.w) + float((fm.V ** 2).sum()))
        return log_loss(y, fm.predict_proba(X)) + reg

    _, dw, dV = fm._grads(X, y)
    assert np.allclose(dw, _finite_diff_grad(loss, fm.w), atol=1e-5)
    assert np.allclose(dV, _finite_diff_grad(loss, fm.V), atol=1e-5)
    # w0 gradient
    eps = 1e-6
    fm.w0 += eps
    hi = loss()
    fm.w0 -= 2 * eps
    lo = loss()
    fm.w0 += eps
    dw0, _, _ = fm._grads(X, y)
    assert abs(dw0 - (hi - lo) / (2 * eps)) < 1e-5


def test_mlp_gradients_match_finite_differences():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (6, 4))
    y = np.array([0.0, 1.0, 1.0, 0.0, 1.0, 0.0])
    mlp = MLPClassifier(hidden=3, l2=1e-3)
    mlp.W1 = rng.normal(0, 0.2, (4, 3))
    mlp.b1 = rng.normal(0, 0.1, 3)
    mlp.w2 = rng.normal(0, 0.2, 3)
    mlp.b2 = 0.05

    def loss():
        from mf_mdp.models.response import log_loss
        reg = 0.5 * mlp.l2 * (float((mlp.W1 ** 2).sum()) + float(mlp.w2 @ mlp.w2))
        return log_loss(y, mlp.predict_proba(X)) + reg

    dW1, db1, dw2, _ = mlp._grads(X, y)
    assert np.allclose(dW1, _finite_diff_grad(loss, mlp.W1), atol=1e-5)
    assert np.allclose(db1, _finite_diff_grad(loss, mlp.b1), atol=1e-5)
    assert np.allclose(dw2, _finite_diff_grad(loss, mlp.w2), atol=1e-5)


def test_fm_and_mlp_fit_deterministic():
    rng = np.random.default_rng(3)
    X = rng.normal(0, 1, (200, 6))
    z = X[:, 0] * X[:, 1] - 0.5 * X[:, 2]
    y = (rng.random(200) < 1 / (1 + np.exp(-z))).astype(float)
    fm_a = FactorizationMachine(k=4, n_epochs=5, seed=7)
    fm_a.fit(X, y)
    fm_b = FactorizationMachine(k=4, n_epochs=5, seed=7)
    fm_b.fit(X, y)
    np.testing.assert_array_equal(fm_a.predict_proba(X), fm_b.predict_proba(X))
    mlp_a, mlp_b = MLPClassifier(hidden=8, n_epochs=5, seed=7), MLPClassifier(hidden=8, n_epochs=5, seed=7)
    mlp_a.fit(X, y)
    mlp_b.fit(X, y)
    np.testing.assert_array_equal(mlp_a.predict_proba(X), mlp_b.predict_proba(X))
    # capability smoke: pure multiplicative signal (no linear component) ->
    # LR cannot do better than a constant predictor; FM must beat it clearly.
    from mf_mdp.models.response import log_loss
    lr = LogisticResponseModel(l2=1e-4, n_steps=400)
    lr.fit(X, y)
    fm_c = FactorizationMachine(k=8, n_epochs=200, lr=0.2, seed=7)
    fm_c.fit(X, y)
    assert log_loss(y, fm_c.predict_proba(X)) < log_loss(y, lr.predict_proba(X)) - 0.05


def test_cross_builders_known_values():
    base = np.array([0.5, 3.0, 0.2, 0.1, 1.0, 4.0, 7.0, 2.0, 1.5, 1.1])
    c = envc_crosses(base, mean_eng_hist=0.8, repeat_cnt=1.0)
    assert len(c) == len(ENVC_CROSS_NAMES) == 4
    np.testing.assert_allclose(c, [0.5 * 7.0, 0.8 * 0.5, 0.2 * 4.0, 1.0 * 0.5])
    k = kuairand_crosses(base, mean_wr_hist=0.6, recency_min=0.0, repeat_cnt=2.0,
                         author_repeat_cnt=1.0)
    assert len(k) == len(KUAI_CROSS_NAMES) == 6
    np.testing.assert_allclose(k[0], 0.5 * 7.0)
    np.testing.assert_allclose(k[1], 0.5 * 1.0)  # exp(0)=1 at zero recency
    np.testing.assert_allclose(k[2], 0.6 * 0.5)
    assert not (np.isnan(k).any() or np.isnan(c).any())


def test_random_crosses_frozen_pairs():
    rng = np.random.default_rng(4)
    X = rng.normal(0, 1, (50, 10))
    rc = RandomCrosses(k=4, seed=0)
    Z = rc.fit_transform(X)
    assert Z.shape == (50, 4)
    pairs_a = list(rc.pairs)
    rc2 = RandomCrosses(k=4, seed=0).fit(X)
    assert rc2.pairs == pairs_a
    # frozen at transform: pairs do not depend on the transformed matrix
    X2 = rng.normal(0, 1, (50, 10))
    np.testing.assert_array_equal(rc.transform(X2), rc2.transform(X2))
    assert all(i != j for i, j in rc.pairs)
    assert len(set(rc.pairs)) == 4


def _toy_mf_setup(seed=0):
    rng = np.random.default_rng(seed)
    n_items, d = 6, 4
    V = rng.normal(0, 1, (n_items, d))
    V /= np.linalg.norm(V, axis=1, keepdims=True)

    class _IALS:
        alpha = 10.0

    log_pop = np.log1p(np.arange(n_items, dtype=float))
    q_hat = np.full(n_items, 0.5)
    return V, _IALS(), log_pop, q_hat


def test_cross_planner_base_mode_parity():
    """feature_mode='base' must reproduce HybridPlanner decisions exactly."""
    from mf_mdp.mf import IALSConfig, user_state
    V, ials_cfg, log_pop, q_hat = _toy_mf_setup()
    rng = np.random.default_rng(0)
    hist = [0, 2, 2, 1]
    st = user_state(V, np.asarray(hist, dtype=np.int64), IALSConfig(dim=4, alpha=10.0))
    hstate = HybridState(P=st.P, u=st.u, h=None, consumed=tuple(sorted(hist)))

    X = np.random.default_rng(1).normal(0, 1, (40, 10))
    y = (np.random.default_rng(2).random(40) > 0.5).astype(float)
    enc = FeatureEncoder()
    enc.fit(X)
    resp = LogisticResponseModel(n_steps=5)
    resp.fit(enc.transform(X), y)

    cfg_h = HybridPlannerConfig(state_arm="mf", transition="stochastic")
    pl_h = HybridPlanner(V, ials_cfg, None, resp, None, enc, log_pop, q_hat, 0.3, cfg_h)
    cfg_c = CrossPlannerConfig(state_arm="mf", transition="stochastic", feature_mode="base")
    pl_c = CrossPlanner(V, ials_cfg, None, resp, None, enc, log_pop, q_hat, 0.3,
                        cfg=cfg_c, enc_resp=enc, enc_haz=enc)
    for f in (0, 1, 2):
        a_h = pl_h.plan(hstate, f, rng)
        a_c = pl_c.plan(hstate, f, rng)
        assert a_h == a_c
        for a in range(V.shape[0]):
            assert pl_h.p_resp(hstate, a) == pl_c.p_resp(hstate, a)


def test_cross_planner_mode_feature_shapes():
    from mf_mdp.mf import IALSConfig, user_state
    V, ials_cfg, log_pop, q_hat = _toy_mf_setup()
    hist = [0, 1]
    st = user_state(V, np.asarray(hist, dtype=np.int64), IALSConfig(dim=4, alpha=10.0))
    hstate = HybridState(P=st.P, u=st.u, h=None, consumed=tuple(sorted(hist)))
    X = np.random.default_rng(1).normal(0, 1, (40, 10))
    enc = FeatureEncoder()
    enc.fit(X)
    resp = LogisticResponseModel(n_steps=5)
    resp.fit(enc.transform(X), np.zeros(40))
    for mode, dim in [("base", 10), ("baseh", 12), ("cross", 16), ("uv", 12 + 4)]:
        cfg = CrossPlannerConfig(state_arm="mf", transition="stochastic", feature_mode=mode)
        extra = {}
        if mode == "randcross":
            continue
        pl = CrossPlanner(V, ials_cfg, None, resp, None, enc, log_pop, q_hat, 0.3,
                          cfg=cfg, enc_resp=enc, enc_haz=enc, **extra)
        feats = pl.features(hstate, 3, events=[(0, 1.0), (1, 0.0)])
        assert feats.shape == (dim,), (mode, feats.shape)
        assert not np.isnan(feats).any()
    cfg = CrossPlannerConfig(state_arm="mf", transition="stochastic", feature_mode="randcross")
    rc = RandomCrosses(k=4, seed=0).fit(np.random.default_rng(5).normal(0, 1, (30, 12)))
    pl = CrossPlanner(V, ials_cfg, None, resp, None, enc, log_pop, q_hat, 0.3,
                      cfg=cfg, enc_resp=enc, enc_haz=enc, rand_crosses=rc)
    assert pl.features(hstate, 3, events=[(0, 1.0), (1, 0.0)]).shape == (16,)


def test_cross_planner_randcross_requires_fitted():
    from mf_mdp.mf import IALSConfig, user_state
    V, ials_cfg, log_pop, q_hat = _toy_mf_setup()
    enc = FeatureEncoder()
    enc.fit(np.zeros((2, 10)))
    cfg = CrossPlannerConfig(state_arm="mf", transition="stochastic", feature_mode="randcross")
    try:
        CrossPlanner(V, ials_cfg, None, None, None, enc, log_pop, q_hat, 0.3,
                     cfg=cfg, enc_resp=enc, enc_haz=enc)
    except ValueError:
        pass
    else:
        raise AssertionError("randcross without RandomCrosses must raise")
