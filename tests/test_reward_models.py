"""Tests for R1/R2 reward models and the response model."""

import numpy as np
import pytest

from mf_mdp.mdp import MFState
from mf_mdp.mf import IALSConfig
from mf_mdp.models.features import FeatureEncoder, features_for
from mf_mdp.models.response import (
    LogisticResponseModel,
    brier_score,
    expected_calibration_error,
    log_loss,
    roc_auc,
)
from mf_mdp.models.reward import (
    R1ExpectedReward,
    R2DirectExpectedReward,
    UtilityMapping,
    fit_ridge,
    ridge_predict,
)


def _toy(rng, n=200, d=10):
    X = rng.normal(0, 1, (n, d))
    w = rng.normal(0, 1, d)
    p = 1 / (1 + np.exp(-X @ w))
    y = (rng.random(n) < p).astype(int)
    return X, y


def test_utility_mapping():
    u = UtilityMapping(u_rating5=1.0, u_rating4=0.25)
    assert u.utility(5) == 1.0
    assert u.utility(4) == 0.25
    assert u.utility(3) == 0.0


def test_response_model_learns():
    rng = np.random.default_rng(0)
    X, y = _toy(rng)
    enc = FeatureEncoder()
    m = LogisticResponseModel(n_steps=800)
    m.fit(enc.fit_transform(X), y)
    p = m.predict_proba(enc.transform(X))
    assert roc_auc(y, p) > 0.9
    assert log_loss(y, p) < 0.5
    assert 0.0 <= expected_calibration_error(y, p) <= 0.2


def test_expected_reward_discrete_math():
    """R1 = p5*U5 + (1-p5)*U4 must equal the discrete expectation."""
    rng = np.random.default_rng(1)
    X, y = _toy(rng, n=100)
    enc = FeatureEncoder()
    m = LogisticResponseModel(n_steps=300)
    m.fit(enc.fit_transform(X), y)
    util = UtilityMapping(u_rating5=1.0, u_rating4=0.25)
    r1 = R1ExpectedReward(lambda f: m.predict_proba(f), util, enc)
    V = rng.normal(0, 1, (30, 8))
    st = MFState(P=np.eye(8), u=np.ones(8) * 0.1, consumed=())
    log_pop = np.zeros(30)
    x = enc.transform(features_for(st, V, 3, log_pop)[None, :])
    p5 = float(m.predict_proba(x)[0])
    assert r1.predict_reward(st, V, 3, log_pop) == pytest.approx(p5 * 1.0 + (1 - p5) * 0.25)


def test_r1_with_alignment_term():
    rng = np.random.default_rng(2)
    enc = FeatureEncoder()
    enc.fit(np.ones((4, 10)))
    r1 = R1ExpectedReward(lambda f: np.full(len(f), 0.8), UtilityMapping(), enc, eta=1.0)
    V = rng.normal(0, 1, (30, 8))
    st = MFState(P=np.eye(8), u=np.ones(8) * 0.01, consumed=())
    log_pop = np.zeros(30)
    base = 0.8 * 1.0 + 0.2 * 0.25
    val = r1.predict_reward(st, V, 3, log_pop)
    assert val > base  # alignment term is positive


def test_r2_direct_expected_reward():
    """R2 learns a linear utility target from real features_for vectors."""
    rng = np.random.default_rng(3)
    cfg = IALSConfig(dim=6, alpha=1.0, lam=1.0, n_iter=3)
    V = rng.normal(0, 1, (30, 6))
    P = np.eye(6) + V[:5].T @ V[:5]
    log_pop = rng.integers(0, 50, 30).astype(float)
    states, X_rows, y = [], [], []
    for _ in range(400):
        items = rng.choice(30, 6, replace=False)
        st = MFState(P=P, u=P @ V[items].sum(axis=0), consumed=tuple(sorted(items.tolist())))
        a = int(rng.choice([i for i in range(30) if i not in items]))
        X_rows.append(features_for(st, V, a, log_pop))
        states.append((st, a))
    X = np.vstack(X_rows)
    w_true = np.zeros(X.shape[1]); w_true[0] = 0.5; w_true[4] = -0.3
    y = X @ w_true + 0.05 * rng.normal(0, 1, len(X))
    enc = FeatureEncoder()
    w = fit_ridge(enc.fit_transform(X), y, lam=1.0)
    pred = ridge_predict(enc.transform(X), w)
    assert np.corrcoef(pred, y)[0, 1] > 0.9
    r2 = R2DirectExpectedReward(w, enc)
    st, a = states[0]
    val = r2.predict_reward(st, V, a, log_pop)
    assert isinstance(val, float)
    x1 = enc.transform(features_for(st, V, a, log_pop)[None, :])
    assert val == pytest.approx(float((np.hstack([x1, [[1.0]]]) @ w).item()), rel=1e-6)


def test_metrics_functions():
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.4, 0.35, 0.8])
    assert roc_auc(y, p) == pytest.approx(0.75)
    assert brier_score(y, p) > 0
    assert log_loss(y, p) > 0
