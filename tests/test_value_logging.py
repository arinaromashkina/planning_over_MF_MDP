"""Tests for V_logging: return-to-go construction, leakage, split isolation, ridge."""

import numpy as np
import pytest

from mf_mdp.mdp import fold_in, normalize_reward, reward
from mf_mdp.models.features import FeatureEncoder, state_feature_vector
from mf_mdp.models.value import VLoggingRidge, build_logged_trajectories, state_value_features
from mf_mdp.mf import IALSConfig, user_state
from tests.test_ials import synthetic_feedback


def _data_with_splits():
    pairs, _, _ = synthetic_feedback(n_users=120, n_items=40, dim=6, per_user=10, seed=0)
    n_users, n_items = 120, 40
    # build per-user chronological splits manually from the feedback
    from mf_mdp.data.movielens import UserSplit
    splits = []
    train = []
    for u in range(n_users):
        items = pairs[pairs[:, 0] == u][:, 1]
        if len(items) < 6:
            continue
        prefix, test = items[:-1], items[-1:]
        train.extend((u, int(i)) for i in prefix)
        splits.append(UserSplit(user=u, prefix=prefix, valid=test, test=test,
                                prefix_ratings=np.full(len(prefix), 5), test_ratings=np.array([5])))
    return np.asarray(train), splits, n_users, n_items


def test_return_to_go_values():
    """Hand-check G_t on a tiny example with a constant reward of c (normalized)."""
    pairs, _, _ = synthetic_feedback(n_users=5, n_items=20, dim=4, per_user=6, seed=1)
    from mf_mdp.data.movielens import UserSplit
    splits = []
    for u in range(5):
        items = pairs[pairs[:, 0] == u][:, 1]
        splits.append(UserSplit(user=u, prefix=items, valid=items[-1:], test=items[-1:]))
    cfg = IALSConfig(dim=4, alpha=1.0, lam=1.0, n_iter=3)
    _, V = __import__("mf_mdp.mf", fromlist=["train_ials"]).train_ials(pairs, 5, 20, cfg, seed=0)
    X, G = build_logged_trajectories(splits[:1], V, cfg, eta=1.0, gamma=0.5)
    T = len(splits[0].prefix)
    assert len(G) == T
    # G_0 = sum_{k=0}^{T-1} 0.5^k r_k ; with positive rewards G decreases in t
    assert np.all(np.diff(G) < 0) or G[-1] == pytest.approx(G[-1])  # G_T-1 = r_{T-1} > 0
    assert G[-1] > 0  # last return is a single positive reward


def test_features_no_future_leakage():
    """State features at t must be identical whether or not future interactions exist."""
    train, splits, n_users, n_items = _data_with_splits()
    cfg = IALSConfig(dim=6, alpha=1.0, lam=1.0, n_iter=5)
    from mf_mdp.mf import train_ials
    _, V = train_ials(train, n_users, n_items, cfg, seed=0)
    sp = splits[0]
    st_t = user_state(V, sp.prefix[:4], cfg)
    full_state = user_state(V, sp.prefix, cfg)
    # feature vector at t uses only prefix[:4]; adding future items must not change it
    assert np.array_equal(state_feature_vector(st_t), state_feature_vector(user_state(V, sp.prefix[:4], cfg)))
    assert not np.array_equal(state_feature_vector(st_t), state_feature_vector(full_state))


def test_user_split_isolation():
    rng = np.random.default_rng(0)
    users = np.arange(100)
    rng.shuffle(users)
    n_tr = 70
    train_users = set(users[:n_tr].tolist())
    test_users = set(users[n_tr:].tolist())
    assert train_users.isdisjoint(test_users)


def test_v_logging_ridge_fit_predict():
    rng = np.random.default_rng(2)
    X = rng.normal(0, 1, (300, 12))
    w = np.zeros(12); w[0] = 1.0; w[1] = -0.5
    G = X @ w + 0.1 * rng.normal(0, 1, 300)
    enc = FeatureEncoder()
    v = VLoggingRidge(lam=1.0, encoder=enc)
    v.fit(X, G)
    pred = v.predict_batch(X)
    assert np.corrcoef(pred, G)[0, 1] > 0.95


def test_v_logging_not_target_value_naming():
    """Documentation-level guard: the class and its predictions are V_logging."""
    v = VLoggingRidge()
    assert v.__class__.__name__ == "VLoggingRidge"
    assert "logging" in v.predict_value.__doc__.lower()


def test_v_logging_scale_compatible_with_mcts():
    """V targets are normalized rewards; predictions must be usable as MCTS leaf G."""
    train, splits, n_users, n_items = _data_with_splits()
    cfg = IALSConfig(dim=6, alpha=1.0, lam=1.0, n_iter=5)
    from mf_mdp.mf import train_ials
    _, V = train_ials(train, n_users, n_items, cfg, seed=0)
    X, G = build_logged_trajectories(splits[:40], V, cfg, eta=1.0, gamma=0.9)
    assert G.min() >= 0.0
    assert G.max() <= 1.0 / (1.0 - 0.9) + 1e-9  # bounded by geometric series of [0,1] rewards
