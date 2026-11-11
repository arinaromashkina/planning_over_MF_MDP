"""iALS training sanity: recovers structure on synthetic data."""

import numpy as np

from mf_mdp.mf import IALSConfig, train_ials, user_state
from mf_mdp.mdp import similarity


def synthetic_feedback(n_users=300, n_items=60, dim=6, per_user=8, seed=0):
    rng = np.random.default_rng(seed)
    U_true = rng.normal(0, 1, (n_users, dim))
    V_true = rng.normal(0, 1, (n_items, dim))
    prefs = U_true @ V_true.T
    pairs = []
    for u in range(n_users):
        liked = np.argsort(-prefs[u])[:per_user]
        pairs.extend((u, i) for i in liked)
    return np.asarray(pairs), U_true, V_true


def test_ials_recovers_signal():
    pairs, _, _ = synthetic_feedback()
    cfg = IALSConfig(dim=6, alpha=1.0, lam=1.0, n_iter=20)
    n_users, n_items = 300, 60
    U, V = train_ials(pairs, n_users, n_items, cfg, seed=0)
    assert U.shape == (n_users, 6) and V.shape == (n_items, 6)
    # Held-out style check: fold-in scores rank a fresh positive above random items.
    rng = np.random.default_rng(123)
    hits = 0
    trials = 0
    for u in range(0, n_users, 5):
        hist = pairs[pairs[:, 0] == u][:, 1]
        heldout = hist[-1]
        state = user_state(V, hist[:-1], cfg)
        liked = set(int(i) for i in hist)
        pool = np.array([i for i in range(n_items) if i not in liked])
        cand = np.concatenate([[heldout], rng.choice(pool, 19, replace=False)])
        scores = [similarity(state.u, V[i], "cos") for i in cand]
        rank = int(np.argsort(-np.asarray(scores)).tolist().index(0))
        hits += rank < 5
        trials += 1
    # random baseline is 5/20 = 0.25; require clearly-above-random ranking
    assert hits / trials > 0.4
