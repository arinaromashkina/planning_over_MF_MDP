"""Fold-in / Sherman–Morrison vs direct recomputation (paper Appendix A)."""

import numpy as np
import pytest

from mf_mdp.mdp import MFState, fold_in, reward, similarity
from mf_mdp.mf import IALSConfig, user_state

D = 16


def make_state(rng, consumed=(0, 3)):
    V = rng.normal(0, 1, (20, D))
    cfg = IALSConfig(dim=D, alpha=1.0, lam=1.0, n_iter=3)
    items = np.array(list(consumed))
    state = user_state(V, items, cfg)
    return V, cfg, state


def test_sherman_morrison_vs_direct_inverse():
    rng = np.random.default_rng(0)
    V, cfg, state = make_state(rng)
    v = V[7]
    A = np.linalg.inv(state.P)
    P_plus, _, _ = fold_in(state, v)
    A_plus_direct = A + np.outer(v, v)
    np.testing.assert_allclose(P_plus, np.linalg.inv(A_plus_direct), atol=1e-10)


def test_fold_in_paper_semantics():
    """Paper eq. (13): the optimistic transition adds a NEW unit-confidence observation on
    top of the baseline: A+ = A + vv^T, b+ = b + v.

    Consequently A+ coincides with a re-fold of history + {a}, but b+ does NOT (a naive
    re-fold under eq. (2) with alpha=1 would give b + 2v, since a history item carries
    c=2, p=1). u+ must equal the direct WLS solve with b+ = b + v.
    """
    rng = np.random.default_rng(1)
    V, cfg, state = make_state(rng)
    a = 7
    A = np.linalg.inv(state.P)
    b = A @ state.u
    P_plus, u_plus, _ = fold_in(state, V[a])
    refold = user_state(V, np.array([0, 3, a]), cfg)
    # Precision matches the naive re-fold exactly...
    np.testing.assert_allclose(P_plus, refold.P, atol=1e-8)
    # ...but the user vector follows the paper's b+ = b + v, not the re-fold's b + 2v.
    u_direct = np.linalg.solve(A + np.outer(V[a], V[a]), b + V[a])
    np.testing.assert_allclose(u_plus, u_direct, atol=1e-8)
    assert not np.allclose(u_plus, refold.u, atol=1e-3)


def test_fold_in_matches_direct_wls():
    """u+ solves the exact weighted least-squares problem with the item added as positive."""
    rng = np.random.default_rng(2)
    V, cfg, state = make_state(rng)
    a = 7
    A = np.linalg.inv(state.P)
    b = A @ state.u
    v = V[a]
    _, u_plus, _ = fold_in(state, v)
    A_plus, b_plus = A + np.outer(v, v), b + v
    u_direct = np.linalg.solve(A_plus, b_plus)
    np.testing.assert_allclose(u_plus, u_direct, atol=1e-8)


def test_delta_s_formula():
    rng = np.random.default_rng(3)
    V, _, state = make_state(rng)
    a = 7
    v = V[a]
    _, u_plus, delta_s = fold_in(state, v)
    delta_direct = float(u_plus @ v - state.u @ v)
    np.testing.assert_allclose(delta_s, delta_direct, atol=1e-10)


def test_reward_is_sum_form():
    rng = np.random.default_rng(4)
    V, _, state = make_state(rng)
    v = V[7]
    _, u_plus, delta_s = fold_in(state, v)
    r = reward(u_plus, v, delta_s, eta=1.5, kind="cos")
    expected = similarity(u_plus, v, "cos") + 1.5 / (1.0 + abs(delta_s))
    assert r == pytest.approx(expected)
