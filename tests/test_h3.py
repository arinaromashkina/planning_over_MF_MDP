"""H3 tests: SASRec causality/masking, proposer correctness, pool-restricted MCTS
invariants, and the J(pi) rollout. Torch-dependent tests skip when torch is absent
(docs/H3_LEARNED_ACTION_PROPOSALS_DESIGN.md §4)."""

from __future__ import annotations

import numpy as np
import pytest

from mf_mdp.mdp import MFState, fold_in, reward
from mf_mdp.mf import IALSConfig, user_state
from mf_mdp.planning.planners import PlannerConfig
from mf_mdp.proposal.planning import _expand_pool, _Node, planK_topk_pool, trajectory_return
from mf_mdp.proposal.pools import (
    coverage,
    mf_pool,
    mf_random_pool,
    oracle_coverage,
    sasrec_pool_from_scores,
    union_pool,
)

try:
    import torch

    from mf_mdp.proposal.sasrec import (
        SASRec,
        SASRecConfig,
        build_seq_array,
        score_histories,
    )

    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


# ---------------------------------------------------------------- pools ----

def test_sasrec_pool_excludes_consumed_and_sorts():
    scores = np.array([0.9, 0.1, 0.8, 0.7, 0.2])
    pool = sasrec_pool_from_scores(scores, consumed=(0,), m=3)
    assert list(pool) == [2, 3, 4]  # 0 excluded despite top score
    pool_all = sasrec_pool_from_scores(scores, consumed=(), m=5)
    assert list(pool_all) == [0, 2, 3, 4, 1]


def test_mf_pool_matches_exact_topl():
    rng = np.random.default_rng(0)
    V = rng.normal(size=(20, 4))
    u = rng.normal(size=4)
    consumed = (1, 5)
    pool = mf_pool(V, u, consumed, 5)
    assert len(pool) == 5
    assert not set(pool) & set(consumed)
    scores = V @ u / (np.linalg.norm(V, axis=1) * np.linalg.norm(u))
    expected = np.argsort(-scores)
    expected = [a for a in expected if a not in consumed][:5]
    assert list(pool) == expected


def test_union_pool_dedup_preserves_order():
    a = np.array([3, 1, 4])
    b = np.array([1, 5, 3, 2])
    u = union_pool(a, b)
    assert list(u) == [3, 1, 4, 5, 2]


def test_mf_random_pool_sizes_and_disjointness():
    rng = np.random.default_rng(1)
    V = rng.normal(size=(50, 4))
    u = rng.normal(size=4)
    consumed = (0, 1, 2)
    m, rho = 20, 0.25
    pool = mf_random_pool(V, u, consumed, m, rho, rng)
    assert len(pool) == m  # 15 greedy + 5 random
    assert not set(pool) & set(consumed)
    greedy = mf_pool(V, u, consumed, int(np.ceil((1 - rho) * m)))
    n_random = len(set(pool) - set(greedy))
    assert n_random == int(np.floor(rho * m))


def test_coverage_metrics():
    pools = [np.array([1, 2]), np.array([3]), np.array([2, 5])]
    assert coverage(pools, np.array([2, 3, 4])) == pytest.approx(2 / 3)
    assert oracle_coverage(pools, np.array([5, 3, 2])) == pytest.approx(2 / 3)
    with pytest.raises(ValueError):
        coverage(pools, np.array([1]))


# ------------------------------------------------- pool-restricted MCTS ----

def _toy_v(d: int = 8, n: int = 12, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, d))


def test_pool_mcts_never_expands_outside_pool():
    V = _toy_v()
    ials = IALSConfig(dim=V.shape[1])
    state = user_state(V, np.array([0, 3, 5]), ials)
    pool = np.array([1, 2, 4, 6, 8])
    cfg = PlannerConfig(n_simulations=40, horizon=4, expansion_width=3)
    recs = planK_topk_pool(V, state, 3, cfg, np.random.default_rng(0), pool)
    assert set(recs) <= set(pool)
    # Direct expansion check: branching factor = min(expansion_width, pool-at-node),
    # children always inside the pool.
    node = _Node(state, 0)
    _expand_pool(V, node, cfg, pool)
    assert set(node.children) <= set(pool)
    assert len(node.children) == min(cfg.expansion_width, len(pool))


def test_pool_mcts_degenerate_pool_falls_back():
    V = _toy_v()
    ials = IALSConfig(dim=V.shape[1])
    state = user_state(V, np.array([0, 3]), ials)
    cfg = PlannerConfig(n_simulations=5, horizon=2)
    recs = planK_topk_pool(V, state, 2, cfg, np.random.default_rng(0), np.array([], dtype=np.int64))
    assert len(recs) == 2  # fell back to unrestricted catalog search


def test_trajectory_return_matches_manual_sum():
    V = _toy_v()
    ials = IALSConfig(dim=V.shape[1])
    state = user_state(V, np.array([0]), ials)
    items = [2, 4, 1]
    eta, gamma, sim = 1.0, 0.9, "cos"
    total, cur, disc = 0.0, state, 1.0
    for a in items:
        v = V[a]
        P_plus, u_plus, d_s = fold_in(cur, v)
        total += disc * reward(cur.u, v, d_s, eta, sim)
        cur = MFState(P=P_plus, u=u_plus, consumed=tuple(sorted(cur.consumed + (a,))))
        disc *= gamma
    assert trajectory_return(V, state, items, eta, sim, gamma) == pytest.approx(total)


# --------------------------------------------------------------- SASRec ----

@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_sasrec_causality_future_tokens_do_not_change_past_hidden():
    cfg = SASRecConfig(n_items=10, dim=16, n_layers=2, n_heads=2, maxlen=8, dropout=0.0)
    model = SASRec(cfg).eval()
    a = np.full((1, 8), cfg.pad, dtype=np.int64)
    a[0, -3:] = [1, 2, 3]
    b = a.copy()
    b[0, -1] = 7  # change the LAST token only
    with torch.no_grad():
        ha = model.encode(torch.as_tensor(a))[:, -3]  # position of token 1
        hb = model.encode(torch.as_tensor(b))[:, -3]
    assert torch.allclose(ha, hb)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_sasrec_left_padding_invariance():
    cfg = SASRecConfig(n_items=10, dim=16, n_layers=1, n_heads=2, maxlen=8, dropout=0.0)
    model = SASRec(cfg).eval()
    full = np.array([[4, 5, 6, 7, 1, 2, 3, 3]], dtype=np.int64)  # length 8, no pad
    padded = np.full((1, 12), cfg.pad, dtype=np.int64)  # same tail, 4 pads (maxlen sliced)
    padded[0, -8:] = full[0]
    with torch.no_grad():
        la = model.logits(torch.as_tensor(full))
        lb = model.logits(torch.as_tensor(padded))
    assert torch.allclose(la, lb, atol=1e-5)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_build_seq_array_left_pads_and_truncates():
    n_items, maxlen = 100, 5
    seqs = build_seq_array(
        [np.array([1, 2, 3]), np.array([7, 8, 9, 10, 11, 12]), np.empty(0, dtype=np.int64)],
        n_items, maxlen,
    )
    assert seqs.shape == (3, 5)
    assert list(seqs[0]) == [n_items, n_items, 1, 2, 3]
    assert list(seqs[1]) == [8, 9, 10, 11, 12]  # truncated to the last 5
    assert list(seqs[2]) == [n_items] * 5


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_score_histories_shapes():
    cfg = SASRecConfig(n_items=13, dim=8, n_layers=1, n_heads=1, maxlen=6, dropout=0.0)
    model = SASRec(cfg).eval()
    seqs = build_seq_array([np.array([1, 2]), np.array([3])], cfg.n_items, cfg.maxlen)
    out = score_histories(model, seqs, device="cpu")
    assert out.shape == (2, 13)


@pytest.mark.skipif(not _HAS_TORCH, reason="torch not installed")
def test_fit_sasrec_overfits_tiny_set():
    """Sanity: the training loop can drive the loss down on a tiny deterministic set."""
    cfg = SASRecConfig(n_items=6, dim=8, n_layers=1, n_heads=1, maxlen=4, dropout=0.0)
    model = SASRec(cfg)
    seqs = np.array(
        [[6, 6, 0, 1], [6, 6, 1, 2], [6, 0, 2, 3], [6, 6, 3, 4], [6, 4, 5, 0], [6, 6, 5, 1]]
    )  # pad id = 6
    model, history = __import__("mf_mdp.proposal.sasrec", fromlist=["fit_sasrec"]).fit_sasrec(
        model, seqs, valid_fn=lambda m: 0.0, device="cpu",
        max_epochs=30, patience=30, seed=0, verbose=False,
    )
    assert history["train_loss"][-1] < history["train_loss"][0]
