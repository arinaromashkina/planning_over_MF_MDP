"""Planner structure and end-to-end synthetic smoke tests."""

import numpy as np

from mf_mdp.mf import IALSConfig, train_ials, user_state
from mf_mdp.planning import PlannerConfig, plan, static_topk
from tests.test_ials import synthetic_feedback


def _setup(seed=0):
    pairs, _, _ = synthetic_feedback()
    cfg = IALSConfig(dim=6, alpha=1.0, lam=1.0, n_iter=15)
    U, V = train_ials(pairs, 300, 60, cfg, seed=seed)
    hist = pairs[pairs[:, 0] == 5][:, 1][:-1]
    state = user_state(V, hist, cfg)
    return V, state


def test_static_excludes_consumed():
    V, state = _setup()
    recs = static_topk(V, state, 10, "cos")
    assert len(recs) == 10
    assert not set(recs) & set(state.consumed)


def test_plan1_structure():
    V, state = _setup()
    cfg = PlannerConfig(method="plan1", plan1_pool=30, eta=1.0)
    recs = plan(V, state, 10, cfg)
    assert len(recs) == 10
    assert len(set(recs)) == 10
    assert not set(recs) & set(state.consumed)


def test_plank_structure_and_determinism():
    V, state = _setup()
    cfg = PlannerConfig(method="plank", expansion_width=10, n_simulations=50, horizon=10)
    rng = np.random.default_rng(0)
    recs1 = plan(V, state, 10, cfg, rng)
    recs2 = plan(V, state, 10, cfg, np.random.default_rng(0))
    assert recs1 == recs2
    assert len(recs1) == 10 and len(set(recs1)) == 10
    assert not set(recs1) & set(state.consumed)


def test_plan1_variants_run():
    V, state = _setup()
    for variant in ("post", "root", "bellman"):
        cfg = PlannerConfig(method="plan1", plan1_variant=variant, plan1_pool=30)
        recs = plan(V, state, 5, cfg)
        assert len(recs) == 5, variant
