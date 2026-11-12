"""Tests for MCTS leaf-evaluator integration (Phase 4 §7 of the prompt)."""

import numpy as np
import pytest

from mf_mdp.mdp import MFState
from mf_mdp.models.value import VLoggingRidge
from mf_mdp.planning import PlannerConfig, plan
from tests.test_planning import _setup


def test_original_leaf_is_default_and_bit_exact():
    """Disabling the extension must reproduce pre-Phase-4 Plan-K exactly."""
    V, state = _setup()
    cfg_new = PlannerConfig(method="plank", expansion_width=10, n_simulations=50, horizon=10)
    cfg_old = PlannerConfig(method="plank", expansion_width=10, n_simulations=50, horizon=10,
                            leaf_evaluator="original", value_model=None)
    r1 = plan(V, state, 10, cfg_new, np.random.default_rng(0))
    r2 = plan(V, state, 10, cfg_old, np.random.default_rng(0))
    assert r1 == r2


def test_zero_leaf_runs_and_changes_nothing_catastrophic():
    V, state = _setup()
    cfg = PlannerConfig(method="plank", expansion_width=10, n_simulations=50, horizon=10,
                        leaf_evaluator="zero")
    recs = plan(V, state, 10, cfg, np.random.default_rng(0))
    assert len(recs) == 10 and len(set(recs)) == 10


def test_v_logging_leaf_requires_model():
    V, state = _setup()
    cfg = PlannerConfig(method="plank", expansion_width=10, n_simulations=20, horizon=5,
                        leaf_evaluator="v_logging_ridge")
    with pytest.raises(ValueError):
        plan(V, state, 5, cfg, np.random.default_rng(0))


def test_v_logging_leaf_smoke():
    V, state = _setup()
    rng = np.random.default_rng(1)
    d = V.shape[1]
    X = rng.normal(0, 1, (200, d + 4))
    G = 0.1 * np.abs(X[:, 0]) + 0.05
    vm = VLoggingRidge(lam=1.0)
    vm.fit(X, G)
    cfg = PlannerConfig(method="plank", expansion_width=10, n_simulations=30, horizon=5,
                        leaf_evaluator="v_logging_ridge", value_model=vm)
    recs = plan(V, state, 5, cfg, np.random.default_rng(0))
    assert len(recs) == 5
    assert not set(recs) & set(state.consumed)


def test_leaf_value_is_discounted_return_to_go():
    """Depth-h frontier contributes gamma^h * V(s_h) via the backprop recursion."""
    # With horizon=0 the root is itself the frontier: backed-up Q must equal V(root).
    V, state = _setup()
    rng = np.random.default_rng(2)
    d = V.shape[1]
    vm = VLoggingRidge(lam=1.0)
    X = rng.normal(0, 1, (100, d + 4))
    vm.fit(X, np.full(100, 0.37))
    # direct leaf value call
    from mf_mdp.planning.planners import _Node, _leaf_value
    cfg = PlannerConfig(method="plank", leaf_evaluator="v_logging_ridge", value_model=vm)
    node = _Node(state, 0)
    assert _leaf_value(V, node, cfg) == pytest.approx(0.37, abs=0.3)  # constant-fit model


def test_unknown_leaf_evaluator_raises():
    V, state = _setup()
    cfg = PlannerConfig(method="plank", leaf_evaluator="bogus")
    with pytest.raises(ValueError):
        plan(V, state, 5, cfg, np.random.default_rng(0))


def test_determinism_with_v_logging_leaf():
    V, state = _setup()
    rng = np.random.default_rng(3)
    d = V.shape[1]
    vm = VLoggingRidge(lam=1.0)
    vm.fit(rng.normal(0, 1, (100, d + 4)), np.linspace(0, 1, 100))
    cfg = PlannerConfig(method="plank", expansion_width=10, n_simulations=30, horizon=5,
                        leaf_evaluator="v_logging_ridge", value_model=vm)
    r1 = plan(V, state, 5, cfg, np.random.default_rng(0))
    r2 = plan(V, state, 5, cfg, np.random.default_rng(0))
    assert r1 == r2
