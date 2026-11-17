"""Tests for the H1 line (belief vs preference dynamics): filters, regimes, no-leakage."""

import numpy as np
import pytest

from mf_mdp.belief import JointMFBelief, MFPosteriorBelief, ParticleBelief
from mf_mdp.envs.h1_envs import H1Env, H1EnvConfig, make_J, make_L, make_P
from mf_mdp.planning.stochastic_belief import (
    BeliefMCTSConfig,
    BeliefMCTSPolicy,
    MyopicPolicy,
    OracleHistoryPolicy,
    OracleStatePolicy,
)


def _env_P():
    return make_P(seed=0)


# filter correctness ---------------------------------------------------------

def test_joint_belief_tracks_theta_in_envP():
    """In Env-P (known theta0, assimilation), JointMFBelief tracks theta; static drifts."""
    env = _env_P()
    rng = np.random.default_rng(0)
    theta0 = env.sample_theta0(rng)
    b_static = env.initial_belief(theta0, rng)
    b_joint = env.initial_belief_joint(theta0, rng)
    theta = theta0.copy()
    f = 0
    for _ in range(6):
        i = int(np.argmax([env.q[j] * float(theta @ env.V[j]) for j in range(env.n_items)]))
        r, term, theta, nf, eng = env.step(theta, f, i, rng, theta0)
        b_static = b_static.update(env, i, True)
        b_joint = b_joint.update(env, i, True)
        f = nf
        if term:
            break
    assert np.linalg.norm(b_joint.mean() - theta) < np.linalg.norm(b_static.mean() - theta)


def test_particle_filter_tracks_theta_in_envP():
    """SIR particles with true kernels track theta (history-oracle belief)."""
    env = _env_P()
    rng = np.random.default_rng(1)
    theta0 = env.sample_theta0(rng)
    b = env.initial_particles(theta0, rng, n=300)
    theta = theta0.copy()
    f = 0
    for _ in range(5):
        i = int(np.argmax([env.q[j] * float(theta @ env.V[j]) for j in range(env.n_items)]))
        r, term, theta, nf, eng = env.step(theta, f, i, rng, theta0)
        b = b.update(env, i, eng)
        f = nf
        if term:
            break
    assert np.linalg.norm(b.mean() - theta) < 0.2


def test_envL_static_theta_regime():
    """Env-L: push_theta keeps theta fixed (alpha=0, lam=0)."""
    env = make_L(seed=0)
    rng = np.random.default_rng(2)
    th = env.sample_theta0(rng)
    th2 = env.push_theta(th, 3, True, th)
    np.testing.assert_allclose(th2, th, atol=1e-9)


def test_fold_in_weights_phase5_convention():
    env = _env_P()
    a_w, b_w = env.fold_in_weights(True)
    assert a_w == env.cfg.ials_alpha and b_w == 1.0 + env.cfg.ials_alpha
    a_w, b_w = env.fold_in_weights(False)
    assert a_w == -0.5 * env.cfg.ials_alpha and b_w == 0.0


# no-leakage API -------------------------------------------------------------

def test_policy_path_has_no_theta():
    """Policy state after reset/act/update holds only Belief objects, never theta arrays."""
    env = _env_P()
    pol = BeliefMCTSPolicy(env, env.initial_belief_joint,
                           BeliefMCTSConfig(n_sims=10, depth=4, observe="sample", update="joint"))
    rng = np.random.default_rng(3)
    th = env.sample_theta0(rng)
    state = pol.reset(th, rng)
    assert isinstance(state, (JointMFBelief, MFPosteriorBelief, ParticleBelief))
    i = pol.act(state, 0, 0, rng)
    state2 = pol.update(state, i, True)
    assert isinstance(state2, (JointMFBelief, MFPosteriorBelief, ParticleBelief))
    # the env theta array must not be referenced by the belief state
    assert not any(np.array_equal(getattr(state2, a), th) for a in ("u", "P", "particles", "weights") if hasattr(state2, a))


# determinism ----------------------------------------------------------------

def test_rollout_deterministic():
    env = _env_P()
    pol = BeliefMCTSPolicy(env, env.initial_belief, BeliefMCTSConfig(n_sims=20, depth=4))
    th = env.sample_theta0(np.random.default_rng(0))
    r1 = env.rollout(pol, th, np.random.default_rng(7))
    th = env.sample_theta0(np.random.default_rng(0))
    r2 = env.rollout(pol, th, np.random.default_rng(7))
    assert r1 == r2


def test_greedy_differs_from_oracle_in_envP():
    """Pilot sign-off: steering tension exists (greedy != oracle-state J)."""
    from experiments.h1.run_e1_isolation import run_pilots
    pilots = run_pilots()
    assert pilots["pilot1_pass"], pilots
    assert pilots["pilot2_pass"], pilots
