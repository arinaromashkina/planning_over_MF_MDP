"""Tests for the Env-S controlled environment and planning-correctness checks (Gate B)."""

import numpy as np
import pytest

from mf_mdp.envs import EnvS, EnvSConfig, DPPolicy, GreedyPolicy, MCTSPolicy


@pytest.fixture()
def env():
    return EnvS(EnvSConfig(seed=0))


def test_probability_normalization(env):
    for s in range(env.K):
        for i in range(env.n_items):
            p = env.engagement_prob(s, i)
            assert 0.0 <= p <= 1.0
        for r in (0.0, 0.5, 1.0):
            assert 0.0 <= env.termination_prob(r) <= 1.0


def test_termination_decreases_in_reward(env):
    assert env.termination_prob(0.0) > env.termination_prob(0.8)


def test_assimilation_transition_frequency(env):
    """Engagement with an off-mode item moves the mode to the item's dominant mode ~rho."""
    rng = np.random.default_rng(0)
    s, i = 0, 6  # item 6 dominant mode 2
    moves = sum(env.next_mode(s, i, True, rng) == 2 for _ in range(4000))
    assert abs(moves / 4000 - env.cfg.rho) < 0.03
    # skip never moves
    assert all(env.next_mode(s, i, False, rng) == s for _ in range(100))


def test_fatigue_increments_on_same_mode_and_resets(env):
    assert env.next_fatigue(0, 0, 0) == 1
    assert env.next_fatigue(0, 1, 0) == 2
    assert env.next_fatigue(0, 2, 0) == 2
    assert env.next_fatigue(0, 2, 6) == 0  # off-mode item resets


def test_terminal_transition_ends_episode(env):
    rng = np.random.default_rng(1)
    # force termination by stubbing
    orig = env.termination_prob
    env.termination_prob = lambda r: 1.0
    r, term, ns, nf = env.step(0, 0, 0, rng)
    assert term is True or term == np.True_
    env.termination_prob = orig


def test_dp_bellman_backup_correctness(env):
    dp = DPPolicy(env)
    c = env.cfg
    # last-step values equal max one-step expected Q
    for s in range(env.K):
        for f in range(3):
            q1 = []
            for i in range(env.n_items):
                p = env.engagement_prob(s, i)
                r_eng = env.reward(s, f, i, True)
                term_eng = env.termination_prob(r_eng)
                ns = int(np.argmax(env.M[i]))
                nf = env.next_fatigue(s, f, i)
                q = p * (r_eng + 0.0) + (1 - p) * 0.0
                q1.append(q)
            assert dp.V[c.H - 1][s][f] == pytest.approx(max(q1), abs=1e-12)
    # Bellman residual at earlier steps
    for h in range(c.H - 1):
        for s in range(env.K):
            for f in range(3):
                i = dp.pi[h][s][f]
                p = env.engagement_prob(s, i)
                r_eng = env.reward(s, f, i, True)
                term_eng = env.termination_prob(r_eng)
                dominant = int(np.argmax(env.M[i]))
                if dominant == s:
                    eng_next = [(s, env.next_fatigue(s, f, i), 1.0)]
                else:
                    eng_next = [(dominant, 0, c.rho), (s, 0, 1.0 - c.rho)]
                q = 0.0
                for ns, nf, prob in eng_next:
                    q += prob * (r_eng + c.gamma * (1.0 - term_eng) * dp.V[h + 1][ns][nf])
                q *= p
                term_skip = env.termination_prob(0.0)
                q += (1.0 - p) * (c.gamma * (1.0 - term_skip) * dp.V[h + 1][s][f])
                assert dp.V[h][s][f] == pytest.approx(q, abs=1e-10)


def test_two_step_enumeration_matches_dp(env):
    """Exact 2-step expected value at h=H-2 matches DP value (assimilation averaged)."""
    dp = DPPolicy(env)
    c = env.cfg
    h = c.H - 2
    s, f = 0, 0
    best = -np.inf
    for i0 in range(env.n_items):
        p0 = env.engagement_prob(s, i0)
        r0e = env.reward(s, f, i0, True)
        te = env.termination_prob(r0e)
        dominant = int(np.argmax(env.M[i0]))
        if dominant == s:
            eng_next = [(s, env.next_fatigue(s, f, i0), 1.0)]
        else:
            eng_next = [(dominant, 0, c.rho), (s, 0, 1.0 - c.rho)]
        v_e = 0.0
        for ns, nf, prob in eng_next:
            v1 = max(
                env.engagement_prob(ns, i1) * env.reward(ns, nf, i1, True)
                for i1 in range(env.n_items)
            )
            v_e += prob * (r0e + c.gamma * (1 - te) * v1)
        ts = env.termination_prob(0.0)
        v_s = c.gamma * (1 - ts) * max(
            env.engagement_prob(s, i1) * env.reward(s, f, i1, True)
            for i1 in range(env.n_items)
        )
        best = max(best, p0 * v_e + (1 - p0) * v_s)
    assert dp.V[h][s][f] == pytest.approx(best, abs=1e-10)


def test_greedy_differs_from_optimal(env):
    dp, greedy = DPPolicy(env), GreedyPolicy(env)
    disagreements = 0
    for s in range(env.K):
        for f in range(3):
            if dp.pi[0][s][f] != greedy(s, f, 0):
                disagreements += 1
    assert disagreements > 0, "handcrafted env must create greedy/optimal tension"


def test_dp_beats_greedy_in_rollouts(env):
    dp, greedy = DPPolicy(env), GreedyPolicy(env)
    rng = np.random.default_rng(2)
    n = 2000
    J_dp = np.mean([env.rollout(dp, 0, 0, rng) for _ in range(n)])
    J_greedy = np.mean([env.rollout(greedy, 0, 0, rng) for _ in range(n)])
    assert J_dp > J_greedy


def test_mcts_converges_to_dp_choice(env):
    dp = DPPolicy(env)
    mcts = MCTSPolicy(env, n_sims=20000, c_uct=1.4, rollout_greedy=True)
    rng = np.random.default_rng(3)

    def dp_q(s, f, i, V):
        c = env.cfg
        p = env.engagement_prob(s, i)
        r_eng = env.reward(s, f, i, True)
        te = env.termination_prob(r_eng)
        dominant = int(np.argmax(env.M[i]))
        if dominant == s:
            eng_next = [(s, env.next_fatigue(s, f, i), 1.0)]
        else:
            eng_next = [(dominant, 0, c.rho), (s, 0, 1.0 - c.rho)]
        q = sum(prob * (r_eng + c.gamma * (1 - te) * V[1][ns][nf]) for ns, nf, prob in eng_next) * p
        ts = env.termination_prob(0.0)
        return q + (1 - p) * (c.gamma * (1 - ts) * V[1][s][f])

    agree, tie, disagree = 0, 0, []
    for s in range(env.K):
        for f in range(3):
            a = mcts.action(s, f, rng)
            qs = sorted((dp_q(s, f, i, dp.V) for i in range(env.n_items)), reverse=True)
            gap = qs[0] - qs[1]
            if a == dp.pi[0][s][f]:
                agree += 1
            elif gap < 0.02:  # near-tie: choice is statistically equivalent
                tie += 1
            else:
                disagree.append((s, f, a, int(dp.pi[0][s][f]), gap))
    assert not disagree, f"non-tie disagreements: {disagree}"
    assert agree + tie == 12
    print(f"\nmcts-dp: agree={agree} near-tie={tie}")


def test_determinism(env):
    rng1, rng2 = np.random.default_rng(7), np.random.default_rng(7)
    dp = DPPolicy(env)
    traj1 = [env.rollout(dp, 0, 0, rng1) for _ in range(50)]
    traj2 = [env.rollout(dp, 0, 0, rng2) for _ in range(50)]
    assert traj1 == traj2
