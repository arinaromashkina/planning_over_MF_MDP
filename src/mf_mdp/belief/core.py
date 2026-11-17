"""Belief representations for the H1 line (belief updates vs preference dynamics).

Filter factorization (design doc §2.2):
    b_{t+1} = Push_T( BayesUpdate(b_t, a_t, y_t) )

- MFPosteriorBelief: Gaussian N(u, P) on the fold-in posterior; BayesUpdate = Phase-5
  T1 weighted fold-in; Push_T = identity (the paper's silent static-preference kernel).
- JointMFBelief: same, WITH the push (doc §2.2): mu <- normalize((1-lam) mu^+ + alpha(y) v
  + lam * theta_bar), Sigma <- (1-lam)^2 Sigma^+ + Q. O(d^2).
- ParticleBelief: SIR particle filter implementing the EXACT filter (Bernoulli response
  likelihood + true deterministic push). Used as the history-oracle belief and as the
  robustness variant isolating representation error.

The planner sees ONLY the Belief interface (mean / predictive_prob / update / samples).
The no-leakage rule (env theta never enters the policy path) is enforced by API and
tested in tests/test_h1.py.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


class Belief:
    """Interface for belief over the user's latent state."""

    def mean(self) -> np.ndarray:
        raise NotImplementedError

    def predictive_prob(self, env, item: int) -> float:
        """P(y=1 | belief, item) under the response model."""
        raise NotImplementedError

    def expected_reward(self, env, item: int, f_streak: int = 0) -> float:
        raise NotImplementedError

    def update(self, env, item: int, engaged: bool) -> "Belief":
        raise NotImplementedError

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        raise NotImplementedError


def _resp_logit(env, theta: np.ndarray, item: int) -> float:
    return float(env.cfg.a * (theta @ env.V[item]) + env.cfg.b * env.q[item])


class MFPosteriorBelief(Belief):
    """Gaussian on the fold-in posterior; static-preference kernel (Push_T = id)."""

    def __init__(self, u: np.ndarray, P: np.ndarray):
        self.u = u
        self.P = P

    def mean(self) -> np.ndarray:
        return self.u

    def predictive_prob(self, env, item: int) -> float:
        z = _resp_logit(env, self.u, item)
        return float(1.0 / (1.0 + np.exp(-z)))

    def expected_reward(self, env, item: int, f_streak: int = 0) -> float:
        p = self.predictive_prob(env, item)
        cos = float(self.u @ env.V[item])
        return p * env.q[item] * max(cos, 0.0) * (1.0 - env.cfg.kappa_sat * f_streak)

    def update(self, env, item: int, engaged: bool) -> "Belief":
        from mf_mdp.mdp import fold_in_general

        v = env.V[item]
        a_w, b_w = env.fold_in_weights(engaged)
        P_p, u_p, _ = fold_in_general(
            _State(self.P, self.u), v, a_w, b_w
        )
        return MFPosteriorBelief(u_p, P_p)


@dataclass
class _State:
    """Minimal shim to reuse fold_in_general without importing MFState semantics."""

    P: np.ndarray
    u: np.ndarray
    consumed: tuple = ()


class JointMFBelief(MFPosteriorBelief):
    """MF posterior belief WITH the preference-dynamics push (doc §2.2)."""

    def __init__(self, u, P, lam: float, theta_bar: np.ndarray, Q: float = 1e-3):
        super().__init__(u, P)
        self.lam = lam
        self.theta_bar = theta_bar
        self.Q = Q

    def update(self, env, item: int, engaged: bool) -> "JointMFBelief":
        base = super().update(env, item, engaged)
        alpha_y = env.alpha_of(engaged)
        new_u = (1.0 - self.lam) * base.u + alpha_y * env.V[item] + self.lam * self.theta_bar
        nrm = np.linalg.norm(new_u)
        if nrm > 0:
            new_u = new_u / nrm
        new_P = (1.0 - self.lam) ** 2 * base.P + self.Q * np.eye(base.P.shape[0])
        return JointMFBelief(new_u, new_P, self.lam, self.theta_bar, self.Q)


class ParticleBelief(Belief):
    """SIR particle filter: exact BayesUpdate (Bernoulli likelihood) + true push."""

    def __init__(self, particles: np.ndarray, weights: np.ndarray | None = None,
                 ess_threshold: float = 0.5):
        self.particles = particles
        self.weights = weights if weights is not None else np.full(len(particles), 1.0 / len(particles))
        self.ess_threshold = ess_threshold

    def mean(self) -> np.ndarray:
        return (self.weights[:, None] * self.particles).sum(axis=0)

    def predictive_prob(self, env, item: int) -> float:
        logits = env.cfg.a * (self.particles @ env.V[item]) + env.cfg.b * env.q[item]
        p = 1.0 / (1.0 + np.exp(-logits))
        return float(self.weights @ p)

    def expected_reward(self, env, item: int, f_streak: int = 0) -> float:
        cos = np.clip(self.particles @ env.V[item], 0, None)
        p = 1.0 / (1.0 + np.exp(-(env.cfg.a * (self.particles @ env.V[item]) + env.cfg.b * env.q[item])))
        return float(self.weights @ (p * env.q[item] * cos)) * (1.0 - env.cfg.kappa_sat * f_streak)

    def update(self, env, item: int, engaged: bool) -> "ParticleBelief":
        p = 1.0 / (1.0 + np.exp(-(env.cfg.a * (self.particles @ env.V[item]) + env.cfg.b * env.q[item])))
        w = self.weights * np.where(engaged, p, 1.0 - p)
        w = w / w.sum()
        # push particles by the true deterministic dynamics
        pushed = np.array([env.push_theta(th, item, engaged) for th in self.particles])
        ess = 1.0 / np.sum(w ** 2)
        if ess < self.ess_threshold * len(w):
            idx = np.random.default_rng(0).choice(len(w), size=len(w), p=w)
            return ParticleBelief(pushed[idx], None, self.ess_threshold)
        return ParticleBelief(pushed, w, self.ess_threshold)

    def samples(self, n: int, rng: np.random.Generator) -> np.ndarray:
        idx = rng.choice(len(self.particles), size=n, p=self.weights)
        return self.particles[idx]
