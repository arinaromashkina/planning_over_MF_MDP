"""Env-C: continuous MF-flavored controlled environment (Phase 5, Experiment A on the
MF-MDP setting). True latent-preference dynamics + iALS posterior states + hybrid
learned-stochastic/analytic planning.

Generative model (docs/PHASE5_EXPERIMENT_PLAN.md):
- True user state theta_t in R^d; items v_i in R^d with quality q_i.
- Engagement: y ~ Bernoulli(sigmoid(a * cos(theta, v_i) + b * q_i + eps_t)),
  eps_t ~ N(0, sigma_obs^2) is per-step latent noise (response is not fully determined
  by observables -> the response model has irreducible error).
- True transition: theta_{t+1} = normalize((1-lam) theta_t + alpha(y) v_i + lam theta_0),
  alpha(engage) = +alpha_pos, alpha(skip) = alpha_neg (< 0).
- Reward on engage: r = q_i * cos(theta, v_i) * (1 - kappa_sat * f); skip -> 0.
- Termination: h = sigmoid(c0 - c1 * cos(theta, v_i)) on every step.
- Horizon H, discount gamma.

The LEARNED model uses iALS posterior states (P, u) trained on logged trajectories,
a learned response model p(y | decision features), and exact weighted fold-in
transitions (engage: (alpha, 1+alpha); skip: (alpha_skip, 0)) — documented approximation
of the true theta-dynamics through the posterior.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EnvCConfig:
    d: int = 16
    n_items: int = 30
    H: int = 6
    gamma: float = 0.9
    a: float = 6.0
    b: float = 2.0
    sigma_obs: float = 0.5  # latent response noise (irreducible model error)
    lam: float = 0.1  # drift toward theta_0
    alpha_pos: float = 0.35
    alpha_neg: float = -0.2
    kappa_sat: float = 0.3
    c0: float = -2.0
    c1: float = 3.0
    seed: int = 0


class EnvC:
    def __init__(self, cfg: EnvCConfig = EnvCConfig()):
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        # item embeddings on the sphere; qualities spread out
        V = rng.normal(0, 1, (cfg.n_items, cfg.d))
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        self.V = V
        self.q = rng.uniform(0.2, 1.0, cfg.n_items)
        self.n_items = cfg.n_items
        # a few "cluster centers" so assimilation steering matters
        self.theta0 = np.zeros(cfg.d)

    def engagement_prob(self, theta: np.ndarray, i: int, eps: float = 0.0) -> float:
        cos = float(theta @ self.V[i])
        return float(1.0 / (1.0 + np.exp(-(self.cfg.a * cos + self.cfg.b * self.q[i] + eps))))

    def reward(self, theta: np.ndarray, f: int, i: int, engaged: bool) -> float:
        if not engaged:
            return 0.0
        cos = float(theta @ self.V[i])
        return float(self.q[i] * cos * (1.0 - self.cfg.kappa_sat * f))

    def termination_prob(self, theta: np.ndarray, i: int) -> float:
        cos = float(theta @ self.V[i])
        return float(1.0 / (1.0 + np.exp(-(self.cfg.c0 - self.cfg.c1 * cos))))

    def step(self, theta: np.ndarray, f: int, i: int, rng: np.random.Generator):
        eps = rng.normal(0.0, self.cfg.sigma_obs)
        engaged = rng.random() < self.engagement_prob(theta, i, eps)
        r = self.reward(theta, f, i, engaged)
        term = rng.random() < self.termination_prob(theta, i)
        c = self.cfg
        alpha = c.alpha_pos if engaged else c.alpha_neg
        new_theta = (1.0 - c.lam) * theta + alpha * self.V[i] + c.lam * self.theta0
        nrm = np.linalg.norm(new_theta)
        new_theta = new_theta / nrm if nrm > 0 else theta
        nf = min(2, f + 1) if engaged else f
        return r, term, new_theta, nf, engaged

    def rollout(self, policy, theta0: np.ndarray, rng: np.random.Generator) -> float:
        total, theta, f = 0.0, theta0.copy(), 0
        for t in range(self.cfg.H):
            i = policy(theta, f, t)
            r, term, theta, nf, _ = self.step(theta, f, i, rng)
            total += (self.cfg.gamma ** t) * r
            if term:
                break
            f = nf
        return total


def gen_theta0s(n_users: int, env: EnvC, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.normal(0, 1, (n_users, env.cfg.d)) / np.sqrt(env.cfg.d)
