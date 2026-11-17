"""H1 environment — one generative model, three regimes (design doc §2.1).

Regimes (flags of ONE config, not three architectures):
  Env-L (Learning-only):  alpha = 0, lam = 0, prior over theta_0 (sigma_0 > 0)
  Env-P (Preference-shaping-only): alpha > 0, point-mass belief at true theta_0
  Env-J (Joint):          alpha > 0, prior over theta_0 (both processes active)

Latent state theta_t on the sphere; assimilation push theta' =
normalize((1-lam) theta + alpha(y) v + lam * theta0); response
y ~ Bernoulli(sigmoid(a cos + b q + eps)); reward on engagement;
termination hazard on mismatch. Catalog has an off-mode high-quality cluster so
that myopic play and steering-optimal play differ (steering tension; validated in
the E1 pilot like Phase-5 A1).

The planner-facing API is belief-only (see belief/core.py); env.theta never enters
the policy path (enforced by tests/test_h1.py).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class H1EnvConfig:
    regime: str = "P"          # 'L' | 'P' | 'J'
    d: int = 16
    n_items: int = 30
    H: int = 8
    gamma: float = 0.9
    a: float = 6.0             # response sharpness on match
    b: float = 2.0             # response bias from quality
    sigma_obs: float = 0.3     # latent response logit noise
    lam: float = 0.05          # drift to anchor
    alpha_pos: float = 0.25    # preference push on engagement (0 for Env-L)
    alpha_neg: float = -0.1    # push on skip (<= 0)
    sigma_0: float = 0.3       # initial belief std (0 = point mass at theta_0)
    n_cluster: int = 4         # off-mode high-quality cluster size
    cluster_gap: float = 0.7   # reward premium of the cluster
    kappa_sat: float = 0.0     # fatigue disabled by default (H1 isolates the push)
    c0: float = -2.0
    c1: float = 3.0
    seed: int = 0
    ials_alpha: float = 10.0   # fold-in confidence scale (belief updates)


class H1Env:
    def __init__(self, cfg: H1EnvConfig = H1EnvConfig()):
        assert cfg.regime in ("L", "P", "J")
        self.cfg = cfg
        rng = np.random.default_rng(cfg.seed)
        self.d = cfg.d
        # catalog: base items spread on the sphere + a high-quality cluster at `w`
        V = rng.normal(0, 1, (cfg.n_items, cfg.d))
        V /= np.linalg.norm(V, axis=1, keepdims=True)
        q = rng.uniform(0.2, 0.6, cfg.n_items)
        w = np.zeros(cfg.d); w[0] = 1.0
        cluster = np.arange(cfg.n_cluster)
        for i in cluster:
            V[i] = w + 0.15 * rng.normal(0, 1, cfg.d)
            V[i] /= np.linalg.norm(V[i])
        q[cluster] = cfg.cluster_gap + rng.uniform(0, 0.2, cfg.n_cluster)
        self.V = V
        self.q = q
        self.cluster = cluster
        # user anchor direction: orthogonal to the cluster direction w
        e1 = np.zeros(cfg.d); e1[1] = 1.0
        self.anchor_dir = e1
        self.n_items = cfg.n_items

    # ---- true dynamics ----------------------------------------------------

    def sample_theta0(self, rng: np.random.Generator) -> np.ndarray:
        th = self.anchor_dir + 0.3 * rng.normal(0, 1, self.d)
        th[0] += 0.3 * rng.normal(0, 1)  # small cluster-ward component
        return th / np.linalg.norm(th)

    def alpha_of(self, engaged: bool) -> float:
        return self.cfg.alpha_pos if engaged else self.cfg.alpha_neg

    def engagement_prob(self, theta: np.ndarray, i: int, eps: float = 0.0) -> float:
        return float(1.0 / (1.0 + np.exp(-(self.cfg.a * float(theta @ self.V[i]) + self.cfg.b * self.q[i] + eps))))

    def reward(self, theta: np.ndarray, f: int, i: int, engaged: bool) -> float:
        if not engaged:
            return 0.0
        return float(self.q[i] * float(theta @ self.V[i]) * (1.0 - self.cfg.kappa_sat * f))

    def termination_prob(self, theta: np.ndarray, i: int) -> float:
        return float(1.0 / (1.0 + np.exp(-(self.cfg.c0 - self.cfg.c1 * float(theta @ self.V[i])))))

    def push_theta(self, theta: np.ndarray, i: int, engaged: bool, theta0: np.ndarray | None = None) -> np.ndarray:
        c = self.cfg
        anchor = self.anchor_dir if theta0 is None else theta0
        alpha = self.alpha_of(engaged) if c.regime != "L" else 0.0
        new = (1.0 - c.lam) * theta + alpha * self.V[i] + c.lam * anchor
        nrm = np.linalg.norm(new)
        return new / nrm if nrm > 0 else theta

    def step(self, theta: np.ndarray, f: int, i: int, rng: np.random.Generator, theta0: np.ndarray):
        eps = rng.normal(0.0, self.cfg.sigma_obs)
        engaged = rng.random() < self.engagement_prob(theta, i, eps)
        r = self.reward(theta, f, i, engaged)
        term = rng.random() < self.termination_prob(theta, i)
        ntheta = self.push_theta(theta, i, engaged, theta0)
        nf = min(2, f + 1) if engaged else f
        return r, term, ntheta, nf, engaged

    # ---- belief-side helpers (planner-facing, no theta) --------------------

    def fold_in_weights(self, engaged: bool) -> tuple[float, float]:
        """Phase-5 T1 convention: engage -> (alpha, 1+alpha); skip -> (-alpha/2, 0)."""
        a = self.cfg.ials_alpha
        return (a, 1.0 + a) if engaged else (-0.5 * a, 0.0)

    def initial_belief(self, theta0: np.ndarray, rng: np.random.Generator):
        """Initial belief per regime. MF beliefs reuse the fold-in parameterization."""
        from mf_mdp.belief import JointMFBelief, MFPosteriorBelief, ParticleBelief

        c = self.cfg
        if c.regime == "P":
            return MFPosteriorBelief(theta0.copy(), np.zeros((c.d, c.d)))
        m0 = theta0 + c.sigma_0 * rng.normal(0, 1, c.d)
        m0 /= np.linalg.norm(m0)
        P0 = (c.sigma_0 ** 2 + 1e-6) * np.eye(c.d)
        return MFPosteriorBelief(m0, P0)

    def initial_belief_joint(self, theta0: np.ndarray, rng: np.random.Generator):
        from mf_mdp.belief import JointMFBelief, MFPosteriorBelief, ParticleBelief

        c = self.cfg
        if c.regime == "P":
            return JointMFBelief(theta0.copy(), np.zeros((c.d, c.d)), c.lam, theta0.copy())
        m0 = theta0 + c.sigma_0 * rng.normal(0, 1, c.d)
        m0 /= np.linalg.norm(m0)
        P0 = (c.sigma_0 ** 2 + 1e-6) * np.eye(c.d)
        return JointMFBelief(m0, P0, c.lam, m0.copy())

    def initial_particles(self, theta0: np.ndarray, rng: np.random.Generator, n: int = 400):
        from mf_mdp.belief import ParticleBelief

        if self.cfg.regime == "P":
            parts = np.tile(theta0, (n, 1))
        else:
            parts = theta0 + self.cfg.sigma_0 * rng.normal(0, 1, (n, self.d))
            parts /= np.linalg.norm(parts, axis=1, keepdims=True)
        return ParticleBelief(parts)

    # ---- evaluation ---------------------------------------------------------

    def rollout(self, policy, theta0: np.ndarray, rng: np.random.Generator) -> float:
        """Roll out a POLICY (belief-facing or oracle) in the true environment."""
        total, theta, f = 0.0, theta0.copy(), 0
        state = policy.reset(theta0, rng)
        for t in range(self.cfg.H):
            i = policy.act(state, f, t, rng)
            r, term, theta, nf, engaged = self.step(theta, f, i, rng, theta0)
            total += (self.cfg.gamma ** t) * r
            if term:
                break
            state = policy.update(state, i, engaged)
            f = nf
        return total


def make_L(seed: int = 0) -> H1Env:
    return H1Env(H1EnvConfig(regime="L", alpha_pos=0.0, alpha_neg=0.0, lam=0.0, sigma_0=0.5, seed=seed))


def make_P(seed: int = 0) -> H1Env:
    return H1Env(H1EnvConfig(regime="P", alpha_pos=0.25, alpha_neg=-0.1, lam=0.05, sigma_0=0.0, seed=seed))


def make_J(seed: int = 0) -> H1Env:
    return H1Env(H1EnvConfig(regime="J", alpha_pos=0.25, alpha_neg=-0.1, lam=0.05, sigma_0=0.5, seed=seed))
