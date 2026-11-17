"""Depth-2 stochastic/optimistic lookahead over the H2 factorial states.

Arms (docs/H2_HYBRID_SEQUENTIAL_STATE_DESIGN.md):
    state_arm   mf     -> s = (P, u)            iALS fold-in posterior
                seq    -> s = h                 GRU sequential state
                hybrid -> s = (P, u, h)         both
    transition  optimistic -> engage branch only, zero hazard (paper T0 analog)
                stochastic -> branch over y ~ p_hat(y|s,a) with (1 - h_hat) continuation
                              (Phase-5 Bellman form)

MF stochastic transitions reuse the Phase-5 T1 weighted fold-ins:
engage (alpha_w, beta_w) = (alpha, 1 + alpha), skip (-0.5*alpha, 0).
Seq/hybrid transitions step the GRU with the branch's y (feedback-conditioned g).

The planner sees only the state; env internals (theta, true q) never enter here
(q_hat is estimated from logs by the caller and shared across arms).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mf_mdp.mdp import MFState, fold_in_general
from mf_mdp.models.features import FeatureEncoder, features_for
from mf_mdp.models.response import LogisticResponseModel
from mf_mdp.sequential.world import SeqWorldModel


@dataclass(frozen=True)
class HybridPlannerConfig:
    state_arm: str = "hybrid"  # mf | seq | hybrid
    transition: str = "stochastic"  # optimistic | stochastic
    pool: int = 8
    gamma: float = 0.9


@dataclass
class HybridState:
    P: np.ndarray | None
    u: np.ndarray | None
    h: np.ndarray | None
    consumed: tuple[int, ...]


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(a @ b) / (na * nb)


class HybridPlanner:
    """One planner code path for all six arms (matched budget by construction)."""

    def __init__(
        self,
        V: np.ndarray,
        ials_cfg,
        world: SeqWorldModel | None,
        mf_resp: LogisticResponseModel | None,
        mf_hazard: LogisticResponseModel | None,
        enc: FeatureEncoder | None,
        log_pop: np.ndarray,
        q_hat: np.ndarray,
        kappa: float,
        cfg: HybridPlannerConfig,
    ):
        if cfg.state_arm != "mf" and world is None:
            raise ValueError("seq/hybrid arms require a SeqWorldModel")
        if cfg.state_arm == "mf" and (mf_resp is None or enc is None):
            raise ValueError("mf arm requires mf_resp + enc")
        self.V, self.ials_cfg, self.world = V, ials_cfg, world
        self.mf_resp, self.mf_hazard, self.enc = mf_resp, mf_hazard, enc
        self.log_pop, self.q_hat, self.kappa, self.cfg = log_pop, q_hat, kappa, cfg
        self.n_items = V.shape[0]

    # ---------- state views ----------

    def _E(self) -> np.ndarray:
        return self.world.E[: self.n_items]

    def user_vec(self, st: HybridState) -> np.ndarray:
        arm = self.cfg.state_arm
        if arm == "mf":
            return st.u
        uh = self.world.readout(st.h)
        if arm == "seq":
            return uh
        u_mf = st.u / (np.linalg.norm(st.u) + 1e-12)
        u_s = uh / (np.linalg.norm(uh) + 1e-12)
        return np.concatenate([u_mf, u_s])

    def _emb(self, a: int) -> np.ndarray:
        arm = self.cfg.state_arm
        if arm == "mf":
            return self.V[a]
        e = self._E()[a]
        if arm == "seq":
            return e
        v = self.V[a] / (np.linalg.norm(self.V[a]) + 1e-12)
        e = e / (np.linalg.norm(e) + 1e-12)
        return np.concatenate([v, e])

    def relevance(self, st: HybridState, a: int) -> float:
        return _cos(self.user_vec(st), self._emb(a))

    # ---------- learned heads per arm ----------

    def p_resp(self, st: HybridState, a: int) -> float:
        if self.cfg.state_arm == "mf":
            x = self.enc.transform(features_for(
                MFState(P=st.P, u=st.u, consumed=st.consumed),
                self.V, a, self.log_pop)[None, :])
            return float(self.mf_resp.predict_proba(x)[0])
        return self.world.response_prob(st.h, a)

    def p_term(self, st: HybridState, a: int) -> float:
        if self.cfg.state_arm == "mf":
            if self.mf_hazard is None:
                return 0.0
            x = self.enc.transform(features_for(
                MFState(P=st.P, u=st.u, consumed=st.consumed),
                self.V, a, self.log_pop)[None, :])
            return float(self.mf_hazard.predict_proba(x)[0])
        return self.world.hazard_prob(st.h, a)

    def _p(self, st: HybridState, a: int) -> float:
        """Branch/expected-response prob; optimistic arms replace it by 1 (paper T0 analog)."""
        if self.cfg.transition == "optimistic":
            return 1.0
        return self.p_resp(st, a)

    # ---------- transitions ----------

    def _transition(self, st: HybridState, a: int, engaged: bool) -> HybridState:
        P, u, h = st.P, st.u, st.h
        if self.cfg.state_arm in ("mf", "hybrid"):
            alpha = self.ials_cfg.alpha
            aw, bw = (alpha, 1.0 + alpha) if engaged else (-0.5 * alpha, 0.0)
            P2, u2, _ = fold_in_general(MFState(P=P, u=u, consumed=st.consumed),
                                        self.V[a], aw, bw)
            P, u = P2, u2
        if self.cfg.state_arm in ("seq", "hybrid"):
            h = self.world.step_state(st.h, a, 1.0 if engaged else 0.0)
        return HybridState(P=P, u=u, h=h,
                           consumed=tuple(sorted(st.consumed + (a,))))

    # ---------- reward (shared semantics across arms) ----------

    def _r_if_engage(self, st: HybridState, a: int, f: int) -> float:
        return float(self.q_hat[a] * max(self.relevance(st, a), 0.0)
                     * (1.0 - self.kappa * f))

    def _expected_r(self, st: HybridState, a: int, f: int) -> float:
        return self._p(st, a) * self._r_if_engage(st, a, f)

    def _pool(self, st: HybridState) -> np.ndarray:
        avail = np.array([i for i in range(self.n_items) if i not in st.consumed])
        if len(avail) == 0:
            avail = np.arange(self.n_items)
        scores = np.array([self.relevance(st, int(a)) for a in avail])
        if scores.max() <= 1e-9:  # cold start: popularity-quality fallback (documented)
            scores = self.q_hat[avail]
        k = min(self.cfg.pool, len(avail))
        part = np.argpartition(-scores, k - 1)[:k]
        return avail[part[np.argsort(-scores[part], kind="stable")]]

    # ---------- depth-2 lookahead ----------

    def plan(self, st: HybridState, f: int, rng: np.random.Generator | None = None) -> int:
        optimistic = self.cfg.transition == "optimistic"
        pool = self._pool(st)
        best_v, best_a = -np.inf, int(pool[0])
        for a in pool:
            a = int(a)
            p = self._p(st, a)
            hterm = 0.0 if optimistic else self.p_term(st, a)
            q_val = 0.0
            for engaged, prob in ((True, p), (False, 1.0 - p)):
                if prob <= 0.0:
                    continue
                r1 = self._r_if_engage(st, a, f) if engaged else 0.0
                st1 = self._transition(st, a, engaged)
                f2 = min(2, f + 1) if engaged else f
                pool2 = self._pool(st1)
                leaf = max(self._expected_r(st1, int(b), f2) for b in pool2)
                q_val += prob * (r1 + (1.0 - hterm) * self.cfg.gamma * leaf)
            if q_val > best_v:
                best_v, best_a = q_val, a
        return best_a
