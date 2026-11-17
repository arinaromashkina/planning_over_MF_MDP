"""CrossPlanner — HybridPlanner with the mf response-feature path extended.

The H2 `HybridPlanner` computes the mf-arm response/hazard features via
`features_for` (10 base features). `CrossPlanner` keeps EVERYTHING else
(transitions, pool, depth-2 lookahead, reward semantics) identical and only
replaces the response-model input vector:

    mode "base"     : base10            -> exact parity with HybridPlanner (test-locked)
    mode "baseh"    : base10 + HIST
    mode "cross"    : base10 + HIST + grounded Env-C crosses (the hypothesis)
    mode "randcross": base10 + HIST + RandomCrosses of matched count (control C1)
    mode "uv"       : base10 + HIST + u_odot_v dims (interaction #5, MLP only)

The hazard model intentionally stays on base10 features in every mode — the
hypothesis is about the response model only, so the shared hazard must not move.
Callers must pass `events` (list of (item, engaged)) to plan()/p_resp() so the
history features are decision-time legal; events=None is allowed and yields
mean_eng_hist=0 (cold start), which is what the parity test uses.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mf_mdp.interactions.cross import N_BASE, RandomCrosses, envc_crosses
from mf_mdp.mdp import MFState
from mf_mdp.models.features import FeatureEncoder, features_for
from mf_mdp.planning.hybrid import HybridPlanner, HybridPlannerConfig, HybridState

FEATURE_MODES = ("base", "baseh", "cross", "randcross", "uv")


@dataclass(frozen=True)
class CrossPlannerConfig(HybridPlannerConfig):
    feature_mode: str = "cross"  # one of FEATURE_MODES

    def __post_init__(self):
        if self.feature_mode not in FEATURE_MODES:
            raise ValueError(f"unknown feature_mode {self.feature_mode}")


class CrossPlanner(HybridPlanner):
    def __init__(self, *args, cfg: CrossPlannerConfig, enc_resp: FeatureEncoder,
                 enc_haz: FeatureEncoder, rand_crosses: RandomCrosses | None = None,
                 **kwargs):
        super().__init__(*args, cfg=cfg, **kwargs)
        if cfg.feature_mode == "randcross" and rand_crosses is None:
            raise ValueError("randcross mode requires fitted RandomCrosses")
        self.cfg: CrossPlannerConfig
        self.enc_resp, self.enc_haz = enc_resp, enc_haz
        self.rand_crosses = rand_crosses

    # ---------- feature assembly ----------

    @staticmethod
    def _hist(state: HybridState, a: int, events) -> np.ndarray:
        mean_eng = float(np.mean([e for _, e in events])) if events else 0.0
        repeat_cnt = float(state.consumed.count(a))
        return np.array([mean_eng, repeat_cnt])

    def features(self, st: HybridState, a: int, events=None) -> np.ndarray:
        mode = self.cfg.feature_mode
        base = features_for(
            MFState(P=st.P, u=st.u, consumed=st.consumed), self.V, a, self.log_pop)
        if mode == "base":
            return base
        hist = self._hist(st, a, events)
        bh = np.concatenate([base, hist])
        if mode == "baseh":
            return bh
        if mode == "cross":
            return np.concatenate([bh, envc_crosses(base, hist[0], hist[1])])
        if mode == "randcross":
            return np.concatenate([bh, self.rand_crosses.transform(bh[None, :])[0]])
        # "uv": raw elementwise product u ⊙ v (not aggregated — dot already is)
        return np.concatenate([bh, st.u * self.V[a]])

    # ---------- learned heads (response input replaced; hazard fixed on base) ----------

    def p_resp(self, st: HybridState, a: int, events=None) -> float:
        if self.cfg.state_arm != "mf":
            return super().p_resp(st, a)
        x = self.features(st, a, events)
        return float(self.mf_resp.predict_proba(self.enc_resp.transform(x[None, :]))[0])

    def p_term(self, st: HybridState, a: int, events=None) -> float:
        if self.cfg.state_arm != "mf" or self.mf_hazard is None:
            return super().p_term(st, a)
        base = features_for(
            MFState(P=st.P, u=st.u, consumed=st.consumed), self.V, a, self.log_pop)
        return float(self.mf_hazard.predict_proba(self.enc_haz.transform(base[None, :]))[0])

    # ---------- lookahead (same as parent, but events flow into p_resp) ----------

    def _expected_r_ev(self, st: HybridState, a: int, f: int, events) -> float:
        p = 1.0 if self.cfg.transition == "optimistic" else self.p_resp(st, a, events)
        return p * self._r_if_engage(st, a, f)

    def plan(self, st: HybridState, f: int, rng: np.random.Generator | None = None,
             events=None) -> int:
        if self.cfg.feature_mode == "base":
            # exact HybridPlanner code path (parity locked by tests)
            return super().plan(st, f, rng)
        optimistic = self.cfg.transition == "optimistic"
        pool = self._pool(st)
        best_v, best_a = -np.inf, int(pool[0])
        for a in pool:
            a = int(a)
            p = 1.0 if optimistic else self.p_resp(st, a, events)
            hterm = 0.0 if optimistic else self.p_term(st, a, events)
            q_val = 0.0
            for engaged, prob in ((True, p), (False, 1.0 - p)):
                if prob <= 0.0:
                    continue
                r1 = self._r_if_engage(st, a, f) if engaged else 0.0
                st1 = self._transition(st, a, engaged)
                events1 = (list(events) if events else []) + [(a, float(engaged))]
                f2 = min(2, f + 1) if engaged else f
                pool2 = self._pool(st1)
                leaf = max(self._expected_r_ev(st1, int(b), f2, events1) for b in pool2)
                q_val += prob * (r1 + (1.0 - hterm) * self.cfg.gamma * leaf)
            if q_val > best_v:
                best_v, best_a = q_val, a
        return best_a
