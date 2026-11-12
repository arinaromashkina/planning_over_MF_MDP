"""V_logging: ridge regression on Monte-Carlo return-to-go from LOGGED sequences.

LABELING CONTRACT (docs/PHASE4_DESIGN.md §4.2): this estimates the value of the LOGGING
POLICY (the behavior that generated the data), conditional on the consumption support.
It is NOT V_target and must never be labeled as such.

Return definition (documented): per user, over the observed positive sequence
x_1..x_T (LLN prefix), states re-folded along the ACTUAL history:
    s_t = fold-in(x_1..x_{t-1}),   a_t = x_t,   r_t = r0(s_t, a_t)   (paper-exact R0)
    G_t = sum_{k=0}^{T-t-1} gamma^k * r_norm(s_{t+k}, a_{t+k})
where r_norm maps R0 into [0,1] via mdp.normalize_reward (so leaf values share the MCTS
scale). Truncation/censoring: sequences end at the observed LLN cut; no terminal flag
exists in ML-1M (documented limitation). gamma configurable.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from mf_mdp.data.movielens import UserSplit
from mf_mdp.mdp import MFState, fold_in, fold_in_general, normalize_reward, reward
from mf_mdp.mf import IALSConfig, user_state
from mf_mdp.models.features import FeatureEncoder, state_feature_vector


def state_value_features(state: MFState) -> np.ndarray:
    """V features: raw user embedding u + state summary. Decision-point information only."""
    return np.concatenate([state.u, state_feature_vector(state)])


def build_logged_trajectories(
    splits: list[UserSplit],
    V: np.ndarray,
    ials_cfg: IALSConfig,
    eta: float,
    gamma: float,
    max_t: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (X_state, G): one row per (user, t) decision point on the ACTUAL history.

    States are built incrementally with the exact weighted transition
    (alpha_w=alpha, beta_w=1+alpha — the true re-fold per step, tested against direct
    re-folding). r_t is the paper-exact R0 reward of the realized (state, action),
    normalized to [0,1]. With max_t, only the LAST max_t decision points per user are
    kept (recent states match MCTS leaf distribution better).
    """
    rows_X, rows_G = [], []
    for sp in splits:
        seq = sp.prefix
        T = len(seq)
        if T < 2:
            continue
        # incremental exact states along the true history
        st = user_state(V, np.empty(0, dtype=np.int64), ials_cfg)
        rs = np.empty(T)
        states: list[MFState] = []
        for t in range(T):
            a = int(seq[t])
            v = V[a]
            _, _, delta_s = fold_in(st, v)
            rs[t] = normalize_reward(reward(st.u, v, delta_s, eta, "cos"), eta)
            states.append(st)
            P_p, u_p, _ = fold_in_general(st, v, ials_cfg.alpha, 1.0 + ials_cfg.alpha)
            st = MFState(P=P_p, u=u_p, consumed=tuple(sorted(st.consumed + (a,))))
        t0 = max(0, T - max_t) if max_t is not None else 0
        for t in range(t0, T):
            G = float(np.sum(gamma ** np.arange(T - t) * rs[t:]))
            rows_X.append(state_value_features(states[t]))
            rows_G.append(G)
    return np.vstack(rows_X), np.asarray(rows_G)


@dataclass
class VLoggingRidge:
    """Ridge estimator of logging-policy return-to-go. Never V_target."""

    lam: float = 1.0
    encoder: FeatureEncoder = field(default_factory=FeatureEncoder)
    weights: np.ndarray | None = field(default=None, init=False)

    def fit(self, X: np.ndarray, G: np.ndarray) -> None:
        Xn = self.encoder.fit_transform(X)
        n, d = Xn.shape
        Xb = np.hstack([Xn, np.ones((n, 1))])
        reg = self.lam * np.eye(d + 1)
        reg[d, d] = 0.0
        self.weights = np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ G)

    def predict_value(self, state: MFState) -> float:
        """Predicted normalized logging-policy return-to-go from `state`."""
        if self.weights is None:
            raise RuntimeError("model not fitted")
        x = self.encoder.transform(state_value_features(state)[None, :])
        return float((np.hstack([x, [[1.0]]]) @ self.weights).item())

    def predict_batch(self, X: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise RuntimeError("model not fitted")
        x = self.encoder.transform(X)
        return np.hstack([x, np.ones((x.shape[0], 1))]) @ self.weights
