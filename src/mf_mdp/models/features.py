"""Decision-time features for response/reward/value models.

Leakage contract: every feature is computable from information available at the decision
point — the fold-in state (P, u) built from the user's past interactions only, and the
candidate item's fixed embedding / train-time popularity. No future interactions, no
held-out fields.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mf_mdp.mdp import MFState, fold_in


def state_feature_vector(state: MFState) -> np.ndarray:
    """User-state summary features: [hist_len, tr(P), ||P||_F, ||u||]."""
    P = state.P
    return np.array([
        float(len(state.consumed)),
        float(np.trace(P)),
        float(np.linalg.norm(P)),
        float(np.linalg.norm(state.u)),
    ])


def decision_features(
    state: MFState,
    v: np.ndarray,
    leverage: float,
    delta_s: float,
    log_popularity: float,
) -> np.ndarray:
    """Per-(state, item) features: [cos, dot, leverage, |delta_s|, ||v||, log_pop]."""
    u = state.u
    un = np.linalg.norm(u)
    vn = np.linalg.norm(v)
    cos = 0.0 if un == 0 or vn == 0 else float(u @ v) / (un * vn)
    return np.array([cos, float(u @ v), float(leverage), abs(float(delta_s)), float(vn), float(log_popularity)])


N_STATE = 4
N_DECISION = 6


@dataclass
class FeatureEncoder:
    """Assembles feature vectors and fits feature normalization on training data only."""

    mean: np.ndarray | None = None
    std: np.ndarray | None = None

    def fit(self, X: np.ndarray) -> None:
        self.mean = X.mean(axis=0)
        self.std = np.where(X.std(axis=0) == 0.0, 1.0, X.std(axis=0))

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None or self.std is None:
            raise RuntimeError("FeatureEncoder not fitted")
        return (X - self.mean) / self.std

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)


def features_for(state: MFState, V: np.ndarray, item: int, log_pop: np.ndarray) -> np.ndarray:
    """Full feature vector for one (state, item): state summary + decision features."""
    v = V[item]
    _, _, delta_s = fold_in(state, v)
    leverage = float(v @ (state.P @ v))
    return np.concatenate([state_feature_vector(state), decision_features(state, v, leverage, delta_s, log_pop[item])])
