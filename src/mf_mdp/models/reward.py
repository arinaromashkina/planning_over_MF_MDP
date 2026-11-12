"""Reward variants R1 (response distribution + fixed utility) and R2 (direct learned
expected utility). R0 is the paper heuristic in mf_mdp.mdp.reward.

Labeling contract (docs/PHASE4_DESIGN.md §4.1): these objects estimate EXPECTED UTILITY
of feedback under the logging policy's consumption support. They are not exposure models
and not satisfaction ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from mf_mdp.mdp import MFState, fold_in
from mf_mdp.models.features import FeatureEncoder, features_for


@dataclass(frozen=True)
class UtilityMapping:
    """Transparent fixed mapping from observed feedback to utility. Chosen a priori."""

    u_rating5: float = 1.0
    u_rating4: float = 0.25  # configurable; sensitivity reported

    def utility(self, rating: int) -> float:
        if rating >= 5:
            return self.u_rating5
        if rating >= 4:
            return self.u_rating4
        return 0.0


class R1ExpectedReward:
    """R1: expected utility under the response distribution.

    r_hat(s,a) = p_phi(y=5|s,a) * U(5) + p_phi(y=4|s,a) * U(4)   [p(y=4)=1-p(y=5)]
    Optionally plus the paper alignment term eta/(1+|Delta_s(a)|) (ablated via `eta`).
    """

    def __init__(self, response: Callable[[np.ndarray], np.ndarray], utility: UtilityMapping,
                 encoder: FeatureEncoder, eta: float = 0.0):
        self.response = response  # features -> P(y=5)
        self.utility = utility
        self.encoder = encoder
        self.eta = eta

    def predict_reward(self, state: MFState, V: np.ndarray, item: int, log_pop: np.ndarray) -> float:
        x = self.encoder.transform(features_for(state, V, item, log_pop)[None, :])
        p5 = float(self.response(x)[0])
        u = p5 * self.utility.u_rating5 + (1.0 - p5) * self.utility.u_rating4
        if self.eta > 0.0:
            v = V[item]
            _, _, delta_s = fold_in(state, v)
            u += self.eta / (1.0 + abs(delta_s))
        return float(u)


class R2DirectExpectedReward:
    """R2: direct ridge regression of observed utility on decision-time features."""

    def __init__(self, weights: np.ndarray, encoder: FeatureEncoder, eta: float = 0.0):
        self.weights = weights
        self.encoder = encoder
        self.eta = eta

    def predict_reward(self, state: MFState, V: np.ndarray, item: int, log_pop: np.ndarray) -> float:
        x = self.encoder.transform(features_for(state, V, item, log_pop)[None, :])
        u = float((np.hstack([x, [[1.0]]]) @ self.weights).item())
        if self.eta > 0.0:
            v = V[item]
            _, _, delta_s = fold_in(state, v)
            u += self.eta / (1.0 + abs(delta_s))
        return float(u)


def fit_ridge(X: np.ndarray, y: np.ndarray, lam: float) -> np.ndarray:
    """Closed-form ridge with intercept (unpenalized)."""
    n, d = X.shape
    Xb = np.hstack([X, np.ones((n, 1))])
    reg = lam * np.eye(d + 1)
    reg[d, d] = 0.0
    return np.linalg.solve(Xb.T @ Xb + reg, Xb.T @ y)


def ridge_predict(X: np.ndarray, w: np.ndarray) -> np.ndarray:
    Xb = np.hstack([X, np.ones((X.shape[0], 1))])
    return Xb @ w
