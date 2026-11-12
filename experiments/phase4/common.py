"""Shared builders for Phase 4 experiments (response/value datasets)."""

from __future__ import annotations

import numpy as np

from mf_mdp.data.movielens import Dataset, UserSplit
from mf_mdp.mdp import MFState, fold_in, fold_in_general
from mf_mdp.mf import IALSConfig, user_state
from mf_mdp.models.features import features_for


def build_decision_dataset(
    data: Dataset,
    splits: list[UserSplit],
    V: np.ndarray,
    ials_cfg: IALSConfig,
    max_t: int = 40,
):
    """Per (user, t) decision-point examples on the ACTUAL history.

    Returns (X, y5, u, t) where X = features_for(state_t, item_t), y5 = 1 if the
    consumed rating is 5 else 0, u = user id, t = position in the sequence.
    States are built incrementally with the exact weighted transition.
    Decision-time information only (leakage-safe by construction).
    """
    log_pop = np.log1p(np.bincount(data.positives_train[:, 1], minlength=data.n_items).astype(float))
    X_rows, y_rows, u_rows, t_rows = [], [], [], []
    for sp in splits:
        seq, rat = sp.prefix, sp.prefix_ratings
        T = len(seq)
        if T < 3 or rat is None:
            continue
        st = user_state(V, np.empty(0, dtype=np.int64), ials_cfg)
        t0 = max(1, T - max_t)
        for t in range(1, T):
            a = int(seq[t])
            v = V[a]
            if t >= t0:
                X_rows.append(features_for(st, V, a, log_pop))
                y_rows.append(1.0 if rat[t] >= 5 else 0.0)
                u_rows.append(sp.user)
                t_rows.append(t)
            P_p, u_p, _ = fold_in_general(st, v, ials_cfg.alpha, 1.0 + ials_cfg.alpha)
            st = MFState(P=P_p, u=u_p, consumed=tuple(sorted(st.consumed + (a,))))
    return np.vstack(X_rows), np.asarray(y_rows), np.asarray(u_rows), np.asarray(t_rows)


def split_users(users: np.ndarray, seed: int = 0, fracs=(0.7, 0.15, 0.15)):
    """User-level disjoint split (train/valid/test)."""
    rng = np.random.default_rng(seed)
    u = np.unique(users)
    rng.shuffle(u)
    n = len(u)
    n_tr = int(fracs[0] * n)
    n_va = int(fracs[1] * n)
    return u[:n_tr], u[n_tr:n_tr + n_va], u[n_tr + n_va:]
