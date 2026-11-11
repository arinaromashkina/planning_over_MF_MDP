"""Implicit-ALS training and user fold-in under the paper's all-items precision convention.

Paper eq. (2): for user history H,
    A(H) = lam*I + sum_{i in I} c_i v_i v_i^T,   u(H) = A(H)^{-1} sum_{i in H} c_i v_i,
with p_i = 1 on H, 0 otherwise and confidence c_i = 1 + alpha * x_i (x_i in {0, 1} the
thresholded positive indicator; unobserved items keep baseline c = 1, p = 0).

Because unobserved items all share c = 1, the dense sum factors:
    A(H) = lam*I + V'V + alpha * sum_{i in H} v_i v_i^T,
which is what we implement for both user solves and item solves during training.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from mf_mdp.mdp import MFState


@dataclass(frozen=True)
class IALSConfig:
    dim: int = 64
    alpha: float = 1.0  # confidence c = 1 + alpha * x
    lam: float = 1.0  # L2 regularizer
    n_iter: int = 10
    init_scale: float = 0.05


def _group_indices(pairs: np.ndarray, n_groups: int) -> list[np.ndarray]:
    """Group row indices of `pairs` (columns [group, other]) by group id."""
    order = np.argsort(pairs[:, 0], kind="stable")
    groups: list[np.ndarray] = [np.empty(0, dtype=np.int64) for _ in range(n_groups)]
    sorted_pairs = pairs[order]
    boundaries = np.flatnonzero(np.diff(sorted_pairs[:, 0])) + 1
    starts = np.concatenate(([0], boundaries))
    ends = np.concatenate((boundaries, [len(sorted_pairs)]))
    for start, end in zip(starts, ends):
        groups[int(sorted_pairs[start, 0])] = order[start:end]
    return groups


def train_ials(
    positives: np.ndarray,
    n_users: int,
    n_items: int,
    cfg: IALSConfig,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Train iALS user/item factors under the paper's convention.

    Args:
        positives: int array (n_pos, 2) of (user, item) pairs with x_ui = 1.
        n_users, n_items: catalog sizes.
        cfg: hyperparameters.
        seed: RNG seed for factor initialization.

    Returns:
        (U, V) with shapes (n_users, dim), (n_items, dim).
    """
    rng = np.random.default_rng(seed)
    U = rng.normal(0.0, cfg.init_scale, (n_users, cfg.dim))
    V = rng.normal(0.0, cfg.init_scale, (n_items, cfg.dim))
    eye = np.eye(cfg.dim)
    user_groups = _group_indices(positives, n_users)
    item_groups = _group_indices(positives[:, ::-1], n_items)

    for _ in range(cfg.n_iter):
        # Solve user factors given V.
        A0 = V.T @ V + cfg.lam * eye
        for u_idx, rows in enumerate(user_groups):
            items = positives[rows, 1]
            if len(items) == 0:
                U[u_idx] = 0.0
                continue
            Vh = V[items]
            A = A0 + cfg.alpha * (Vh.T @ Vh)
            b = (1.0 + cfg.alpha) * Vh.sum(axis=0)
            U[u_idx] = np.linalg.solve(A, b)
        # Solve item factors given U.
        A0i = U.T @ U + cfg.lam * eye
        for i_idx, rows in enumerate(item_groups):
            users = positives[rows, 0]  # column 0 = user (row positions are swap-invariant)
            if len(users) == 0:
                continue
            Uh = U[users]
            A = A0i + cfg.alpha * (Uh.T @ Uh)
            b = (1.0 + cfg.alpha) * Uh.sum(axis=0)
            V[i_idx] = np.linalg.solve(A, b)
    return U, V


def user_state(V: np.ndarray, items: np.ndarray, cfg: IALSConfig) -> MFState:
    """Fold-in: exact MFState for a user history (paper eq. (2))."""
    dim = V.shape[1]
    A = V.T @ V + cfg.lam * np.eye(dim)
    b = np.zeros(dim)
    if len(items) > 0:
        Vh = V[np.asarray(items)]
        A = A + cfg.alpha * (Vh.T @ Vh)
        b = (1.0 + cfg.alpha) * Vh.sum(axis=0)
    P = np.linalg.inv(A)
    u = P @ b
    return MFState(P=P, u=u, consumed=tuple(sorted(int(i) for i in np.atleast_1d(items))))
