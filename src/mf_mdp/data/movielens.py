"""MovieLens-1M download, parsing, and chronological splits (LLN / GTS).

Positive feedback: rating >= min_rating (paper: thresholded at 4 stars). Histories are
the positive interactions sorted by timestamp; lower ratings are ignored entirely.
"""

from __future__ import annotations

import hashlib
import io
import os
import urllib.request
import zipfile
from dataclasses import dataclass

import numpy as np

ML1M_URL = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
ML1M_SHA1 = None  # filled in on first download and recorded in EXPERIMENT_LOG


@dataclass(frozen=True)
class UserSplit:
    """Chronological per-user split of positive interactions."""

    user: int
    prefix: np.ndarray  # item ids (positives) used to build the initial posterior
    valid: np.ndarray  # held-out validation positives
    test: np.ndarray  # held-out test positives
    prefix_ratings: np.ndarray | None = None  # ratings aligned with `prefix` (optional)
    valid_ratings: np.ndarray | None = None
    test_ratings: np.ndarray | None = None


@dataclass(frozen=True)
class Dataset:
    n_users: int
    n_items: int
    positives_train: np.ndarray  # (n,2) [user, item] pairs available for MF training
    splits: list[UserSplit]


def _data_root() -> str:
    root = os.environ.get("MFMDP_DATA_ROOT", "./data")
    return root


def ensure_movielens() -> str:
    """Download ml-1m.zip if absent; return path to the extracted ratings.dat."""
    root = _data_root()
    target_dir = os.path.join(root, "ml-1m")
    ratings_path = os.path.join(target_dir, "ratings.dat")
    if os.path.exists(ratings_path):
        return ratings_path
    os.makedirs(root, exist_ok=True)
    zip_path = os.path.join(root, "ml-1m.zip")
    if not os.path.exists(zip_path):
        print(f"downloading {ML1M_URL} -> {zip_path}")
        urllib.request.urlretrieve(ML1M_URL, zip_path)
    sha1 = hashlib.sha1(open(zip_path, "rb").read()).hexdigest()
    print(f"ml-1m.zip sha1: {sha1}")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(root)
    if not os.path.exists(ratings_path):
        raise RuntimeError("ratings.dat not found after extraction")
    return ratings_path


def load_movielens(min_rating: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse ratings.dat. Returns (user, item, rating, timestamp) 0-based arrays."""
    path = ensure_movielens()
    users, items, ratings, ts = [], [], [], []
    with open(path, "r", encoding="latin-1") as f:
        for line in f:
            u, i, r, t = line.strip().split("::")
            users.append(int(u))
            items.append(int(i))
            ratings.append(int(r))
            ts.append(int(t))
    u_raw = np.asarray(users)
    i_raw = np.asarray(items)
    _, u_idx = np.unique(u_raw, return_inverse=True)
    _, i_idx = np.unique(i_raw, return_inverse=True)
    return u_idx, i_idx, np.asarray(ratings), np.asarray(ts, dtype=np.int64)


def build_lln(min_rating: int = 4, n_test: int = 1, n_valid: int = 1) -> Dataset:
    """Leave-last-n split per user over positive interactions.

    MF training uses every user's prefix. Validation = the n_valid positives right before
    the test block; test = the last n_test positives.
    """
    u_idx, i_idx, ratings, ts = load_movielens(min_rating)
    pos = ratings >= min_rating
    users = np.unique(u_idx)
    n_users = int(users.max()) + 1
    n_items = int(i_idx.max()) + 1
    train_pairs: list[tuple[int, int]] = []
    splits: list[UserSplit] = []
    for u in users:
        mask = (u_idx == u) & pos
        order = np.argsort(ts[mask], kind="stable")
        items = i_idx[mask][order]
        r = ratings[mask][order]
        if len(items) < n_test + n_valid + 1:
            continue
        prefix = items[: len(items) - n_test - n_valid]
        valid = items[len(items) - n_test - n_valid : len(items) - n_test]
        test = items[len(items) - n_test :]
        train_pairs.extend((int(u), int(i)) for i in prefix)
        splits.append(UserSplit(
            user=int(u), prefix=prefix, valid=valid, test=test,
            prefix_ratings=r[: len(items) - n_test - n_valid],
            valid_ratings=r[len(items) - n_test - n_valid : len(items) - n_test],
            test_ratings=r[len(items) - n_test :],
        ))
    positives_train = np.asarray(train_pairs, dtype=np.int64)
    return Dataset(n_users=n_users, n_items=n_items, positives_train=positives_train, splits=splits)
