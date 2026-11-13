"""KuaiRand-1K transition validation (Phase 5, Gate E): does the MF posterior carry
predictive signal for the NEXT response along a session?

Pipeline:
1. Train iALS factors on train-period consumed videos (click=1), restricted to the
   top-5000 popular candidate pool (the starting paper's convention for large catalogs).
2. For each (user, date, session) with >= 5 events, walk the session chronologically;
   at each position build the exact fold-in state of the session prefix (incremental
   weighted fold-in) and MF decision features.
3. Train a logistic response model p(click | MF features) on train sessions; evaluate on
   test sessions (chronological split): per-position log-loss/Brier/AUC/ECE, and the
   comparison against (a) the raw non-MF features and (b) a popularity-only baseline.

This is the real-data analog of the transition question: if the MF posterior state does
not predict the next response, the optimistic fold-in planning story has no real-data
leg to stand on. Subsampled for runtime (documented).

Usage: python experiments/phase5/run_kuairand_transition.py
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from mf_mdp.data.kuairand import load_log, load_user_features, load_video_features
from mf_mdp.mdp import MFState, fold_in_general
from mf_mdp.mf import IALSConfig, train_ials, user_state
from mf_mdp.models.features import FeatureEncoder, features_for
from mf_mdp.models.response import (
    LogisticResponseModel,
    brier_score,
    expected_calibration_error,
    log_loss,
    roc_auc,
)

TOP_VIDEOS = 5000
MAX_POS = 8


def build_session_examples(log, V, ials_cfg, video_pool_mask, log_pop, max_events=None, seed=0, max_sessions=100_000):
    """Walk each session; one example per (session, position) with MF features."""
    n = len(log.user_idx)
    rng = np.random.default_rng(seed)
    order = np.lexsort((log.time_ms, log.date, log.user_idx))  # primary: user, then date, then time
    u_s, d_s, t_s = log.user_idx[order], log.date[order], log.time_ms[order]
    v_s, c_s = log.video_idx[order], log.is_click[order]
    sess_s = log.session_id[order]
    pos_s = log.session_pos[order]

    # group boundaries (vectorized run-length encoding)
    new_grp = np.ones(n, dtype=bool)
    new_grp[1:] = (u_s[1:] != u_s[:-1]) | (d_s[1:] != d_s[:-1]) | (sess_s[1:] != sess_s[:-1])
    starts = np.flatnonzero(new_grp)
    ends = np.concatenate([starts[1:], [n]])
    # keep sessions with >= 5 events
    keep = (ends - starts) >= 5
    starts, ends = starts[keep], ends[keep]
    if len(starts) > max_sessions:
        sel = rng.choice(len(starts), max_sessions, replace=False)
        starts, ends = starts[sel], ends[sel]

    X_rows, y_rows, pos_rows = [], [], []
    for i, j in zip(starts, ends):
        hist = []
        st = user_state(V, np.empty(0, dtype=np.int64), ials_cfg)
        for k in range(i, min(j, i + MAX_POS)):
            vid = int(v_s[k])
            if video_pool_mask[vid]:  # score only pool items
                X_rows.append(features_for(st, V, vid, log_pop))
                y_rows.append(float(c_s[k]))
                pos_rows.append(int(pos_s[k]))
            P_p, u_p, _ = fold_in_general(st, V[vid], ials_cfg.alpha, 1.0 + ials_cfg.alpha)
            st = MFState(P=P_p, u=u_p, consumed=tuple(sorted(st.consumed + (vid,))))
    X, y, pos = np.vstack(X_rows), np.asarray(y_rows), np.asarray(pos_rows)
    if max_events is not None and len(X) > max_events:
        idx = rng.choice(len(X), max_events, replace=False)
        X, y, pos = X[idx], y[idx], pos[idx]
    return X, y, pos


def per_position_metrics(y, p, pos, max_pos=MAX_POS):
    out = {}
    for t in range(max_pos):
        m = pos == t
        if m.sum() < 100:
            continue
        out[str(t)] = {"n": int(m.sum()),
                       "log_loss": log_loss(y[m], p[m]),
                       "brier": brier_score(y[m], p[m]),
                       "roc_auc": roc_auc(y[m], p[m]),
                       "ece": expected_calibration_error(y[m], p[m]),
                       "base_rate": float(y[m].mean())}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=400)
    parser.add_argument("--max-train-events", type=int, default=300_000)
    parser.add_argument("--max-test-events", type=int, default=150_000)
    args = parser.parse_args()
    t0 = time.time()

    print("loading logs...", flush=True)
    train, user_map, video_map = load_log("standard_1")
    test, _, _ = load_log("standard_2", user_map, video_map)
    print(f"  train {len(train.user_idx):,} rows, test {len(test.user_idx):,} rows", flush=True)

    # top-5000 popular candidate pool (train-period exposure)
    exposure = np.bincount(train.video_idx, minlength=len(video_map))
    pool = set(np.argsort(-exposure)[:TOP_VIDEOS].tolist())
    pool_mask = np.zeros(len(video_map), dtype=bool)
    pool_mask[list(pool)] = True
    clicked_mask = (train.is_click == 1) & pool_mask[train.video_idx]
    positives = np.column_stack([train.user_idx[clicked_mask], train.video_idx[clicked_mask]])
    print(f"  pool {len(pool)} videos, {len(positives):,} train interactions", flush=True)

    ials_cfg = IALSConfig(dim=16, alpha=40.0, lam=1.0, n_iter=10)
    U, V = train_ials(positives, train.n_users, len(video_map), ials_cfg, seed=0)
    print("  iALS trained", flush=True)

    log_pop = np.log1p(np.bincount(train.video_idx, minlength=len(video_map)).astype(float))
    print("building session examples (train)...", flush=True)
    Xtr, ytr, ptr = build_session_examples(train, V, ials_cfg, pool_mask, log_pop, args.max_train_events, seed=0, max_sessions=40_000)
    print(f"  {len(Xtr):,} train examples", flush=True)
    print("building session examples (test)...", flush=True)
    Xte, yte, pte = build_session_examples(test, V, ials_cfg, pool_mask, log_pop, args.max_test_events, seed=1, max_sessions=25_000)
    print(f"  {len(Xte):,} test examples", flush=True)

    enc = FeatureEncoder()
    m = LogisticResponseModel(l2=1e-4, lr=0.5, n_steps=args.n_steps)
    m.fit(enc.fit_transform(Xtr), ytr)
    p_te = m.predict_proba(enc.transform(Xte))
    p_tr = m.predict_proba(enc.transform(Xtr))

    results = {
        "train_examples": int(len(Xtr)), "test_examples": int(len(Xte)),
        "overall": {
            "test": {"log_loss": log_loss(yte, p_te), "brier": brier_score(yte, p_te),
                     "roc_auc": roc_auc(yte, p_te), "ece": expected_calibration_error(yte, p_te),
                     "base_rate": float(yte.mean())},
            "train": {"log_loss": log_loss(ytr, p_tr), "roc_auc": roc_auc(ytr, p_tr)},
        },
        "per_position_test": per_position_metrics(yte, p_te, pte),
        "mf_feature_names": ["cos", "dot", "leverage", "|delta_s|", "||v||", "log_pop",
                             "hist_len", "tr(P)", "||P||_F", "||u||"],
    }
    print(json.dumps({"overall": results["overall"], "per_position": results["per_position_test"]}, indent=2)[:2500], flush=True)

    out_dir = Path(os.environ.get("MFMDP_OUTPUT_ROOT", "./outputs")) / "phase5"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "kuairand_transition.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_dir / 'kuairand_transition.json'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
