"""H4 predictive level: cross-feature response models on KuaiRand-1K.

Primary testbed for the interaction hypothesis (docs/H4_CROSS_RESPONSE_MODEL_DESIGN.md
§3). Session-walked MF-posterior features (Gate-E convention: iALS on train-period
clicks, top-5000 pool, positions 0..7, sessions >= 5 events, pool rows scored and
non-pool rows folded in) plus history/context features, the owner's six grounded
crosses, matched random-cross control, FM and MLP capacity arms, and the u-v
elementwise-product arm (interaction #5).

Splits: train = standard_1 rows (dates <= 20220419), validation = last two days of
standard_1 (20220420-21, HP selection only), test = standard_2 (chronological),
probe = random log (randomized-exposure subset). Primary target wr80
(play_time/duration >= 0.8 over confirmed exposures); secondary click, long_view.

Usage: python experiments/h4/run_kuairand_cross.py [--max-train-sessions 40000]
    [--max-test-sessions 25000] [--n-steps 400]
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from mf_mdp.data.kuairand import _data_root, load_log, load_video_features
from mf_mdp.interactions.cross import (
    BASE_NAMES, KUAI_CROSS_NAMES, KUAI_HIST_NAMES, RandomCrosses, kuairand_crosses,
)
from mf_mdp.interactions.fm import FactorizationMachine
from mf_mdp.interactions.mlp import MLPClassifier
from mf_mdp.mdp import MFState, fold_in_general
from mf_mdp.mf import IALSConfig, train_ials, user_state
from mf_mdp.models.features import FeatureEncoder, features_for
from mf_mdp.models.response import (LogisticResponseModel, brier_score,
                                    expected_calibration_error, log_loss, roc_auc)

TOP_VIDEOS = 5000
MAX_POS = 8
VALID_DATES = (20220420, 20220421)
FEATURE_NAMES = BASE_NAMES + KUAI_HIST_NAMES

ARMS = ["m0", "m0h", "m1", "c1", "fm", "mlp", "mlp_uv"]


def load_author_ids(video_map: dict) -> np.ndarray:
    """author_id per contiguous video idx (for the author-repeat cross; the stock
    loader excludes author_id as high-cardinality — used here only as a counter)."""
    authors = {}
    with open(_data_root() / "video_features_basic_1k.csv") as f:
        for row in csv.DictReader(f):
            vid = int(row["video_id"])
            if vid in video_map:
                authors[video_map[vid]] = int(row["author_id"])
    n = len(video_map)
    out = np.zeros(n, dtype=np.int64)
    for idx, a in authors.items():
        out[idx] = a
    return out


def walk_sessions(log, V, ials_cfg, pool_mask, log_pop, authors, video_feats, max_sessions, seed):
    """One row per scored (session, position): component blocks + labels.

    Same walk as Gate E (non-pool videos folded in, pool rows scored), plus the
    H4 history/context features. All quantities are decision-time legal.
    """
    n = len(log.user_idx)
    rng = np.random.default_rng(seed)
    order = np.lexsort((log.time_ms, log.date, log.user_idx))
    u_s, d_s, t_s = log.user_idx[order], log.date[order], log.time_ms[order]
    v_s, c_s = log.video_idx[order], log.is_click[order]
    lv_s = log.long_view[order]
    sess_s, pos_s = log.session_id[order], log.session_pos[order]
    pt_s, du_s = log.play_time_ms[order], log.duration_ms[order]
    h_s = log.hourmin[order]

    new_grp = np.ones(n, dtype=bool)
    new_grp[1:] = (u_s[1:] != u_s[:-1]) | (d_s[1:] != d_s[:-1]) | (sess_s[1:] != sess_s[:-1])
    starts = np.flatnonzero(new_grp)
    ends = np.concatenate([starts[1:], [n]])
    keep = (ends - starts) >= 5
    starts, ends = starts[keep], ends[keep]
    if len(starts) > max_sessions:
        sel = rng.choice(len(starts), max_sessions, replace=False)
        starts, ends = starts[sel], ends[sel]

    base_rows, hist_rows, uv_rows, vid_rows = [], [], [], []
    y_click, y_lv, y_wr80, dates, poss = [], [], [], [], []
    for i, j in zip(starts, ends):
        st = user_state(V, np.empty(0, dtype=np.int64), ials_cfg)
        wr_hist, prev_t = [], None
        for k in range(i, min(j, i + MAX_POS)):
            vid = int(v_s[k])
            tm = int(t_s[k])
            wr = float(np.clip(pt_s[k] / max(du_s[k], 1), 0, None))
            if pool_mask[vid]:
                recency_min = 0.0 if prev_t is None else max((tm - prev_t) / 60000.0, 0.0)
                prefix_vids = v_s[i:k]
                auth = authors[vid]
                hrow = np.array([
                    float(np.mean(wr_hist)) if wr_hist else 0.0,   # mean_wr_hist
                    float(np.log1p(recency_min)),                  # log1p_recency
                    float(np.sum(prefix_vids == vid)),             # repeat_cnt
                    float(np.sum(authors[prefix_vids] == auth)) if k > i else 0.0,  # author_repeat
                    float(h_s[k] // 100),                          # hour
                    float(np.log1p(max(video_feats[vid, 1], 0.0))),  # log_duration
                    float(video_feats[vid, 0]),                    # video_type
                    float(pos_s[k]),                               # session_pos
                ])
                base_rows.append(features_for(st, V, vid, log_pop))
                hist_rows.append(hrow)
                uv_rows.append(st.u * V[vid])
                vid_rows.append(vid)
                y_click.append(float(c_s[k]))
                y_lv.append(float(lv_s[k]))
                y_wr80.append(1.0 if wr >= 0.8 else 0.0)
                dates.append(int(d_s[k]))
                poss.append(int(pos_s[k]))
            wr_hist.append(wr)
            prev_t = tm
            P_p, u_p, _ = fold_in_general(st, V[vid], ials_cfg.alpha, 1.0 + ials_cfg.alpha)
            st = MFState(P=P_p, u=u_p, consumed=tuple(sorted(st.consumed + (vid,))))
    return {
        "base": np.vstack(base_rows).astype(np.float32),
        "hist": np.vstack(hist_rows).astype(np.float32),
        "uv": np.vstack(uv_rows).astype(np.float32),
        "vid": np.asarray(vid_rows, dtype=np.int64),
        "y_click": np.asarray(y_click),
        "y_long_view": np.asarray(y_lv),
        "y_wr80": np.asarray(y_wr80),
        "date": np.asarray(dates),
        "pos": np.asarray(poss),
    }


def assemble(rows, arm, rand_crosses=None):
    base, hist, uv = rows["base"], rows["hist"], rows["uv"]
    if arm == "m0":
        return base
    bh = np.hstack([base, hist])
    if arm in ("m0h", "fm", "mlp"):
        return bh
    if arm == "m1":
        cos, leverage, log_pop_c, hist_len = base[:, 0], base[:, 2], base[:, 5], base[:, 6]
        recency_decay = np.exp(-(np.expm1(hist[:, 1])) / 30.0)
        cross = np.column_stack([
            cos * hist_len,
            cos * recency_decay,
            hist[:, 0] * cos,
            leverage * log_pop_c,
            hist[:, 2] * cos,
            hist[:, 3] * cos,
        ])
        return np.hstack([bh, cross])
    if arm == "c1":
        return np.hstack([bh, rand_crosses.transform(bh)])
    if arm == "mlp_uv":
        return np.hstack([bh, uv])
    raise ValueError(arm)


def metrics_for(y, p):
    return {"n": int(len(y)), "base_rate": float(y.mean()),
            "log_loss": log_loss(y, p), "brier": brier_score(y, p),
            "roc_auc": roc_auc(y, p), "ece": expected_calibration_error(y, p)}


def eval_by_date(y, p, dates):
    return {str(d): expected_calibration_error(y[dates == d], p[dates == d]) for d in np.unique(dates)}


def eval_by_pos(y, p, pos):
    out = {}
    for t in range(MAX_POS):
        m = pos == t
        if m.sum() >= 100:
            out[str(t)] = {"n": int(m.sum()), "roc_auc": roc_auc(y[m], p[m]),
                           "ece": expected_calibration_error(y[m], p[m]),
                           "base_rate": float(y[m].mean())}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-train-sessions", type=int, default=40000)
    ap.add_argument("--max-test-sessions", type=int, default=25000)
    ap.add_argument("--n-steps", type=int, default=400)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if args.quick:
        args.max_train_sessions, args.max_test_sessions = 4000, 2500

    t0 = time.time()
    print("loading logs...", flush=True)
    train, user_map, video_map = load_log("standard_1")
    test, _, _ = load_log("standard_2", user_map, video_map)
    rand, _, _ = load_log("random", user_map, video_map)
    vf = load_video_features(video_map)
    authors = load_author_ids(video_map)
    print(f"  train {len(train.user_idx):,} rows, test {len(test.user_idx):,} rows, "
          f"rand {len(rand.user_idx):,} rows, {len(video_map):,} videos", flush=True)

    exposure = np.bincount(train.video_idx, minlength=len(video_map)).astype(float)
    pool = set(np.argsort(-exposure)[:TOP_VIDEOS].tolist())
    pool_mask = np.zeros(len(video_map), dtype=bool)
    pool_mask[list(pool)] = True
    clicked_mask = (train.is_click == 1) & pool_mask[train.video_idx]
    positives = np.column_stack([train.user_idx[clicked_mask], train.video_idx[clicked_mask]])
    print(f"  pool {len(pool)} videos, {len(positives):,} train interactions", flush=True)

    ials_cfg = IALSConfig(dim=16, alpha=40.0, lam=1.0, n_iter=10)
    U, V = train_ials(positives, train.n_users, len(video_map), ials_cfg, seed=0)
    log_pop = np.log1p(exposure)
    print("  iALS trained", flush=True)

    print("walking sessions...", flush=True)
    rows_tr = walk_sessions(train, V, ials_cfg, pool_mask, log_pop, authors, vf,
                            args.max_train_sessions, seed=0)
    rows_te = walk_sessions(test, V, ials_cfg, pool_mask, log_pop, authors, vf,
                            args.max_test_sessions, seed=1)
    rows_ra = walk_sessions(rand, V, ials_cfg, pool_mask, log_pop, authors, vf, 100_000, seed=2)
    print(f"  scored rows: train {len(rows_tr['y_wr80']):,}, test {len(rows_te['y_wr80']):,}, "
          f"rand {len(rows_ra['y_wr80']):,}", flush=True)

    va_mask = np.isin(rows_tr["date"], VALID_DATES)
    tr_mask = ~va_mask

    y_name = "y_wr80"
    y_tr, y_va = rows_tr[y_name][tr_mask], rows_tr[y_name][va_mask]
    y_te, y_ra = rows_te[y_name], rows_ra[y_name]

    # ---- validation-grid HP selection for FM k / MLP hidden ----
    X_va_baseh = assemble({k: v[va_mask] for k, v in rows_tr.items()}, "m0h")
    X_tr_baseh = assemble({k: v[tr_mask] for k, v in rows_tr.items()}, "m0h")
    hp = {}
    best = {}
    for k in (4, 8, 16):
        enc = FeatureEncoder()
        m = FactorizationMachine(k=k, l2=1e-4, lr=0.05, n_epochs=10, seed=0)
        m.fit(enc.fit_transform(X_tr_baseh), y_tr)
        ll = log_loss(y_va, m.predict_proba(enc.transform(X_va_baseh)))
        print(f"  FM k={k}: valid LL {ll:.4f}", flush=True)
        if "fm" not in best or ll < best["fm"][1]:
            best["fm"] = (k, ll)
    for h in (32, 64):
        enc = FeatureEncoder()
        m = MLPClassifier(hidden=h, l2=1e-4, lr=0.05, n_epochs=10, seed=0)
        m.fit(enc.fit_transform(X_tr_baseh), y_tr)
        ll = log_loss(y_va, m.predict_proba(enc.transform(X_va_baseh)))
        print(f"  MLP h={h}: valid LL {ll:.4f}", flush=True)
        if "mlp" not in best or ll < best["mlp"][1]:
            best["mlp"] = (h, ll)
    hp["fm_k"], hp["fm_valid_ll"] = best["fm"]
    hp["mlp_hidden"], hp["mlp_valid_ll"] = best["mlp"]

    rand_crosses = RandomCrosses(k=len(KUAI_CROSS_NAMES), seed=0).fit(X_tr_baseh)

    def fit_arm(arm):
        Xtr = assemble({k: v[tr_mask] for k, v in rows_tr.items()}, arm, rand_crosses)
        enc = FeatureEncoder()
        Xt = enc.fit_transform(Xtr)
        if arm in ("m0", "m0h", "m1", "c1"):
            m = LogisticResponseModel(l2=1e-4, lr=0.5, n_steps=args.n_steps)
        elif arm == "fm":
            m = FactorizationMachine(k=hp["fm_k"], l2=1e-4, lr=0.05, n_epochs=10, seed=0)
        elif arm == "mlp":
            m = MLPClassifier(hidden=hp["mlp_hidden"], l2=1e-4, lr=0.05, n_epochs=10, seed=0)
        else:  # mlp_uv
            m = MLPClassifier(hidden=hp["mlp_hidden"], l2=1e-4, lr=0.05, n_epochs=10, seed=0)
        m.fit(Xt, y_tr)
        return enc, m

    results = {"config": vars(args),
               "feature_names": FEATURE_NAMES, "cross_names": KUAI_CROSS_NAMES,
               "hp": hp, "target_primary": y_name,
               "rows": {"train": int(tr_mask.sum()), "valid": int(va_mask.sum()),
                        "test": int(len(y_te)), "rand": int(len(y_ra))}}
    for arm in ARMS:
        enc, m = fit_arm(arm)
        p_te = m.predict_proba(enc.transform(assemble(rows_te, arm, rand_crosses)))
        p_ra = m.predict_proba(enc.transform(assemble(rows_ra, arm, rand_crosses)))
        res = {"test": metrics_for(y_te, p_te), "random_subset": metrics_for(y_ra, p_ra),
               "test_ece_by_date": eval_by_date(y_te, p_te, rows_te["date"]),
               "test_by_pos": eval_by_pos(y_te, p_te, rows_te["pos"])}
        # secondary targets for the main comparison arms
        for tgt in ("y_click", "y_long_view"):
            enc2, m2 = None, None
            if arm in ("m0", "m1", "fm", "mlp"):
                Xtr = assemble({k: v[tr_mask] for k, v in rows_tr.items()}, arm, rand_crosses)
                enc2 = FeatureEncoder()
                if arm in ("m0", "m1"):
                    m2 = LogisticResponseModel(l2=1e-4, lr=0.5, n_steps=args.n_steps)
                elif arm == "fm":
                    m2 = FactorizationMachine(k=hp["fm_k"], l2=1e-4, lr=0.05, n_epochs=10, seed=0)
                else:
                    m2 = MLPClassifier(hidden=hp["mlp_hidden"], l2=1e-4, lr=0.05, n_epochs=10, seed=0)
                m2.fit(enc2.fit_transform(Xtr), rows_tr[tgt][tr_mask])
                p2 = m2.predict_proba(enc2.transform(assemble(rows_te, arm, rand_crosses)))
                res[tgt] = metrics_for(rows_te[tgt], p2)
        results[arm] = res
        print(f"  {arm}: test AUC={res['test']['roc_auc']:.4f} LL={res['test']['log_loss']:.4f} "
              f"ECE={res['test']['ece']:.4f} | rand AUC={res['random_subset']['roc_auc']:.4f}", flush=True)

    # reference baselines on the primary target
    p_global = np.full(len(y_te), y_tr.mean())
    pop_click = np.bincount(train.video_idx, weights=train.is_click, minlength=len(video_map))
    video_rate = (pop_click + 10 * y_tr.mean()) / (exposure + 10.0)
    results["baseline_global"] = metrics_for(y_te, p_global)
    results["baseline_popularity"] = metrics_for(y_te, video_rate[rows_te["vid"]])

    try:
        commit = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                         cwd=Path(__file__).resolve().parent.parent.parent).decode().strip()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"],
                                         cwd=Path(__file__).resolve().parent.parent.parent).decode().strip()
    except Exception:
        commit, branch = None, None
    results["git"] = {"commit": commit, "branch": branch}
    results["runtime_s"] = time.time() - t0
    out_dir = Path(os.environ.get("MFMDP_OUTPUT_ROOT", "./outputs")) / "h4"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "kuairand_cross.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
