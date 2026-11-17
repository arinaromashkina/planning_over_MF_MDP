"""H4 predictive level, secondary testbed: ML-1M (Phase-4 E2 protocol).

Same ladder as KuaiRand but with the ML-1M-feasible subset of the grounded
crosses (docs/H4_CROSS_RESPONSE_MODEL_DESIGN.md §3.2): #1 cos x hist_len,
#3 mean_hist_rating x cos, #4 leverage x log_pop. #2 (recency) is order-based
in the LLN protocol (timestamps unused) and #6 is inert (re-consumption rare) —
both documented as not built. Label: rating == 5; user-disjoint 70/15/15 split.

Usage: python experiments/h4/run_ml1m_cross.py [--seed 0]
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "phase4"))

from common import split_users
from mf_mdp.data.movielens import build_lln
from mf_mdp.interactions.cross import RandomCrosses
from mf_mdp.interactions.fm import FactorizationMachine
from mf_mdp.interactions.mlp import MLPClassifier
from mf_mdp.mdp import MFState, fold_in_general
from mf_mdp.mf import IALSConfig, train_ials, user_state
from mf_mdp.models.features import FeatureEncoder, features_for
from mf_mdp.models.response import (LogisticResponseModel, brier_score,
                                    expected_calibration_error, log_loss, roc_auc)


def walk_decision_rows(data, V, cfg, log_pop, max_t=40):
    """Mirror of phase4 build_decision_dataset plus the rating-history feature."""
    X_base, mean_rat, y5, uid = [], [], [], []
    for sp in data.splits:
        seq, rat = sp.prefix, sp.prefix_ratings
        T = len(seq)
        if T < 3 or rat is None:
            continue
        st = user_state(V, np.empty(0, dtype=np.int64), cfg)
        t0 = max(1, T - max_t)
        for t in range(1, T):
            a = int(seq[t])
            if t >= t0:
                X_base.append(features_for(st, V, a, log_pop))
                mean_rat.append(float(np.mean(rat[:t])))
                y5.append(1.0 if rat[t] >= 5 else 0.0)
                uid.append(sp.user)
            P_p, u_p, _ = fold_in_general(st, V[a], cfg.alpha, 1.0 + cfg.alpha)
            st = MFState(P=P_p, u=u_p, consumed=tuple(sorted(st.consumed + (a,))))
    return np.vstack(X_base), np.asarray(mean_rat), np.asarray(y5), np.asarray(uid)


def assemble(base, mean_rat, arm, rand_crosses=None):
    hist = mean_rat[:, None]
    if arm == "m0":
        return base
    bh = np.hstack([base, hist])
    if arm in ("m0h", "fm", "mlp"):
        return bh
    if arm == "m1":
        cos, leverage, log_pop_c, hist_len = base[:, 0], base[:, 2], base[:, 5], base[:, 6]
        cross = np.column_stack([cos * hist_len, mean_rat * cos, leverage * log_pop_c])
        return np.hstack([bh, cross])
    if arm == "c1":
        return np.hstack([bh, rand_crosses.transform(bh)])
    raise ValueError(arm)


ARMS = ["m0", "m0h", "m1", "c1", "fm", "mlp"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-steps", type=int, default=400)
    args = ap.parse_args()
    t0 = time.time()

    data = build_lln(min_rating=4, n_test=1, n_valid=1)
    cfg = IALSConfig(dim=64, alpha=40.0, lam=1.0, n_iter=20)
    _, V = train_ials(data.positives_train, data.n_users, data.n_items, cfg, seed=args.seed)
    log_pop = np.log1p(np.bincount(data.positives_train[:, 1], minlength=data.n_items).astype(float))

    print("building decision dataset...", flush=True)
    base, mean_rat, y5, uid = walk_decision_rows(data, V, cfg, log_pop)
    print(f"  {len(y5)} examples, {y5.mean():.3f} rating-5 rate", flush=True)

    tr_u, va_u, te_u = split_users(uid, seed=args.seed)
    m_tr, m_va, m_te = np.isin(uid, tr_u), np.isin(uid, va_u), np.isin(uid, te_u)

    hp = {}
    best = {}
    Xva = assemble(base[m_va], mean_rat[m_va], "m0h")
    Xtr_bh = assemble(base[m_tr], mean_rat[m_tr], "m0h")
    for k in (4, 8, 16):
        enc = FeatureEncoder()
        m = FactorizationMachine(k=k, l2=1e-4, lr=0.05, n_epochs=10, seed=0)
        m.fit(enc.fit_transform(Xtr_bh), y5[m_tr])
        ll = log_loss(y5[m_va], m.predict_proba(enc.transform(Xva)))
        print(f"  FM k={k}: valid LL {ll:.4f}", flush=True)
        if "fm" not in best or ll < best["fm"][1]:
            best["fm"] = (k, ll)
    for h in (32, 64):
        enc = FeatureEncoder()
        m = MLPClassifier(hidden=h, l2=1e-4, lr=0.05, n_epochs=10, seed=0)
        m.fit(enc.fit_transform(Xtr_bh), y5[m_tr])
        ll = log_loss(y5[m_va], m.predict_proba(enc.transform(Xva)))
        print(f"  MLP h={h}: valid LL {ll:.4f}", flush=True)
        if "mlp" not in best or ll < best["mlp"][1]:
            best["mlp"] = (h, ll)
    hp["fm_k"], hp["mlp_hidden"] = best["fm"][0], best["mlp"][0]

    rand_crosses = RandomCrosses(k=3, seed=0).fit(Xtr_bh)

    def metrics(y, p):
        return {"n": int(len(y)), "base_rate": float(y.mean()), "log_loss": log_loss(y, p),
                "brier": brier_score(y, p), "roc_auc": roc_auc(y, p),
                "ece": expected_calibration_error(y, p)}

    results = {"config": vars(args), "hp": hp, "arms": ARMS}
    for arm in ARMS:
        Xtr = assemble(base[m_tr], mean_rat[m_tr], arm, rand_crosses)
        enc = FeatureEncoder()
        if arm in ("m0", "m0h", "m1", "c1"):
            m = LogisticResponseModel(l2=1e-4, lr=0.5, n_steps=args.n_steps)
        elif arm == "fm":
            m = FactorizationMachine(k=hp["fm_k"], l2=1e-4, lr=0.05, n_epochs=10, seed=0)
        else:
            m = MLPClassifier(hidden=hp["mlp_hidden"], l2=1e-4, lr=0.05, n_epochs=10, seed=0)
        m.fit(enc.fit_transform(Xtr), y5[m_tr])
        p_te = m.predict_proba(enc.transform(assemble(base[m_te], mean_rat[m_te], arm, rand_crosses)))
        results[arm] = {"test": metrics(y5[m_te], p_te),
                        "train": metrics(y5[m_tr], m.predict_proba(enc.transform(Xtr)))}
        print(f"  {arm}: test AUC={results[arm]['test']['roc_auc']:.4f} "
              f"LL={results[arm]['test']['log_loss']:.4f} ECE={results[arm]['test']['ece']:.4f}",
              flush=True)

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
    path = out_dir / "ml1m_cross.json"
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
