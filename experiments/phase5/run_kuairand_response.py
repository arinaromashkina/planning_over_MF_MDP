"""KuaiRand-1K response-model validation (Phase 5, Gate E).

Targets (all decision-time honest, exposure semantics documented in the loader):
- y_click: exposed-and-clicked (is_click);
- y_longview: long_view (the dataset's "effective view" flag);
- y_wr80 (on clicks only): watch ratio >= 0.8 (graded engagement).

Splits: train = standard_1 (2022-04-08..21), test = standard_2 (2022-04-22..05-08)
(chronological; includes temporal drift). The randomized-exposure file (is_rand=1)
is an unbiased probe under the randomization assumption (documented).

Baselines: global mean, per-video train click rate (shrunk), per-user train rate.

Usage: python experiments/phase5/run_kuairand_response.py [--n-steps 400]
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

from mf_mdp.data.kuairand import load_log, load_user_features, load_video_features
from mf_mdp.models.features import FeatureEncoder
from mf_mdp.models.response import (
    LogisticResponseModel,
    brier_score,
    expected_calibration_error,
    log_loss,
    roc_auc,
)


def build_features(log, video_feats, user_feats, pop_exposure, pop_click, user_hist):
    n = len(log.user_idx)
    X = np.zeros((n, 12), dtype=np.float32)
    X[:, 0] = video_feats[log.video_idx, 0]           # video_type
    dur = np.clip(video_feats[log.video_idx, 1], 0, None)
    X[:, 1] = np.log1p(dur)                           # log duration (missing -> -1 -> 0)
    X[:, 2] = video_feats[log.video_idx, 2]           # music_type
    X[:, 3:8] = user_feats[log.user_idx]              # static user features
    X[:, 8] = (log.hourmin // 100)                    # hour of day
    X[:, 9] = np.log1p(pop_exposure[log.video_idx])   # train-period popularity
    X[:, 10] = np.log1p(pop_click[log.video_idx])     # train-period clicks
    X[:, 11] = np.log1p(log.session_pos)              # session position
    assert not np.isnan(X).any(), "NaN in features"
    return X


def metrics_for(y, p):
    return {"n": int(len(y)), "base_rate": float(y.mean()),
            "log_loss": log_loss(y, p), "brier": brier_score(y, p),
            "roc_auc": roc_auc(y, p), "ece": expected_calibration_error(y, p)}


def eval_by_date(y, p, dates):
    out = {}
    for d in np.unique(dates):
        m = dates == d
        out[str(d)] = expected_calibration_error(y[m], p[m])
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-steps", type=int, default=400)
    args = parser.parse_args()
    t0 = time.time()

    print("loading logs...", flush=True)
    train, user_map, video_map = load_log("standard_1")
    test, _, _ = load_log("standard_2", user_map, video_map)
    rand, _, _ = load_log("random", user_map, video_map)
    vf = load_video_features(video_map)
    uf = load_user_features(user_map)
    print(f"  train {len(train.user_idx):,} rows, test {len(test.user_idx):,} rows, "
          f"rand {len(rand.user_idx):,} rows, {len(video_map):,} videos", flush=True)

    # train-period popularity (decision-time legal: uses only train data)
    pop_exposure = np.bincount(train.video_idx, minlength=len(video_map)).astype(np.float64)
    pop_click = np.bincount(train.video_idx, weights=train.is_click, minlength=len(video_map)).astype(np.float64)

    Xtr = build_features(train, vf, uf, pop_exposure, pop_click, None)
    Xte = build_features(test, vf, uf, pop_exposure, pop_click, None)
    Xra = build_features(rand, vf, uf, pop_exposure, pop_click, None)
    enc = FeatureEncoder()

    results = {"train_rows": int(len(train.user_idx)), "test_rows": int(len(test.user_idx)),
               "rand_rows": int(len(rand.user_idx))}
    for target, ytr, yte, yra, mask_tr, mask_te, mask_ra in [
        ("click", train.is_click.astype(float), test.is_click.astype(float), rand.is_click.astype(float),
         np.ones(len(train.user_idx), bool), np.ones(len(test.user_idx), bool), np.ones(len(rand.user_idx), bool)),
        ("long_view", train.long_view.astype(float), test.long_view.astype(float), rand.long_view.astype(float),
         np.ones(len(train.user_idx), bool), np.ones(len(test.user_idx), bool), np.ones(len(rand.user_idx), bool)),
        ("wr80_on_clicks", None, None, None,
         train.is_click == 1, test.is_click == 1, rand.is_click == 1),
    ]:
        if target == "wr80_on_clicks":
            wr_tr = np.clip(train.play_time_ms / np.maximum(train.duration_ms, 1), 0, None)
            wr_te = np.clip(test.play_time_ms / np.maximum(test.duration_ms, 1), 0, None)
            wr_ra = np.clip(rand.play_time_ms / np.maximum(rand.duration_ms, 1), 0, None)
            ytr = (wr_tr >= 0.8).astype(float)
            yte = (wr_te >= 0.8).astype(float)
            yra = (wr_ra >= 0.8).astype(float)
        # model
        m = LogisticResponseModel(l2=1e-4, lr=0.5, n_steps=args.n_steps)
        m.fit(enc.fit_transform(Xtr[mask_tr]), ytr[mask_tr])
        res = {"target": target,
               "test": metrics_for(yte[mask_te], m.predict_proba(enc.transform(Xte[mask_te]))),
               "random_subset": metrics_for(yra[mask_ra], m.predict_proba(enc.transform(Xra[mask_ra]))),
               "test_ece_by_date": eval_by_date(yte[mask_te], m.predict_proba(enc.transform(Xte[mask_te])), test.date[mask_te])}
        # baselines on test
        p_global = np.full(mask_te.sum(), ytr[mask_tr].mean())
        video_rate = (pop_click + 10 * ytr[mask_tr].mean()) / (pop_exposure + 10.0)
        p_pop = video_rate[test.video_idx[mask_te]]
        res["baseline_global"] = metrics_for(yte[mask_te], p_global)
        res["baseline_popularity"] = metrics_for(yte[mask_te], p_pop)
        results[target] = res
        print(f"  {target}: test AUC={res['test']['roc_auc']:.4f} ECE={res['test']['ece']:.4f} "
              f"rand AUC={res['random_subset']['roc_auc']:.4f}", flush=True)

    results["runtime_s"] = time.time() - t0
    out_dir = Path(os.environ.get("MFMDP_OUTPUT_ROOT", "./outputs")) / "phase5"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "kuairand_response.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"wrote {out_dir / 'kuairand_response.json'} ({time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
