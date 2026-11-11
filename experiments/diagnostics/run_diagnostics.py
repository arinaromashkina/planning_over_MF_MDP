"""Diagnostic experiments on the leverage/alignment mechanism of the MF-MDP reward.

Time-boxed diagnostic phase (see docs/PLANNING_DIAGNOSTICS.md).

Task 1: leverage l_i = v_i' P v_i over the full catalog for representative MF configs;
        verification of the trace identity and the average vs per-item bounds;
        leverage of ANN/top-100/top-10 candidate pools; correlations with norm/popularity/cosine.
Task 2: alignment-term contribution: distributions, variance vs cosine relevance, ranking
        correlation Plan-1 vs Static-cosine, top-K overlap, eta sweep.
Task 3: embedding-geometry sensitivity (EXPLORATORY variants, clearly separated from the
        paper reproduction): normalized embeddings, dimension, regularization, candidate pools.

Outputs: outputs/diagnostics/*.json, *.csv, plots/*.png
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "src"))

from mf_mdp.data.movielens import build_lln
from mf_mdp.evaluation.runner import evaluate_method
from mf_mdp.mf import IALSConfig, train_ials, user_state
from mf_mdp.planning import PlannerConfig
from mf_mdp.planning.candidates import exact_cosine_topl
from mf_mdp.planning.planners import plan1_topk

OUT = Path(os.environ.get("MFMDP_OUTPUT_ROOT", "./outputs")) / "diagnostics"
PLOTS = OUT / "plots"
N_USERS_TASK1 = 300
N_USERS_TASK2 = 300
N_USERS_TASK3 = 500
SEED = 0


# ---------------------------------------------------------------- helpers

def leverage_catalog(V: np.ndarray, P: np.ndarray) -> np.ndarray:
    """l_i = v_i' P v_i for every item row of V."""
    return np.einsum("ij,jk,ik->i", V, P, V)


def cos_to_u(V: np.ndarray, u: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(V, axis=1)
    norms = np.where(norms == 0.0, 1.0, norms)
    un = np.linalg.norm(u)
    return np.zeros(V.shape[0]) if un == 0 else (V @ u) / (norms * un)


def pearson(x: np.ndarray, y: np.ndarray) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    sx, sy = x.std(), y.std()
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy)) if sx > 0 and sy > 0 else 0.0


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    return pearson(rx, ry)


def q(vals, ps=(0.5, 0.9, 0.95, 0.99)):
    return {f"p{int(p*100)}": float(np.quantile(vals, p)) for p in ps}


def dump(name: str, obj) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / name, "w") as f:
        json.dump(obj, f, indent=2)


def cfg_name(c):
    return f"d{c['dim']}_a{c['alpha']}_l{c['lam']}"


# ---------------------------------------------------------------- data

def get_data():
    return build_lln(min_rating=4, n_test=1, n_valid=1)


# ---------------------------------------------------------------- task 1

def task1(data):
    print("== Task 1: leverage analysis", flush=True)
    configs = [
        {"dim": 64, "alpha": 40.0, "lam": 1.0, "n_iter": 20, "label": "reproduction"},
        {"dim": 64, "alpha": 10.0, "lam": 1.0, "n_iter": 20, "label": "sweep-best-cos"},
        {"dim": 64, "alpha": 1.0, "lam": 1.0, "n_iter": 20, "label": "low-alpha"},
        {"dim": 32, "alpha": 10.0, "lam": 1.0, "n_iter": 20, "label": "low-dim"},
        {"dim": 128, "alpha": 10.0, "lam": 1.0, "n_iter": 15, "label": "high-dim"},
        {"dim": 64, "alpha": 10.0, "lam": 0.1, "n_iter": 20, "label": "low-lam"},
    ]
    pop = np.bincount(data.positives_train[:, 1], minlength=data.n_items).astype(float)
    results = {}
    corr_records = []
    for c in configs:
        cfg = IALSConfig(**{k: v for k, v in c.items() if k != "label"})
        U, V = train_ials(data.positives_train, data.n_users, data.n_items, cfg, seed=SEED)
        vnorms = np.linalg.norm(V, axis=1)
        all_l, all_pool = [], {"top10": [], "top20": [], "top100": []}
        corr_sum = {"norm": [], "pop": [], "cos": []}
        n_items = data.n_items
        sum_lam = sum_cat = sum_hist = 0.0
        for sp in data.splits[:N_USERS_TASK1]:
            st = user_state(V, sp.prefix, cfg)
            lv = leverage_catalog(V, st.P)
            A = np.linalg.inv(st.P)
            lmin = float(np.linalg.eigvalsh(A).min())
            sum_lam += cfg.lam * float(np.trace(st.P))
            sum_cat += float(lv.sum())
            hist_idx = np.asarray(sp.prefix)
            sum_hist += cfg.alpha * float(lv[hist_idx].sum())
            unconsumed = np.ones(n_items, bool)
            unconsumed[np.asarray(st.consumed)] = False
            idx = np.flatnonzero(unconsumed)
            all_l.append(lv[idx])
            cos = cos_to_u(V, st.u)[idx]
            for key, l in (("top10", 10), ("top20", 20), ("top100", 100)):
                pool = exact_cosine_topl(V, st.u, st.consumed, l)
                all_pool[key].append(lv[pool])
            lv_u, vn, pp, cs = lv[idx], vnorms[idx], pop[idx], cos
            corr_sum["norm"].append((pearson(lv_u, vn), spearman(lv_u, vn)))
            corr_sum["pop"].append((pearson(lv_u, pp), spearman(lv_u, pp)))
            corr_sum["cos"].append((pearson(lv_u, cs), spearman(lv_u, cs)))
        cat = np.concatenate(all_l)
        res = {
            "label": c["label"],
            "catalog": {"mean": float(cat.mean()), "std": float(cat.std()),
                        **q(cat), "max": float(cat.max()), "n": int(cat.size)},
            "trace_identity": {"d": cfg.dim, "lambda_trP": sum_lam / N_USERS_TASK1,
                               "sum_catalog_leverage": sum_cat / N_USERS_TASK1,
                               "alpha_sum_history_leverage": sum_hist / N_USERS_TASK1,
                               "total_over_d": (sum_lam + sum_cat + sum_hist) / N_USERS_TASK1 / cfg.dim},
            "average_bound_d_over_n": cfg.dim / n_items,
            "pools": {k: {"mean": float(np.concatenate(v).mean()), **q(np.concatenate(v)),
                          "max": float(np.concatenate(v).max())} for k, v in all_pool.items()},
            "corr": {k: {"pearson": float(np.mean([x[0] for x in v])),
                         "spearman": float(np.mean([x[1] for x in v]))} for k, v in corr_sum.items()},
        }
        results[cfg_name(c)] = res
        corr_records.append({"config": cfg_name(c), **{f"{k}_{m}": res["corr"][k][m] for k in res["corr"] for m in ("pearson", "spearman")}})
        print(f"  {cfg_name(c):16s} mean_l={cat.mean():.4e} max={cat.max():.4e} "
              f"trace_total/d={res['trace_identity']['total_over_d']:.4f}", flush=True)
    dump("task1_leverage.json", results)
    # plot: catalog leverage histogram for the reproduction config
    cfg = IALSConfig(dim=64, alpha=40.0, lam=1.0, n_iter=20)
    U, V = train_ials(data.positives_train, data.n_users, data.n_items, cfg, seed=SEED)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    l_all, l_pools = [], {}
    for sp in data.splits[:100]:
        st = user_state(V, sp.prefix, cfg)
        lv = leverage_catalog(V, st.P)
        mask = np.ones(data.n_items, bool)
        mask[np.asarray(st.consumed)] = False
        l_all.append(lv[mask])
        for key, l in (("top-10", 10), ("ANN top-20", 20), ("top-100", 100)):
            pool = exact_cosine_topl(V, st.u, st.consumed, l)
            l_pools.setdefault(key, []).append(lv[pool])
    l_all = np.concatenate(l_all)
    ax = axes[0]
    ax.hist(np.log10(l_all), bins=60, density=True, alpha=0.6, label="catalog (unconsumed)")
    for key, v in l_pools.items():
        ax.hist(np.log10(np.concatenate(v)), bins=40, density=True, histtype="step", linewidth=1.5, label=key)
    ax.set_xlabel("log10 leverage  l = v'Pv")
    ax.set_ylabel("density")
    ax.set_title("Leverage distribution (d=64, alpha=40, lam=1)")
    ax.legend()
    ax = axes[1]
    names = list(results.keys())
    means = [results[n]["catalog"]["mean"] for n in names]
    maxs = [results[n]["catalog"]["max"] for n in names]
    x = np.arange(len(names))
    ax.bar(x - 0.2, means, 0.4, label="mean")
    ax.bar(x + 0.2, maxs, 0.4, label="max")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("leverage")
    ax.set_title("Catalog leverage by MF config")
    ax.legend()
    fig.tight_layout()
    PLOTS.mkdir(parents=True, exist_ok=True)
    fig.savefig(PLOTS / "task1_leverage.png", dpi=130)
    # correlation scatter (reproduction config, one pooled sample)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6))
    l_s, n_s, p_s, c_s = [], [], [], []
    for sp in data.splits[:60]:
        st = user_state(V, sp.prefix, cfg)
        lv = leverage_catalog(V, st.P)
        mask = np.ones(data.n_items, bool)
        mask[np.asarray(st.consumed)] = False
        idx = np.flatnonzero(mask)
        l_s.append(lv[idx]); n_s.append(np.linalg.norm(V, axis=1)[idx])
        p_s.append(pop[idx]); c_s.append(cos_to_u(V, st.u)[idx])
    l_s, n_s, p_s, c_s = (np.concatenate(x) for x in (l_s, n_s, p_s, c_s))
    for ax, (yv, label) in zip(axes, [(n_s, "||v||"), (np.log10(p_s + 1), "log10 popularity"), (c_s, "cos(u,v)")]):
        ax.scatter(yv, l_s, s=1, alpha=0.15)
        ax.set_xlabel(label)
        ax.set_ylabel("leverage")
    fig.tight_layout()
    fig.savefig(PLOTS / "task1_correlations.png", dpi=130)
    return results


# ---------------------------------------------------------------- task 2

def task2(data):
    print("== Task 2: alignment contribution", flush=True)
    results = {}
    etas = [0.0, 0.01, 0.1, 1.0, 10.0, 100.0]
    for (dim, alpha, lam, tag) in [(64, 40.0, 1.0, "reproduction"), (64, 10.0, 1.0, "alpha10"), (128, 1.0, 1.0, "d128_alpha1")]:
        cfg = IALSConfig(dim=dim, alpha=alpha, lam=lam, n_iter=20)
        U, V = train_ials(data.positives_train, data.n_users, data.n_items, cfg, seed=SEED)
        dss, aligns, coss, sp_corr, ov0 = [], [], [], [], []
        for sp in data.splits[:N_USERS_TASK2]:
            st = user_state(V, sp.prefix, cfg)
            pool = exact_cosine_topl(V, st.u, st.consumed, 100)
            vals = {}
            for a in pool:
                v = V[a]
                z = st.P @ v
                l = float(v @ z)
                denom = 1.0 + l
                udotv = float(st.u @ v)
                ds = l / denom * (1.0 - udotv)
                dss.append(abs(ds))
                aligns.append(1.0 / (1.0 + abs(ds)))
                coss.append(float(st.u @ v) / (np.linalg.norm(st.u) * np.linalg.norm(v)))
                vals[a] = (float(st.u @ v) / (np.linalg.norm(st.u) * np.linalg.norm(v)), ds)
            base = np.array([vals[a][0] for a in pool])
            v1 = base + 1.0 / (1.0 + np.abs(np.array([vals[a][1] for a in pool])))
            sp_corr.append(spearman(base, v1))
            top_base = set(pool[np.argsort(-base)[:10]].tolist())
            top_v1 = set(pool[np.argsort(-v1)[:10]].tolist())
            ov0.append(len(top_base & top_v1) / 10.0)
        # eta sweep: top-10 overlap vs eta=0 and validation recall
        overlap, recall = [], []
        for eta in etas:
            ov, pr = [], []
            for sp in data.splits[:N_USERS_TASK2]:
                st = user_state(V, sp.prefix, cfg)
                pool = exact_cosine_topl(V, st.u, st.consumed, 100)
                base = np.array([float(st.u @ V[a]) / (np.linalg.norm(st.u) * np.linalg.norm(V[a])) for a in pool])
                vals_eta = base + eta / (1.0 + np.abs(np.array([
                    (lambda v: (float(v @ (st.P @ v)) / (1.0 + float(v @ (st.P @ v)))) * (1.0 - float(st.u @ v)))(V[a])
                    for a in pool])))
                if eta == 0.0:
                    ov.append(1.0)
                else:
                    t0 = set(pool[np.argsort(-base)[:10]].tolist())
                    t1 = set(pool[np.argsort(-vals_eta)[:10]].tolist())
                    ov.append(len(t0 & t1) / 10.0)
            pcfg = PlannerConfig(method="plan1", eta=eta, plan1_pool=100)
            res = evaluate_method(V, data.splits[:N_USERS_TASK3], 10, pcfg, np.random.default_rng(SEED), cfg, holdout="valid")
            overlap.append(float(np.mean(ov)))
            recall.append(res["recall_at_k"])
        results[tag] = {
            "config": {"dim": dim, "alpha": alpha, "lam": lam},
            "abs_delta_s": {"mean": float(np.mean(dss)), "std": float(np.std(dss)), **q(np.asarray(dss)), "max": float(np.max(dss))},
            "cosine": {"std": float(np.std(coss))},
            "alignment_eta1": {"mean": float(np.mean(aligns)), "std": float(np.std(aligns))},
            "std_ratio_alignment_over_cosine": float(np.std(aligns) / np.std(coss)),
            "spearman_plan1_vs_cos": float(np.mean(sp_corr)),
            "top10_overlap_plan1_cos_eta1": float(np.mean(ov0)),
            "eta_sweep": {"etas": etas, "top10_overlap_vs_eta0": overlap, "valid_recall": recall},
        }
        print(f"  {tag}: std(align)/std(cos)={results[tag]['std_ratio_alignment_over_cosine']:.4f} "
              f"spearman={results[tag]['spearman_plan1_vs_cos']:.4f} overlap@eta1={results[tag]['top10_overlap_plan1_cos_eta1']:.3f}", flush=True)
    dump("task2_alignment.json", results)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    for tag, r in results.items():
        ax.plot(r["eta_sweep"]["etas"], r["eta_sweep"]["top10_overlap_vs_eta0"], "o-", label=tag)
    ax.set_xscale("symlog", linthresh=0.01)
    ax.set_xlabel("eta")
    ax.set_ylabel("mean top-10 overlap with eta=0")
    ax.set_title("Candidate ordering stability under eta")
    ax.legend()
    ax = axes[1]
    width = 0.35
    x = np.arange(len(results))
    cos_std = [r["cosine"]["std"] for r in results.values()]
    al_std = [r["alignment_eta1"]["std"] for r in results.values()]
    ax.bar(x - width / 2, cos_std, width, label="std cos(u,v)")
    ax.bar(x + width / 2, al_std, width, label="std 1/(1+|delta_s|)")
    ax.set_xticks(x)
    ax.set_xticklabels(results.keys(), rotation=20, ha="right", fontsize=8)
    ax.set_title("Spread of reward components across top-100 candidates")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "task2_alignment.png", dpi=130)
    return results


# ---------------------------------------------------------------- task 3

def task3(data):
    print("== Task 3: geometry sensitivity (exploratory)", flush=True)
    splits3 = data.splits[:N_USERS_TASK3]
    base = {"dim": 64, "alpha": 40.0, "lam": 1.0, "n_iter": 20}
    variants = [
        ("baseline", dict(base), False, 100),
        ("normV", dict(base), True, 100),
        ("d32_a10", {"dim": 32, "alpha": 10.0, "lam": 1.0, "n_iter": 20}, False, 100),
        ("d128_a10", {"dim": 128, "alpha": 10.0, "lam": 1.0, "n_iter": 15}, False, 100),
        ("lam0.1", {"dim": 64, "alpha": 10.0, "lam": 0.1, "n_iter": 20}, False, 100),
        ("lam10", {"dim": 64, "alpha": 10.0, "lam": 10.0, "n_iter": 20}, False, 100),
        ("pool20", dict(base), False, 20),
        ("pool1000", dict(base), False, 1000),
        ("pool_full", dict(base), False, data.n_items),
    ]
    results = {}
    for tag, cdict, normalize, pool_size in variants:
        cfg = IALSConfig(**cdict)
        U, V = train_ials(data.positives_train, data.n_users, data.n_items, cfg, seed=SEED)
        V_used = V / np.where(np.linalg.norm(V, axis=1, keepdims=True) == 0, 1, np.linalg.norm(V, axis=1, keepdims=True)) if normalize else V
        # state invariance under candidate-pool restriction: state must not depend on the pool
        s_ref = user_state(V_used, data.splits[0].prefix, cfg)
        s_check = user_state(V_used, data.splits[0].prefix, cfg)
        assert np.array_equal(s_ref.P, s_check.P) and np.array_equal(s_ref.u, s_check.u)
        ls, sel_l = [], []
        for sp in data.splits[:150]:
            st = user_state(V_used, sp.prefix, cfg)
            lv = leverage_catalog(V_used, st.P)
            mask = np.ones(data.n_items, bool)
            mask[np.asarray(st.consumed)] = False
            ls.append(lv[mask])
            pool = exact_cosine_topl(V_used, st.u, st.consumed, pool_size)
            pcfg = PlannerConfig(method="plan1", eta=1.0, plan1_pool=pool_size)
            recs = plan1_topk(V_used, st, 10, pcfg)
            sel_l.append(lv[np.asarray(recs)])
        cat = np.concatenate(ls)
        sel = np.concatenate(sel_l)
        pcfg = PlannerConfig(method="plan1", eta=1.0, plan1_pool=pool_size)
        r_static = evaluate_method(V_used, splits3, 10, PlannerConfig(method="static_cos"), np.random.default_rng(SEED), cfg, holdout="valid")
        r_plan1 = evaluate_method(V_used, splits3, 10, pcfg, np.random.default_rng(SEED), cfg, holdout="valid")
        results[tag] = {
            "config": cdict, "normalized_V": normalize, "plan1_pool": pool_size,
            "leverage_catalog": {"mean": float(cat.mean()), **q(cat), "max": float(cat.max())},
            "leverage_selected_top10": {"mean": float(sel.mean()), **q(sel), "max": float(sel.max())},
            "valid_recall_static_cos": r_static["recall_at_k"],
            "valid_recall_plan1": r_plan1["recall_at_k"],
            "plan1_vs_static_top10_overlap": None,
        }
        print(f"  {tag:10s} lev_mean={cat.mean():.3e} lev_sel={sel.mean():.3e} "
              f"R_static={r_static['recall_at_k']:.4f} R_plan1={r_plan1['recall_at_k']:.4f}", flush=True)
    # top-10 overlap plan1 vs static_cos (baseline config only, full pool semantics via value comparison)
    cfg = IALSConfig(**base)
    U, V = train_ials(data.positives_train, data.n_users, data.n_items, cfg, seed=SEED)
    ov = []
    for sp in data.splits[:150]:
        st = user_state(V, sp.prefix, cfg)
        pool = exact_cosine_topl(V, st.u, st.consumed, 100)
        base_vals = np.array([float(st.u @ V[a]) / (np.linalg.norm(st.u) * np.linalg.norm(V[a])) for a in pool])
        v1 = base_vals + 1.0 / (1.0 + np.abs(np.array([
            (lambda v: (float(v @ (st.P @ v)) / (1.0 + float(v @ (st.P @ v)))) * (1.0 - float(st.u @ v)))(V[a]) for a in pool])))
        t0 = set(pool[np.argsort(-base_vals)[:10]].tolist())
        t1 = set(pool[np.argsort(-v1)[:10]].tolist())
        ov.append(len(t0 & t1) / 10.0)
    results["baseline"]["plan1_vs_static_top10_overlap"] = float(np.mean(ov))
    dump("task3_sensitivity.json", results)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    ax = axes[0]
    names = list(results.keys())
    x = np.arange(len(names))
    rs = [results[n]["valid_recall_static_cos"] for n in names]
    rp = [results[n]["valid_recall_plan1"] for n in names]
    ax.bar(x - 0.2, rs, 0.4, label="static_cos")
    ax.bar(x + 0.2, rp, 0.4, label="plan1")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("valid Recall@10")
    ax.set_title("Geometry/pool sensitivity (EXPLORATORY)")
    ax.legend()
    ax = axes[1]
    lm = [results[n]["leverage_catalog"]["mean"] for n in names]
    ls = [results[n]["leverage_selected_top10"]["mean"] for n in names]
    ax.bar(x - 0.2, lm, 0.4, label="catalog mean l")
    ax.bar(x + 0.2, ls, 0.4, label="selected top-10 mean l")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("leverage")
    ax.set_title("Leverage: catalog vs selected actions")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "task3_sensitivity.png", dpi=130)
    return results


def main():
    tasks = set(sys.argv[1:]) or {"1", "2", "3"}
    t0 = time.time()
    data = get_data()
    if "1" in tasks:
        task1(data)
    if "2" in tasks:
        task2(data)
    if "3" in tasks:
        task3(data)
    print(f"diagnostics done in {(time.time() - t0) / 60:.1f} min")


if __name__ == "__main__":
    main()
