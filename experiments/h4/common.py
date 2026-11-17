"""Shared builders for the H4 Env-C decision-level experiment.

The trajectory logging / behavior policy / q_hat pieces are reused verbatim from
experiments/h2/run_h2_factorial.py (same protocol, same env) so the H4 arms differ
from the H2 mf+stochastic arm ONLY in the response-model input/features — the
matched-budget convention of the H2 factorial is preserved by construction.
"""

from __future__ import annotations

import numpy as np

from mf_mdp.mf import IALSConfig, user_state
from mf_mdp.models.features import features_for


def behavior_policy(env, theta, f, consumed, rng):
    avail = [i for i in range(env.n_items) if i not in consumed] or list(range(env.n_items))
    if rng.random() < 0.3:
        return int(rng.choice(avail))
    return max(avail, key=lambda i: env.q[i] * float(theta @ env.V[i]))


def log_trajectories(env, thetas, rng):
    trajs = []
    for theta in thetas:
        th, f, consumed = theta.copy(), 0, set()
        traj = []
        for _ in range(env.cfg.H):
            i = behavior_policy(env, th, f, consumed, rng)
            r, term, th, nf, engaged = env.step(th, f, i, rng)
            traj.append((int(i), int(engaged), int(term)))
            consumed.add(i)
            if term:
                break
            f = nf
        trajs.append(traj)
    return trajs


def estimate_q_hat(trajs, n_items, shrink=5.0):
    exp = np.zeros(n_items)
    eng = np.zeros(n_items)
    for traj in trajs:
        for (i, e, _t) in traj:
            exp[i] += 1
            eng[i] += e
    global_mean = eng.sum() / max(exp.sum(), 1.0)
    return (eng + shrink * global_mean) / (exp + shrink)


def envc_feature_rows(trajs, V, ials_cfg, log_pop):
    """Per decision-point raw feature components (decision-time legal).

    Row i of the output corresponds to the i-th (traj step): the state is the
    fold-in of the prefix BEFORE that step; hist features use only events before
    it. Same walk as the H2 mf-arm training, plus the raw components the H4
    feature modes need (kept separate so arms assemble their own vectors).
    """
    X_base, mean_eng, repeat, uv, y, y_term = [], [], [], [], [], []
    for traj in trajs:
        hist, events = [], []
        st = user_state(V, np.asarray(hist, dtype=np.int64), ials_cfg)
        for (i, e, term) in traj:
            X_base.append(features_for(st, V, i, log_pop))
            mean_eng.append(float(np.mean([ee for _, ee in events])) if events else 0.0)
            repeat.append(float(hist.count(i)))
            uv.append(st.u * V[i])
            y.append(float(e))
            y_term.append(float(term))
            hist.append(i)
            events.append((i, float(e)))
            st = user_state(V, np.asarray(hist, dtype=np.int64), ials_cfg)
    return {
        "base": np.vstack(X_base),
        "mean_eng": np.asarray(mean_eng),
        "repeat": np.asarray(repeat),
        "uv": np.vstack(uv),
        "y": np.asarray(y),
        "y_term": np.asarray(y_term),
    }


def assemble_envc(rows, mode, rand_crosses=None):
    """Assemble the model-input matrix for an H4 feature mode (train or predict).

    Must stay exactly consistent with CrossPlanner.features() — the parity tests
    lock the 'base' mode; the other modes are assembled by the same rule here
    and in the planner.
    """
    base = rows["base"]
    if mode == "base":
        return base
    bh = np.column_stack([base, rows["mean_eng"], rows["repeat"]])
    if mode == "baseh":
        return bh
    if mode == "cross":
        # envc_crosses: [cos*hist_len, mean_eng*cos, leverage*log_pop, repeat*cos]
        # (vectorized; identical formula to the planner-time envc_crosses call)
        c = np.column_stack([bh[:, 0] * bh[:, 6],
                             rows["mean_eng"] * bh[:, 0],
                             bh[:, 2] * bh[:, 5],
                             rows["repeat"] * bh[:, 0]])
        return np.column_stack([bh, c])
    if mode == "randcross":
        return np.column_stack([bh, rand_crosses.transform(bh)])
    if mode == "uv":
        return np.column_stack([bh, rows["uv"]])
    raise ValueError(mode)
