# AGENTS.md — planning_over_MF_MDP

Guidance for AI agents working in this repository. Read before modifying the project.

## Mission

Reproduce, then extend, **"Planning over Matrix-Factorization MDPs for Candidate Generation"**
(arXiv:2607.02115v2). The extension direction chosen by the owner: **learned reward models and
learned value models** on top of the MF-MDP (paper §5 future work: "learn ... values from logs",
"use a sequence model as a value function inside the MF-MDP"). Full brief:
`../Beyond_Optimistic_Planning_Full_Research_Prompt.md` (phases, RQs, infra, standards).

Do not jump ahead of Phase 1: the reproduction must exist and be tested before reward/value
models are evaluated against it.

## Paper ground truth (verified against 2607.02115v2.pdf)

- State `s = (P, u)`, `P = A^{-1}`, per-user `A = lam*I + sum_{i in I} c_i v_i v_i^T` (all catalog
  items; unobserved contribute baseline confidence), `b = sum_{i in H} c_i v_i`, `u = A^{-1} b`.
- Optimistic transition (unit-confidence positive fold-in), `z = P v`, `l = v' P v`:
  `P+ = P - z z' / (1+l)`, `u+ = u + z (1 - u'v) / (1+l)`; self-shift `Delta_s = l/(1+l) (1 - u'v)`.
- Reward, eq. (8): `r(s,a) = sim(u, v_a) + eta * 1/(1 + |Delta_s(a)|)`, trajectory reward
  `R(tau) = sum_k gamma^k r(s_k, a_k)`. sim is dot or cosine; all planner gains use cosine.
- UCT, eq. (10): `Q(s,a) + c_uct * sqrt(log N(s) * (2 + cos(u_s, v_a)) / (N(s,a)+1))`.
- MCTS: ANN top-`l` expansion near current `u`, closed-form leaf value, discounted
  return-to-go backprop. Readout: best trajectory by backed-up Q (Fig. 1b).
- Not reported in the paper (we choose + document, see docs/DECISIONS.md): MF hyperparameters,
  LLN n, gamma, eta, c_uct, expansion width, #simulations, leaf rho definition.

## Architecture

- `src/mf_mdp/mf.py` — iALS training + user fold-in (uses the all-items precision convention).
- `src/mf_mdp/mdp.py` — `MFState`, closed-form transition, reward, similarities. Match paper notation.
- `src/mf_mdp/planning/` — `static`, `plan1`, `mcts`. Candidate generation is exact top-l by
  cosine (catalogs here are <= ~5k; a real ANN index is unnecessary and would add a dependency).
- `src/mf_mdp/evaluation/metrics.py` — Recall@K, nDCG@K.
- `experiments/run_reproduction.py` — CLI entrypoint writing metrics to `outputs/`.
- `tests/` — math tests compare closed forms against direct recomputation. Keep them green.

## Conventions

- Python 3.11, NumPy core, no heavy deps for the reproduction. Type hints on public functions.
- English everywhere. Match the paper's symbols (A, P, u, v, l, Delta_s) in code.
- Never invent paper-reported values; log every assumption in docs/DECISIONS.md.
- No datasets/checkpoints/outputs/logs in git (`.gitignore` covers them).
- One codebase, local Mac = development, remote L4 server = compute (docs/INFRASTRUCTURE.md).

## Research branches (hypothesis lines beyond Phase 5)

Branch `h1/belief-vs-preference-dynamics`: separating epistemic belief updates (MF
posterior fold-in) from true preference dynamics in model-based planning. Design:
`docs/H1_BELIEF_VS_PREFERENCE_DESIGN.md`. Note: H1.x numbering there is unrelated to
the Phase-5 H1–H3 in `docs/PHASE5_HYPOTHESES.md`.

Branch `h2/hybrid-sequential-state`: hybrid belief state
`b_t = (P_t, u_t, h_t)` — MF posterior plus a GRU sequential state with
feedback-conditioned dynamics `p_theta(h_{t+1}, y_t | h_t, a_t)` — against the
MF-posterior-only state in a state x transition factorial (Env-C) with mandatory
negative controls (shuffle / no-response / random-action). Design:
`docs/H2_HYBRID_SEQUENTIAL_STATE_DESIGN.md`. "H2" here is unrelated to the Phase-5 H2
(learned reward/value). New code lives in `src/mf_mdp/sequential/` (numpy GRU + world
model), `src/mf_mdp/planning/hybrid.py`, `experiments/h2/`, `tests/test_h2.py`;
existing modules stay untouched so Phase 1/4/5 remain reproducible.

Branch `h3/sequential-action-proposals`: sequential model (SASRec, PyTorch) as an
ACTION PROPOSER, not a recommender — candidate pool `C_t = TopM p_theta(a|h_t)` and
MCTS searching only inside `C_t`; hypothesis about the quality–compute trade-off of
planning, not about final recommendation quality. Design:
`docs/H3_LEARNED_ACTION_PROPOSALS_DESIGN.md`. New code lives in
`src/mf_mdp/proposal/` (sasrec/pools/planning), `experiments/h3/`,
`tests/test_h3.py`; torch is optional (import-guarded); existing modules untouched.

Branch `h4/cross-response-model`: cross-feature response models — logistic baseline
vs LR + grounded interactions (owner's six) vs random-cross control vs FM vs MLP vs
MLP+u⊙v, on two levels (predictive: KuaiRand temporal split + randomized-exposure
subset, ML-1M; decision: closed-loop J in Env-C). **H4 REJECTED at both levels on
all three testbeds** (history features linearly carry the gain; grounded ≡ random
crosses ≡ 0; no J contrast CI95 excludes 0; DCN-v2 trigger not met, never built).
Design: `docs/H4_CROSS_RESPONSE_MODEL_DESIGN.md`. New code lives in
`src/mf_mdp/interactions/` (fm/mlp/cross/planner), `experiments/h4/`,
`tests/test_h4.py`; existing modules untouched.
