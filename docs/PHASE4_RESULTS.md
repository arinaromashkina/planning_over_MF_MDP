# Phase 4 results — reward/value foundation

Branch `phase4/reward-value-v1`. Plan: `docs/PHASE4_EXECUTION_PLAN.md`. Contracts:
`docs/PHASE4_DESIGN.md` (response / utility / V_logging / V_target separation enforced).
Prompt: `../Phase4_Reward_Value_Model_Execution_Prompt.md`.
Artifacts: `outputs/phase4/*.json`, `checkpoints/v_logging_ridge_*.npz`.
All HPs selected on validation only; ML-1M LLN; seed 0 unless noted.

## 0. Reward-semantics audit (the §4 gate) — outcome

- Plan-K/MCTS edge rewards and leaf rho were already paper-exact (relevance at the
  edge-origin state). Plan-1's pre-audit canonical variant ('post') used post-transition
  relevance `sim(u+, v)` — a deviation, now renamed `post_transition`; the canonical
  Plan-1 is `paper` (exact eq. (8)); aliases preserve old configs; old results untouched.
- Monotonicity in leverage ℓ (fixed residual q > 0), now test-locked
  (`tests/test_reward_audit.py`): paper alignment `1/(1+|Δ|)` **decreases** in ℓ;
  post-transition relevance **increases** in ℓ; paper reward decreases in ℓ.
- CORRECTION to the diagnostics report (dated addendum in PLANNING_DIAGNOSTICS.md):
  the full-catalog Plan-1 crash (0.044) was a candidate-generation bug — consumed items
  leaked into pools when `l ≥ n_unconsumed` in `exact_cosine_topl` (fixed, regression
  test). After the fix full-catalog Plan-1 = 0.0860 = static_cosine. The "popularity
  amplification" attribution is retracted for the paper reward; it remains a property
  of the `post_transition` variant only.
- Component correlations (repro config, 300 users): corr(uᵀv, ‖v‖)=+0.29,
  corr(uᵀv, popularity)=+0.48, corr(uᵀv, ℓ)=+0.19, corr(uᵀv, alignment)=+0.23.

## 1. E2 — Response model p(y=rating 5 | consumed, s, a)

Labels: consumed interactions only (ML-1M has no exposure fields — no skip/dislike/
termination labels exist; response types beyond rating-4/5 marked INFEASIBLE, not
fabricated). Logistic regression on 10 decision-time features; user-disjoint split.

| split | n | base rate | log-loss | Brier | ROC-AUC | ECE |
|---|---|---|---|---|---|---|
| valid | 29,729 | 0.384 | 0.6456 | 0.2270 | 0.617 | 0.0068 |
| test | 29,294 | 0.388 | 0.6454 | 0.2270 | 0.623 | 0.0064 |

Calibration is good (ECE ≈ 0.007); discrimination is weak (AUC ≈ 0.62). Predicting
4-vs-5-stars from MF-state features is intrinsically hard.

## 2. E1 — Reward ablation with Plan-1 (validation users, 500)

| variant | valid Recall@10 |
|---|---|
| R0 paper heuristic (baseline) | **0.0740** |
| R1 u4=0.0/0.25/0.5, eta=0 | 0.0660 |
| R1 u4=0.25/0.5, eta=1 | 0.0680 |
| R2 direct learned expected utility | 0.0420 |

Selected on validation: **R0**. Test (R0): Recall@10 0.0740, nDCG@10 0.0372.
Learned expected-utility rewards UNDERPERFORM the paper heuristic on ML-1M: the weak
response signal (AUC 0.62) dilutes relevance with popularity-flavored satisfaction
prediction. R1+alignment(eta=1) slightly beats R1 without it — consistent with the
diagnostics (alignment is a small, mostly inert, relevance-correlated tiebreaker).

## 3. E3 — V_logging (logging-policy value; NEVER V_target)

MC return-to-go of normalized R0 rewards, γ=0.9, last-30 decision points per user;
ridge; user-disjoint 70/15/15; normalization on train; λ on validation.

| variant | test MAE | test RMSE | test R² | note |
|---|---|---|---|---|
| base (u + state summary) | 1.555 | 1.866 | **0.019** | ≈ global-mean predictor |
| horizon (+remaining, mean/last reward) | 0.162 | 0.201 | **0.989** | offline-window interpolation |

Honest reading: user-state features alone barely predict future returns (R² 0.02).
The horizon variant's R² 0.99 comes mostly from `log1p(remaining steps)` — a quantity
NOT knowable at a real decision point (session length unknown). The horizon variant is
a within-logged-window interpolation device; its MCTS leaf use (remaining = horizon −
depth) is a documented approximation, not validated extrapolation.

## 4. E4 — MCTS leaf ablation (300 test users, seed 0, 50 sims) — diagnostic, not policy proof

| leaf evaluator | horizon | Recall@10 | latency/user |
|---|---|---|---|
| original (greedy rho) | 10 | 0.0733 | 479 ms |
| zero | 10 | 0.0733 | 32 ms |
| v_logging_ridge (horizon) | 10 | 0.0733 | 48 ms |
| original | 5 | 0.0733 | 472 ms |
| v_logging_ridge | 5 | 0.0733 | 47 ms |
| original | 1 | 0.0733 | 35 ms |
| v_logging_ridge | 1 | 0.0733 | 7 ms |

Findings: in the small-leverage regime the leaf evaluator changes COMPUTE by an order
of magnitude but not QUALITY — all variants tie at 0.0733 (even zero). The greedy-rho
leaf dominates Plan-K's cost; V_logging leaf is ~10× cheaper. H6 (learned value reduces
required depth/sims at comparable quality) is technically supported on the compute side
(v_logging leaf ≈ original quality at ~1/10 leaf cost) but vacuous on quality: nothing
to preserve because the planning layer is quality-inert in this regime. Extrapolation of
V_logging to MCTS-generated (optimistic, off-logged) states is unvalidated — the model
was fit on logged states only.

## 5. Offline-evaluation limitations (mandatory)

- Response model and V_logging are conditioned on the CONSUMPTION support of the logging
  policy; exposures and rejections are unobserved (no IPS/SNIPS/DR possible on ML-1M —
  no exposure logs, no randomized subset).
- E4 is a model/leaf-evaluation diagnostic, not evidence of policy improvement.
- V_logging ≠ V_target by construction; target-policy value remains unimplemented
  (gap documented per prompt §6.1).
- Sequences are right-censored at the LLN cut; "session termination" does not exist in
  the data.

## 6. Conclusions

1. The reward-semantics audit is the phase's key correction: MCTS was paper-exact; the
   Plan-1 deviation is now named/tested; the diagnostics' crash attribution was a bug,
   fixed and regression-tested.
2. On ML-1M, learned reward models (R1/R2) do NOT beat the paper heuristic — the only
   available feedback (4-vs-5 stars on consumed items) is too weak (AUC 0.62) and
   popularity-flavored. Feasible richer labels (watch ratio on KuaiRec) are the
   prerequisite for reward modeling to matter.
3. V_logging v1 is honest but nearly uninformative on deployable features (R² 0.02);
   the horizon variant is a strong but non-deployable interpolator (R² 0.99).
4. Leaf-value choice affects compute, not quality, in this regime (all tie; V_logging
   leaf ~10× cheaper than greedy rho).
5. Next: KuaiRec (watch-ratio labels) for R1/R2; better V features (recency-weighted
   engagement) or sequence models for V_logging; only then target-policy value via
   planner-rollout distillation in a validated model.

## Reproduce

```bash
.venv/bin/python experiments/phase4/run_reward_ablation.py --seed 0
.venv/bin/python experiments/phase4/run_value_model.py --seed 0            # base
.venv/bin/python experiments/phase4/run_value_model.py --seed 0 --horizon-features
.venv/bin/python experiments/phase4/run_leaf_ablation.py --variant horizon --n-users 300
```
