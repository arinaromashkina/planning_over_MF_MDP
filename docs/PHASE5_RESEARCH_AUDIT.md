# Phase 5 research audit (mandatory, prompt §1.2)

Branch `phase5/stochastic-world-model` created from `phase4/reward-value-v1` @ `3c432fd`.
42/42 tests pass. All artifacts verified on disk (reproduction metrics, diagnostics
JSONs + plots, phase4 JSONs + V_logging checkpoints).

## What is implemented and verified

- iALS + fold-in with the paper's all-items precision convention; closed-form
  Sherman–Morrison transition (paper eq. 13 semantics b+=b+v) and the general weighted
  transition (alpha_w, beta_w) == exact re-fold at (alpha, 1+alpha). Test-verified
  against direct WLS/inverse recomputation.
- Static / Plan-1 (paper-exact, post_transition, bellman) / Plan-K MCTS (cosine-prior
  UCT eq. 10, ANN-restricted expansion, closed-form greedy leaf, discounted backprop).
- ML-1M LLN pipeline (n_test=1, n_valid=1), Recall/nDCG@10, HP-sweep runner.
- Phase 4: logistic response model (with AUC/Brier/ECE metrics), R1/R2 reward models,
  V_logging ridge (base + horizon variants), opt-in MCTS leaf_evaluator
  (original|zero|heuristic|v_logging_ridge), reward_fn override for Plan-1.

## What has NOT been reproduced

- The paper's planning gains (Plan-1/Plan-K > Static) on ML-1M LLN: our static_dot
  .0704 ~= paper .0694 (reproduced), but Plan-1 .0694 < paper .0762 (not reproduced).
- Root cause identified in diagnostics: leverage v'Pv is a popularity proxy bounded
  ~d/n_items; the paper alignment term is inert where embeddings are good.

## Which Phase 4 findings are reliable

- Reward-semantics audit: MCTS paper-exact; Plan-1 'post' was a variant; the
  full-catalog crash was a candidate-generation bug (fixed, regression-tested).
- Response model: AUC ~0.62, ECE ~0.007 on ML-1M (consumed-only labels; weak signal).
- R1/R2 learned rewards LOSE to the paper heuristic on ML-1M (validation-selected).
- V_logging base R²≈0.019; horizon variant R²≈0.989 is a non-deployable interpolator
  (remaining-steps feature is not knowable at decision time).
- E4: MCTS leaf evaluators all tie at R@10=0.0733 (subset, seed 0); V_logging leaf is
  ~10x cheaper than greedy rho. Compute effect only, in the inert regime.

## Dataset-limited conclusions

- All Phase 4 conclusions are ML-1M-specific: rating-4/5-only labels, no exposure, no
  watch ratio, no session boundaries. KuaiRand is required for meaningful
  response/dynamics labels (randomized exposure subset enables one-step causal checks).

## Reusable for Phase 5

- Transitions (both forms), reward(), planners + MCTS skeleton, metrics, evaluation
  runner, experiment/artifact logging, models/ (response/reward/value), tests
  infrastructure, sync/remote scripts.
- V_logging checkpoint + horizon-feature machinery (as the logging-policy baseline).

## Assumptions to REMOVE in Phase 5

- "Optimistic positive fold-in is a reasonable transition" — this is exactly what H1
  tests against a true stochastic environment.
- "User embedding updates = preference changes" — latent state changes are model
  representations, not causal preference shifts (prompt §4).
- "Offline Recall@K on logged data is the objective" — Phase 5 primary metric is
  expected cumulative utility in a controlled environment with a true simulator.
- "Learned reward/value components automatically help" — Phase 4 showed they do not on
  ML-1M; Phase 5 tests them factorially in a controlled setting.
- Do NOT reintroduce the retracted interpretation (paper alignment amplifies
  popularity); the addendum in PLANNING_DIAGNOSTICS.md is authoritative.
