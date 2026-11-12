# MEMORY.md

## Current objective

Phase 4 (reward/value foundation) on branch `phase4/reward-value-v1`. Prompt:
`../Phase4_Reward_Value_Model_Execution_Prompt.md`. Results: `docs/PHASE4_RESULTS.md`.

## Status (updated 2026-11-12)

- Phase 0 + Phase 1 (ML-1M LLN) committed on main (140da9d) and pushed; diagnostics on
  main (fb0411b). Phase 4 work is on branch `phase4/reward-value-v1` (NOT pushed — needs
  owner approval per prompt §2).
- Reward audit DONE (gate): MCTS paper-exact; Plan-1 'post' = post_transition variant;
  canonical 'paper'; full-catalog crash = consumed-items bug in exact_cosine_topl (fixed,
  regression test; addendum corrects the diagnostics report).
- Phase 4 experiments DONE: response model AUC 0.62 calibrated (ECE 0.007); learned
  rewards R1/R2 LOSE to paper R0 on ML-1M (weak 4-vs-5 signal, popularity-flavored);
  V_logging base R2=0.019, horizon variant R2=0.989 (non-deployable interpolator);
  E4: all MCTS leaf evaluators tie at R@10=0.0733, V_logging leaf ~10x cheaper than
  greedy rho. 42/42 tests.

## Next steps (owner decision)

1. Push branch after owner review.
2. KuaiRec (watch-ratio labels) — prerequisite for meaningful R1/R2.
3. V_logging v2: recency/engagement features or sequence model; then V_target via
   planner-rollout distillation in a validated model (still open, documented gap).
4. Optional: GTS split implementation for RQ5.

## Open questions

- Whether learned rewards can beat R0 anywhere (needs graded engagement labels).
- Extrapolation of V_logging to MCTS optimistic states: unvalidated; measure feature-
  space discrepancy before trusting leaf values.
