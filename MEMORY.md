# MEMORY.md

## Current objective

H4 (cross-feature response models) — line COMPLETE, hypothesis REJECTED at both
levels, on branch `h4/cross-response-model`.
Design: `docs/H4_CROSS_RESPONSE_MODEL_DESIGN.md`.
Open lines: H1 (`h1/belief-vs-preference-dynamics`, pilot sign-off open), H2
(**rejected at every scale/dynamics regime**, E1+E2, see the h2 branch Parts 7-8), H3
(**COMPLETE**, branch `h3/sequential-action-proposals` @ c376cd1: H3-b SUPPORTED —
MF pools ≡ full-catalog policy exactly at ~half latency; H3-a SUPPORTED w/ twist —
sasrec@M50 R@10 .1079 vs full .0696 at LOWER J, J anti-correlated w/ R@10; H3-c SPLIT —
union coverage super-additive but ≡ MF at policy level). Phase 5 complete
(`docs/PHASE5_RESULTS.md`). Consolidated report: `docs/H1_H4_HYPOTHESES_REPORT_RU.md`.

## Status (updated 2026-11-20)

- **H4 big server runs DONE (2026-11-18, fetched 2026-11-20): the null is not a scale
  artifact.** Env-C at 10 seeds x 2000 train / 1000 eval: every cross arm == lr to 4
  decimals (lr_x -0.0001 [-0.0034,+0.0034]); FM/MLP stay below LR with 5x data on both
  KuaiRand (test n=5,056: fm 0.7214, mlp 0.7097 vs m0h 0.7429) and ML-1M — the feature
  set is at its information ceiling. Grounded==random==0 replicates exactly.
- The single significant J mover is mlp_uv: +0.0071 [+0.0002,+0.0156], 8/10 seeds +,
  driven by seeds 3/6, zero predictive gain anywhere — logged as fragile
  transmission-decoupling, NOT H4 support (rule needs both levels). Artifacts:
  outputs/h4/server_big/{env_c_cross_big,kuairand_cross_big}.json
- IMPORTANT CROSS-LINE OBSERVATION: lr-optimistic = +0.018 [-0.002,+0.039] at 2000
  train users (10 seeds) — the H2 stoch-opt J signal (+0.071 @400 users, our +0.045)
  does NOT replicate at 5x scale. The leading hypothesis: the stochastic arm's
  advantage lived in q_hat estimation noise and vanishes as q_hat improves. This
  weakens the "transitions are the bottleneck" narrative from MEMORY.md's next-steps.
- Big-picture state: three lines (Phase-4 leaf, H2 state, H4 response/crosses) are all
  null at every tested scale; the stochastic-vs-optimistic J signal itself is
  scale-fragile. The bottleneck is NOT any single replaceable component at these scales.
- WIP snapshot commit 3611c13 on `h3/sequential-action-proposals` protects the
  previously uncommitted h1/h2/h3 tree; h4 pushed to origin (84e8045 + docs commit).
- Server: h4 big runs done and fetched; 9 h3 tmux sessions were running on 2026-11-18
  (owner's, untouched).

## Next steps

1. q_hat ablation (true-q vs log-estimated) to isolate the stoch-opt sign flip
   (still open from H2; now the single most consistent J-moving lever).
2. H3 continues on its branch (proposals WIP): candidate-bottleneck metrics,
   union pools, fixed-budget J comparison.
3. H1 pilot sign-off (steering tension CI covers 0 at pilot scale).
4. If a response-model line is ever revived: the H4 ladder code (interactions/)
   is reusable; the null says invest in labels/coverage, not capacity.

## Open questions

- Whether ANY response-model or state upgrade can move Env-C J at 400-user
  scale, or whether J is transition-dominated (q_hat / stochastic branching) in
  this regime — three lines (Phase-4 leaf, H2 state, H4 response) all null.
- Whether the KuaiRand null is coverage-bound: pool-only scoring leaves 1.8k
  test rows; a min-exposure filter (e.g. >=10 train exposures, trained
  embeddings) would give ~10-50x more test rows — exploratory, not pre-registered.
- Whether FM/MLP would beat LR at true Gate-E server scale (5M rows; the
  overfitting gap was monotone in training rows 2.4k<31.7k).
