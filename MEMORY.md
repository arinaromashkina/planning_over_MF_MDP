# MEMORY.md

## Current objective

H4 (cross-feature response models) — line COMPLETE, hypothesis REJECTED at both
levels, on branch `h4/cross-response-model`.
Design: `docs/H4_CROSS_RESPONSE_MODEL_DESIGN.md`.
Open lines: H1 (`h1/belief-vs-preference-dynamics`, pilot sign-off open), H2
(rejected on Env-C at pilot scale), H3 (`h3/sequential-action-proposals`, WIP
snapshot committed 2026-11-17). Phase 5 complete (`docs/PHASE5_RESULTS.md`).

## Status (updated 2026-11-17)

- **H4 REJECTED at both levels, three testbeds, pre-registered criteria.**
  (a) Predictive: history features LINEARLY explain the entire gain — ML-1M
  m0h−m0 dAUC +0.056 CI95 [+0.056,+0.067] (n=29,294 test), KuaiRand m0h test AUC
  0.7474 vs m0 0.5621; grounded crosses m1−m0h +0.0007 [+0.0002,+0.0011] and
  RANDOM crosses c1−m0h +0.0008 [+0.0005,+0.0011] — grounding adds nothing;
  FM significantly worse (−0.0013); per-position slice (incl. cold-start p0)
  flat. (b) Decision (Env-C, true J): no family beats lr with CI95 excluding 0
  (best lr_x +0.013 [−0.002,+0.038], fm +0.016 [−0.013,+0.055]). Owner caveat
  strengthened: fm had the WORST holdout calibration (ECE 0.052) yet the highest
  J point estimate — predictive and planning gains decouple even in sign (ns).
  DCN-v2 trigger not met (Env-C: mlp valid 0.3293 > m1 0.3252; ML-1M equal) → M4
  never built. Sanity: lr−optimistic +0.045 [+0.002,+0.088] reproduces H2's
  stoch-opt signal; m0 click AUC 0.5915 = Gate-E repro; m0 ML-1M AUC 0.6231 =
  Phase-4 E2 repro. Artifacts: outputs/h4/{env_c_cross,ml1m_cross,kuairand_cross}.json
- Big picture across lines: response-model capacity/crosses (H4), sequential state
  (H2), and leaf values (Phase 4 E4) ALL leave Env-C J ~unchanged at 400-user
  scale; only stochastic-vs-optimistic transitions moved J (+0.045..+0.071 under
  q_hat-from-logs). The bottleneck is upstream of the response model.
- WIP snapshot commit 3611c13 on `h3/sequential-action-proposals` protects the
  previously uncommitted h1/h2/h3 tree (all three branch pointers were == main
  ec4ac6c before that).
- Server still SSH-unreachable (since ~2026-11-12); H4 was entirely CPU-scale
  (full KuaiRand run 12 min, Env-C 2 min, ML-1M 30 s).
- Tests: tests/test_h4.py 9/9 green (FM pair-term vs brute force, FM/MLP finite-
  difference gradients, determinism, cross builders, RandomCrosses freezing,
  CrossPlanner base-mode parity with HybridPlanner); suite 92 passed + the
  pre-existing unrelated tests/test_h1.py::test_greedy_differs_from_oracle_in_envP.

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
