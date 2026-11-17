# MEMORY.md

## Current objective

H2 (hybrid sequential belief state) on branch `h2/hybrid-sequential-state`.
Design: `docs/H2_HYBRID_SEQUENTIAL_STATE_DESIGN.md`. (Phase 5 on
`phase5/stochastic-world-model` is complete: `docs/PHASE5_RESULTS.md`; separate H1
line on `h1/belief-vs-preference-dynamics`.)

## Status (updated 2026-11-17)

- H2-E1 factorial done on Env-C (5 seeds): state {mf,seq,hybrid} x transition
  {optimistic,stochastic}, depth-2 matched budget, q_hat from logs.
  **H2 REJECTED at this scale.** J: mf+stochastic 0.227 best learned arm; hybrid+stochastic
  0.148 (hybrid-mf = -0.080 CI95 [-0.155,-0.004]); seq worst (0.097) despite next-item
  recall TIED with mf (0.384 vs 0.384) — seq response model much worse (AUC 0.577 vs
  0.734, ECE 0.073 vs 0.013). Negative controls barely move J (seq no_response +0.018) —
  order/feedback/action conditioning does not drive planning value here. Caveat: GRU
  trained from scratch on 400x6 events vs iALS-structured MF features (inductive-bias
  confound); alpha-sweep and E-init-from-iALS are the cheap follow-ups.
- Surprise vs Phase 5: stoch-opt@mf = +0.071 [+0.027,+0.115] under q_hat-from-logs,
  while Phase 5 measured -0.004 under TRUE env.q — the stochastic-transition advantage
  appears when item quality is estimated, not given. Worth a q_hat ablation before
  claiming.
- Server still SSH-unreachable (since ~2026-11-12); H2 is CPU-scale anyway.
- Tests: tests/test_h2.py 9/9 green; pre-existing UNRELATED failure
  tests/test_h1.py::test_greedy_differs_from_oracle_in_envP (H1 pilot open item, not
  touched by H2).

## Next steps

1. q_hat ablation (true-q vs log-estimated) to isolate the stoch-opt sign flip.
2. alpha/horizon sweep for H2 (regime boundary, per H2 design Part 6 risk 3).
3. Optional: E initialized from iALS V (project dim) to reduce the inductive-bias gap.
4. H2 real-data check on KuaiRand sessions only if (1)-(2) show any signal.

## Open questions

- Whether ANY sequential state can beat the MF posterior for PLANNING (not prediction)
  in the 400-user data regime, or whether the MF posterior is simply the right statistic
  under optimism-corrected fold-ins.
- Whether the H2 mechanism claim survives in a popularity-driven env (controls C1-C3
  were built to catch popularity smuggling; with no positive gain they are moot).
