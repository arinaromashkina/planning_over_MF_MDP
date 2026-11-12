# MEMORY.md

## Current objective

Phase 5 (stochastic world model for sequential recommendation) on branch
`phase5/stochastic-world-model`. Prompt: `../Phase5.md`. Hypotheses/plan:
`docs/PHASE5_HYPOTHESES.md`, `docs/PHASE5_EXPERIMENT_PLAN.md` (frozen protocol).

## Status (updated 2026-11-12)

- Gate A done: audit + literature review + hypotheses + formal problem + positioning.
- Gate B done: Env-S (12-state discrete, exact DP oracle); greedy != DP; MCTS converges
  (residual = near-ties). DP oracle bug (assimilation failure) found via MCTS, fixed.
- Gate C/D done for Env-S: response model AUC 0.75; **H1 SUPPORTED in Env-S**:
  stochastic-optimistic = +0.122 CI95 [0.115, 0.129]; oracle regret 0.006. H3: only
  full-horizon planning reaches oracle; model accuracy not binding (250-3000 traj).
- Env-C (MF-flavored) done: **H1 NOT supported there** — stochastic ≈ optimistic
  (-0.004); planning > greedy_mf by +0.041 (underpowered); all MF planners far below
  true-θ greedy (0.685) — representation-bound. Contrast isolates state quality as the
  binding requirement.
- Gate E (KuaiRand-1K) first part done: loader + sessionization (30-min gap, documented),
  chronological split, response models. wr80-on-clicks AUC 0.839/ECE 0.043 (transfers to
  randomized exposure: 0.826); click AUC 0.607 but ECE 0.234 (temporal drift) and random-
  subset AUC 0.515 (policy-bound). Selection bias in standard logs confirmed by random file.
- Server NOT needed so far (all CPU-scale); KuaiRand full-scale work may need it.
  NOTE: server SSH unreachable from the Mac since ~14:00 (user informed).

## Next steps

1. KuaiRand-1K inspection (fields, exposure semantics, splits) when download completes.
2. Env-S harder response regime (observation noise) for the error-vs-depth curve (H3).
3. PHASE5_RESULTS.md after Gate E; studio deliverables only after all gates.

## Open questions

- Whether any MF-posterior regime can make stochastic planning matter (Env-C says no at
  alignment 0.6 — test with stronger iALS training, e.g., 2000+ users).
- H3 error-vs-depth interaction needs a regime where the response model is
  genuinely hard to learn.
