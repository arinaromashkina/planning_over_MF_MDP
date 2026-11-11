# MEMORY.md

## Current objective

Reproduce arXiv:2607.02115v2 (MF-MDP planning for candidate generation), then build the
owner-chosen extension: learned reward + value models (Phase 4 of the brief; design in
docs/PHASE4_DESIGN.md).

## Status (updated 2026-11-11, evening)

- Phase 0 + Phase 1 (ML-1M LLN) committed (140da9d) and pushed; verdict in REPRODUCTION.md:
  Static reproduced, planning gains not reproduced in reachable regimes.
- DIAGNOSTIC PHASE DONE (docs/PLANNING_DIAGNOSTICS.md): leverage is a popularity proxy
  (Spearman up to 1.0); alignment term inert in good configs, harmful when active;
  no planning gain in 9 geometry variants; full-catalog Plan-1 crashes via self-boost
  popularity amplification (0.086 -> 0.044). No meaningful planning regime found.
- Phase-4 prep done: PHASE4_DESIGN.md has explicit labeling contracts (response model /
  utility / V_logging / V_target). Phase 4 NOT started (awaiting owner review).

## Next steps (owner decision)

1. Review docs/PLANNING_DIAGNOSTICS.md, then start Phase 4 (value-model v1: ridge MC-return
   regression on MF-state features as MCTS leaf; measure depth reduction H6).
2. Optional: vectorized MCTS fold-in batches (10-20x speedup) to afford full-set Plan-K.

## Open questions

- Whether any iALS variant outside the documented family could open a planning regime —
  diagnostics suggest the mechanism (popularity-proxy leverage) is structural; pursuing
  unknown paper HPs further is not time-boxed.
- GTS protocol implementation pending (needed for RQ5/Phase 5).
