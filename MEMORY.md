# MEMORY.md

## Current objective

Reproduce arXiv:2607.02115v2 (MF-MDP planning for candidate generation), then build the
owner-chosen extension: learned reward + value models (Phase 4 of the brief; design in
docs/PHASE4_DESIGN.md).

## Status (updated 2026-11-11)

- Phase 0 done: repo, venv (numpy/pyyaml/pytest/pypdf), docs, configs, sync scripts;
  remote server reachable (adrianna, L4 24GB, driver 615.71.09, documented in INFRASTRUCTURE.md).
- Phase 1 done for ML-1M LLN: Static/Plan-1/Plan-K implemented; 12/12 math+structure tests green
  (SM vs direct inverse, fold-in vs direct WLS, paper b+=b+v semantics, synthetic e2e).
- MF HPs unreported in paper -> validation sweep picked d=64, alpha=40, lambda=1.0, n_iter=20.
- KEY FINDING (small-leverage regime): leverage l = v'Pv is bounded ~ d/n_items for spread
  embeddings (measured 0.004-0.05), so the reward alignment term is nearly constant and
  Plan-1 degenerates to cosine static ranking in every config we can reach.
- ML-1M LLN test (3 seeds, 6035 users): static_dot .0704 (paper .0694 REPRODUCED),
  static_cos .0696, plan1 .0694 (paper .0762 NOT reproduced); Plan-K 2000-user subset
  .0685 = static_cos, no gain over Plan-1 at ~680x latency. See docs/REPRODUCTION.md.

## Next steps

1. Collect Plan-K subset result; fill REPRODUCTION.md table.
2. Reproduction verdict discussion with owner: accept "Static reproduced, planning gain not
   reproduced in reachable regime" or invest in regime-changing variants (normalized V,
   per-user top-N candidate restriction) — candidates listed in REPRODUCTION.md.
3. Begin Phase 4 (owner's direction): value-model v1 per docs/PHASE4_DESIGN.md
   (ridge on state features vs closed-form rho as MCTS leaf; depth x error study).
4. Optional speedup: vectorize MCTS fold-in batches (10-20x) to afford full-set Plan-K.

## Open questions

- Plan-1 interpretation ambiguity ('post' canonical; 'root' marginally better on validation
  at chosen HPs but within noise).
- Whether paper's regime (planning gain + cos>>dot) is reachable with any documented iALS
  variant — their MF HPs are unreported, so this may be unanswerable.
- GTS protocol implementation pending (needed for RQ5 later).
