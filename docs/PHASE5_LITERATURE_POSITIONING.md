# Phase 5 literature positioning (Gate A)

See `LITERATURE_REVIEW_PHASE5.md` for the full table. Summary of what is known, what
overlaps, and what this work adds.

## Already known (not our claims)

- Learned stochastic environments/reward/value models for recommendation (MCRL, ROLeR,
  R3S, WMG-RL, Session-Level REINFORCE 2026).
- Simulators as evaluation environments (KuaiSim, RecSim NG, KuaiEnv/CIRS).
- Chance-node stochastic search (Stochastic MuZero); pessimism under model error (MOReL);
  value-equivalence as a modeling principle (Grimm et al.).
- Learned feedback models + test-time rollout scaling for retrieval (arXiv:2607.02818) —
  the closest to our setting.

## What overlaps

- Session-Level REINFORCE (2607.02818): user-feedback model + simulated futures at
  inference. We share the "simulate responses at test time" idea; we differ in tree
  search over a posterior state, chance-node branching, and the controlled-environment
  oracle evaluation.
- KuaiSim/CIRS: KuaiRand-based evaluation; we use KuaiRand-1K as real-data validation,
  not as the evaluation environment itself.
- Stochastic MuZero: chance-node backup; we adapt the pattern to a tiny analytic+learned
  hybrid, not a full MuZero.

## What this work adds (defensible gap)

1. **First controlled-environment measurement of optimistic-vs-stochastic planning over
   the analytic iALS posterior.** The starting paper's optimistic fold-in is shown (by our
   reproduction + diagnostics) to be quality-inert in practice; we test whether
   feedback-conditioned stochastic dynamics — with the exact weighted fold-in — creates
   genuine planning value where the optimistic model has none, evaluated against a TRUE
   environment with an exact DP oracle.
2. **Hybrid analytic-learned dynamics**: learned stochastic response model + exact
   Sherman–Morrison weighted transitions (feedback-conditioned, per-response-type
   α/β). No published work combines closed-form posterior updates with a learned
   stochastic response model for planning.
3. **Error-vs-depth accounting (H3)**: regret as a function of (response-model error,
   stochasticity, planning depth) — the evidence base for adaptive planning depth, which
   the closest works do not provide.
4. **A negative-result-safe protocol**: pre-registered metrics, matched budgets, oracle
   validation, and explicit labeling of logging-policy value vs target-policy value.

## What we explicitly do NOT claim

- Novelty of learned world models, reward/value learning, MCTS-with-learned-model, or
  uncertainty penalties per se.
- Real-user improvement from simulator results.
- Causal preference dynamics from latent-state changes.
