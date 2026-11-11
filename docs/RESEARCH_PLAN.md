# Research plan

Working title: **Beyond Optimistic Planning: Adaptive, Uncertainty-Aware Model-Based Planning
for Sequential Recommendation** (full brief: `../Beyond_Optimistic_Planning_Full_Research_Prompt.md`).

## Current focus (owner decision, 2026-11-11)

Priority extension: **learned reward model (§4.4) and learned value model (§4.5)** of the brief,
aligned with the paper's stated future work ("learn stochastic dynamics and values from logs ...
use a sequence model as a value function inside the MF-MDP"). This sits on top of a faithful
Phase-1 reproduction; the reproduction is the prerequisite.

## Research questions (from the brief)

- RQ1: Which user/catalog/sequence/environment characteristics predict benefit of Plan-K over Plan-1?
- RQ2: Does stochastic response modeling improve planning when response outcomes are observed?
- RQ3: How do model error and epistemic uncertainty compound with rollout depth?
- RQ4: Can adaptive compute allocation approach deep-MCTS quality with fewer simulations?
- RQ5: Does uncertainty-aware planning improve robustness under temporal shift?
- RQ6: What is the contribution of learned reward and learned value?

Hypotheses H1–H6 as in the brief; all treated as falsifiable, not expected outcomes.

## Phase status

| Phase | Content | Status |
|---|---|---|
| 0 | Bootstrap repo, infra, docs | in progress |
| 1 | Reproduce Static / Plan-1 / Plan-K, ML-1M LLN | in progress |
| 2 | Fair baselines (dot/cos, alignment ablation), planning analysis | pending |
| 3 | Stochastic response model, feedback-conditioned transitions | pending |
| 4 | Learned reward + value models, leaf evaluation, off-policy caveats | pending (design) |
| 5 | Uncertainty-aware adaptive planning, Pareto quality/latency | pending |
