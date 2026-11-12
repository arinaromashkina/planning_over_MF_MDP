# Phase 5 hypotheses (Gate A)

Working title: **Beyond Optimistic Planning: Learning Stochastic User Dynamics for
Model-Based Sequential Recommendation**.

## Research questions

- RQ-H1: does feedback-conditioned stochastic dynamics beat optimistic deterministic
  fold-in for long-horizon decisions, evaluated in a TRUE controlled environment?
- RQ-H2: do learned reward and/or value components add anything beyond stochastic
  dynamics (factorial)?
- RQ-H3: how do model error and feedback stochasticity interact with planning depth;
  is there a depth where planning stops paying (or starts hurting)?

## H1 (primary)

In environments with meaningful action-dependent user dynamics,
J(π_stochastic) > J(π_optimistic), where J is evaluated under the TRUE environment
transition T* (controlled simulator with known generative equations), not inside the
learned model. π_stochastic uses a learned response model + feedback-conditioned
transitions; π_optimistic uses the paper's deterministic positive fold-in.

Falsification: J(π_stochastic) ≤ J(π_optimistic) − margin at matched planning budgets
across both environment regimes, with CIs. A negative result is reported as-is.

## H2 (components)

Factorial over {response/transition model} x {reward} x {value}:
- stochastic dynamics alone may suffice;
- learned reward may help only when its labels carry real signal (KuaiRand watch ratio;
  in the controlled env, the utility is known by construction);
- learned value (as MCTS leaf) may reduce depth/sims at equal quality.
Falsification: no component combination beats A1 (stochastic dynamics + fixed utility)
beyond CIs.

## H3 (depth vs error)

There exists a regime (model error ε, stochasticity σ, horizon h) where deeper planning
stops improving or degrades true-environment return. We measure regret vs (h, ε, σ)
surfaces. Falsification: monotone improvement in h for all tested (ε, σ) — then adaptive
depth is unnecessary in this setting.

## Primary metrics (pre-registered)

- Controlled env: J(π) = E[Σ_t γ^t r_t] under the TRUE environment; regret vs exact DP
  oracle (Env-S) or vs high-budget true-env MCTS oracle labeled approximate (Env-C).
- Real data (KuaiRand): predictive metrics only (log-loss, Brier, AUC, ECE for response;
  RMSE/MAE for reward/value); OPE only where propensities verified — otherwise labeled
  simulation evidence.
- Compute: latency mean/p95, sims, depth, leaf evals, peak memory.
- Recall@10/nDCG@10 kept for comparability with the MF-MDP paper, never as sole metric.

## Expected limitations

- Controlled env results do not transfer automatically to real users (simulator is a
  diagnostic, not proof).
- ML-1M lacks feedback labels; KuaiRand-1K has exposure randomization only in a subset.
- Learned value on logged data is logging-policy value (V_logging), never V_π.
- Latent-state changes are model representations, not causal preference claims.

## Statistical protocol

≥ 5 seeds for controlled-env comparisons; paired user-level bootstrap CIs; two-sided
paired tests; subset/one-seed runs explicitly labeled; validation-only HP selection.
