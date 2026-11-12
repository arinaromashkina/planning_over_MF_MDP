# Phase 5 experiment plan (Gate A; frozen protocol)

Prompt §7–§10. Primary metric: J(π) = E[Σ γ^t r_t] under the TRUE environment.
γ=0.9. All HPs on validation only; ≥5 seeds for controlled-env claims; paired bootstrap CIs.

## Environments

### Env-S (small discrete; exact DP oracle)

- K=4 modes; N=10 items; state (s ∈ {0..3}, f ∈ {0,1,2}) → 12 states.
- Item params: affinity m_i ∈ [0,1]^K (row-normalized), quality q_i ∈ [0,1].
- Engagement: p_i(s) = σ(4(m_i[s] − 0.5) + 1.0·q_i).
- Reward on engage: r = q_i·m_i[s]·(1 − 0.35·f); skip → 0.
- Termination after the step: h(r) = σ(−1.5 − 4r); terminal → no future reward.
- Mode transition on engage: s' = argmax_k m_i[k] w.p. ρ=0.6 (assimilation), else s.
  On skip: s'=s. Fatigue: f' = min(2, f+1) if argmax m_i == s, else 0.
- Horizon H=6. Exact backward induction over 12 states (+terminal) — the ORACLE.

Designed tension: high-q items off the current mode can pay off later via assimilation
(steering), while repetition builds fatigue and mismatch raises churn. Greedy myopic
choice is suboptimal by construction (verified in A1).

### Env-C (continuous MF-flavored; approximate oracle — Gate C/D)

- θ_t ∈ R^32; θ_{t+1} = normalize((1−λ)θ_t + α(y)v_i + λθ_0), α(engage)=+0.3,
  α(skip)=−0.15, λ=0.1; engagement p = σ(6 cos(θ,v) + 2 q_i); reward on engage
  q_i·cos(θ,v)(1−0.3 f); termination hazard σ(−2 − 3·cos(θ,v)) on mismatch steps.
- iALS factors trained on logged trajectories from behavior policies; learned model uses
  MF states (P,u) + exact weighted fold-ins; oracle = high-budget MCTS in the TRUE env
  (labeled approximate).

## Experiment A1 — planning correctness (Gate B)

- Greedy (1-step expected reward) vs DP-optimal: verify decision disagreement at ≥1 state
  and J gap with CIs over ≥5 seeds.
- Chance-node stochastic MCTS (true env as simulator) with growing budget: verify
  convergence to the optimal action at test states.
- Verify the stochastic backup against exact 2-step enumeration (hand-checked).

## Experiment A2 — learn the environment (Gate C)

Behavior policies: ε-greedy-wrt-quality (ε=0.3) and uniform-random; log (s, i, y, d).
Train: logistic response model p̂(y|s,f,i) (+ termination hazard head); transition
parameters estimated where T2. Evaluate: log-loss/Brier/AUC/ECE on held-out trajectories.

## Experiment A3 — component ablation (Gate D, matched budgets)

Variants (planning budget matched): A0 optimistic (paper T0 + paper-style reward),
A1 stochastic response + analytic transition (T1) + fixed utility,
A2 + learned transition (T2), A3 + learned reward, A4 + learned value leaf.
Metric: J under TRUE env; regret vs DP oracle; latency/sims/depth.

## Experiment A4 — error vs depth (H3, Gate D)

Vary response-model training-set size (accuracy), injected response noise, and horizon
h ∈ {1,2,3,6}: measure regret(h) curves per noise level. No tuning on test.

## Experiment B — KuaiRand-1K (Gate E)

Inspect fields; separate randomized-exposure subset; chronological split; response model
p(y|watch ratio etc.) with calibration by time; transition/reward/value predictive
metrics; OPE only if propensities verified; else predictive validity + documented limits.

## Baselines (prompt §9)

Static MF; Plan-1; Plan-K (original); greedy expected utility; stochastic-response +
analytic transition; + learned reward; + learned value; BC policy (no planning);
proposed stochastic planner.

## Leakage checks

All env-model training on train trajectories only; HPs on validation trajectories;
features from past only; oracle computed only in the true env; no tuning on test seeds.

## Execution commands (single entry points)

```bash
.venv/bin/python experiments/phase5/run_env_a.py --checks      # A1
.venv/bin/python experiments/phase5/run_env_a.py --learn       # A2
.venv/bin/python experiments/phase5/run_env_a.py --ablation    # A3
.venv/bin/python experiments/phase5/run_env_a.py --depth       # A4
.venv/bin/python experiments/phase5/inspect_kuairand.py        # Gate E inspection
```
