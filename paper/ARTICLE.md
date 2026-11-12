# Beyond Optimistic Planning: When Does Learning Stochastic User Dynamics Make Recommendation Planning Better?

**Arina Romashkina** — AI VK / Scientific Studio research project
Repository: `github.com/arinaromashkina/planning_over_MF_MDP`

## Abstract

The MF-MDP formulation casts top-K retrieval as planning over the implicit-ALS
posterior: the state is the fold-in posterior `(P, u)`, the transition is a closed-form
rank-one update that *optimistically* assumes every recommendation is accepted, and the
reward mixes cosine relevance with a posterior-alignment term. We reproduce this method
and show that, on MovieLens-1M under the leave-last-n protocol, the static baseline is
reproduced (Recall@10 0.0704 vs. 0.0694 reported) but the reported planning gains are
not; a leverage analysis explains why: the quantity driving the reward's alignment term
is bounded by ~d/n_items for spread embedding geometries and correlates with item
popularity rather than user relevance, leaving the planning layer quality-inert. We then
ask whether *learning stochastic user responses* fixes this. In two controlled
environments with known generative dynamics and exact oracles we show that
feedback-conditioned stochastic planning reaches the oracle and beats the optimistic
planner by +0.122 (paired bootstrap CI95 [0.115, 0.129]) when the state representation
is faithful, and does not differ from it (−0.004, CI95 [−0.010, 0.002]) when the state is
a compressed MF posterior. The benefit of deeper planning is governed by the same
bottleneck: regret falls 5–15× from depth-2 to full-horizon planning once the model is
reasonable, and response-model estimation error stops being binding once the model
saturates. On KuaiRand-1K, graded engagement (watch-ratio) is predictable and transfers
to randomized exposure (AUC 0.839 → 0.826), while binary click prediction is policy-bound
and temporally miscalibrated (ECE 0.234). Our answer to "when does learning a stochastic
recommendation environment make planning better?" is: only when the state representation
carries the dynamics — otherwise the transition model's sophistication is irrelevant.

**Keywords:** recommender systems; model-based reinforcement learning; Monte-Carlo tree
search; implicit ALS; world models; offline evaluation.

---

## 1. Introduction

**Motivation.** A recommendation is never an isolated event: a consumed item changes the
user and therefore what should be retrieved next. Model-based planning promises to
exploit this, but requires a dynamics model. The starting paper (Trapeznikov & Utushkin,
KDD 2026 Workshop) builds such planning directly on the implicit-ALS posterior: the user
state is the pair `(P, u)` where `P = A⁻¹` is the inverse precision of the fold-in
least-squares problem, actions are catalog items, and the transition is a closed-form
Sherman–Morrison rank-one update that treats the recommended item as *accepted* positive
feedback — an **optimistic** model. Its reward combines cosine relevance with a
posterior-alignment term driven by the leverage `ℓ = vᵀPv`, and its planners span
Static, one-step Plan-1, and MCTS Plan-K.

**The problem.** The optimistic transition is a misspecified dynamics model: real users
skip, partially consume, or leave. Whether planning over such a posterior helps at all —
and whether a learned *stochastic* response model changes the answer — is an open,
falsifiable question that logged-data metrics alone cannot answer, because the true
counterfactual dynamics are unobservable.

**What exists.** Learned world models for recommendation are an active area: MCRL learns
reward/transition structure with contrastive objectives; ROLeR and R3S refine reward
models and uncertainty penalties; KuaiSim, RecSim NG, and the KuaiEnv of CIRS provide
hand-built simulators; a recent session-level REINFORCE method simulates user feedback
at test time. What is missing is a controlled measurement — against a *known* true
environment — of when stochastic feedback modeling beats optimistic planning
*specifically over the analytic MF posterior*.

**Our proposal.** We keep the analytic posterior and replace only the optimistic
assumption: a learned stochastic response model `p(y | s, a)` feeds exact
feedback-conditioned weighted fold-in transitions `(α, β)`. All claims are evaluated in
controlled environments with known dynamics and exact oracles, at matched budgets, with
paired bootstrap CIs.

**Contributions.**

1. A test-verified reproduction of the MF-MDP pipeline and an explanation of its
   inertness on MovieLens-1M (leverage as a popularity proxy bounded by ~d/n_items).
2. A controlled-environment answer to when stochastic planning helps: faithful state →
   near-oracle gains; compressed MF posterior → no effect.
3. An error-vs-depth analysis: deeper planning cuts regret sharply once the model is
   reasonable; model error stops binding once the response model saturates.
4. A KuaiRand-1K validation: graded engagement transfers to randomized exposure; binary
   click does not.

**Limitations of prior work (relevant here).** The starting paper's evaluation is
entirely on logged data with unknown counterfactuals; learned world-model approaches
evaluate inside their own models or on offline proxies. Neither isolates the transition
model's contribution against a known truth.

## 2. Problem statement

**Plain form.** At each step a recommender shows one item `a`; the user responds
stochastically with `y ∈ {engage, skip}` and may terminate; the response changes the
user's future preferences. We ask which planning model — optimistic or stochastic —
produces better *sequences* of recommendations.

**Formal.** State `s_t` (a representation of the user's history), action `a_t` from a
catalog, response `y_t`, termination `d_t`, utility `r_t`. A planner maps states to
actions. The objective is the expected discounted return under the TRUE transition `T*`:

```
J(π) = E_{π, T*} [ Σ_{t=0}^{H-1} γ^t r_t ]  →  maximize over planners.
```

Internal criterion (training): negative log-likelihood of observed responses; for value,
MC return-to-go regression under a *named* policy. External criteria: `J(π)` in the true
environment, regret vs an exact oracle, Recall@10/nDCG@10 on logged data (secondary),
latency and simulation counts. Constraints: offline data support, matched compute
budgets, documented assumptions. The model class: logistic/MLP response heads, analytic
weighted fold-in transitions, ridge/MLP value heads, chance-node lookahead search.

## 3. Related work

See `docs/LITERATURE_REVIEW_PHASE5.md` for the full comparison table. Closest lines:
(i) model-based offline RL for recommendation (MCRL, ROLeR, R3S); (ii) simulators
(KuaiSim, RecSim NG, KuaiEnv/CIRS); (iii) stochastic planning (Stochastic MuZero's
chance nodes); (iv) learned feedback models at inference time (session-level REINFORCE,
WMG-RL). Our gap: controlled-environment measurement of optimistic-vs-stochastic planning
over the analytic MF posterior, with error-vs-depth accounting. We do not claim novelty
for learned world models, reward/value learning, or MCTS per se.

## 4. Method

### 4.1 The MF-MDP and the optimistic baseline

State `s = (P, u)`, `P = A⁻¹`, `A = λI + Σ_i c_i v_i v_iᵀ` over the catalog; optimistic
transition `A⁺ = A + vvᵀ`, `b⁺ = b + v` with closed forms `P⁺ = P − zzᵀ/(1+ℓ)`,
`u⁺ = u + z(1−uᵀv)/(1+ℓ)`, `z = Pv`, `ℓ = vᵀPv`; reward
`r(s,a) = sim(u,v_a) + η/(1+|Δs(a)|)` with `Δs(a) = ℓ(1−uᵀv)/(1+ℓ)`.

### 4.2 Reproduction findings that shape the method

- Static(dot) reproduces (0.0704 vs 0.0694); Plan-1's reported gain does not (0.0694 vs
  0.0762). Root cause: `ℓ ≤ ~d/n_items` for spread embeddings; the alignment term is
  then inert (or, where active, sorts by popularity). Full analysis with the trace
  identity `λ·tr(P) + Σℓ + α·Σ_H ℓ = d` verified in `docs/PLANNING_DIAGNOSTICS.md`.
- Learned rewards (R1/R2) lose to the paper heuristic on ML-1M (AUC 0.62 response
  signal, popularity-flavored); `V_logging` on deployable features ≈ a mean predictor
  (R² 0.019).

### 4.3 The stochastic model

Response model `p_φ(y | s, a)` (logistic/MLP on decision-time features); termination
hazard head; transitions by exact weighted fold-in `(α_w, β_w)`: engage →
`(α, 1+α)` (the exact re-fold), skip → negative-α update; planning by chance-node
lookahead `Q(s,a) = E_y[r + γ(1−h)V(s')]`.

### 4.4 Controlled environments (Experiment A)

- **Env-S** (exact oracle): 12 discrete states (mode × fatigue), assimilation,
  termination hazard, handcrafted catalog creating greedy-vs-optimal tension.
- **Env-C** (approximate oracle): continuous θ-dynamics with assimilation/drift,
  iALS posterior states, response + termination heads, 2-step lookahead planners.

## 5. Experimental setup

MovieLens-1M LLN (n_test=1) for the reproduction; HPs selected on validation
(documented gaps in `docs/REPRODUCTION.md`). Env-S: 5 seeds, 4000 rollouts, exact DP
oracle; stochastic planners trained on logged trajectories from ε-greedy/uniform
behavior policies. Env-C: 5 seeds, 200 eval users per seed, true-env J. KuaiRand-1K:
chronological split (standard_1 → standard_2), decision-time-only features, the
randomized-exposure file as an unbiased probe. Metrics: J, regret, Recall@10/nDCG@10,
latency; AUC/log-loss/Brier/ECE for response models.

## 6. Results

### 6.1 Reproduction and diagnostics (MovieLens-1M)

Static reproduced; planning gains not reproduced; mechanism identified (leverage as a
popularity proxy; alignment inert/harmful). A candidate-generation bug found during the
reward-semantics audit (consumed items leaking into ANN pools) was fixed and
regression-tested; the "popularity amplification" interpretation was retracted in a
dated addendum.

### 6.2 Controlled environments

- **A1 (planning correctness):** greedy ≠ DP-optimal (J gap 0.084, CI95 [0.069, 0.099]);
  chance-node MCTS converges to the DP choice as budget grows (residual = near-ties).
  A DP-oracle bug (ignored assimilation failure) was caught by this cross-check.
- **A3 (Env-S ablation, matched depth-H budgets):** optimistic 0.523±0.012, stochastic
  0.645±0.007, oracle 0.651±0.007; stochastic − optimistic = **+0.122, CI95
  [0.115, 0.129]**; oracle regret 0.006. Response model: AUC 0.75, ECE 0.012.
- **A4 (Env-C, MF posterior):** greedy_mf 0.171, optimistic 0.217, stochastic 0.213;
  stochastic − optimistic = **−0.004, CI95 [−0.010, 0.002]**; all MF planners sit far
  below the true-state greedy (0.685) — the bottleneck is representation, not the
  transition model.
- **H3 (depth × noise):** regret falls 5–15× from depth-2 to depth-6 for all noise
  levels; estimation error stops binding once the response model saturates; no
  "depth-hurts" regime appears while the transition structure is exactly known.

### 6.3 KuaiRand-1K

wr80-on-clicks AUC 0.839 (test) / 0.826 (randomized subset); click AUC 0.607 with ECE
0.234 on the chronological test — temporal calibration degrades; popularity-only
baselines AUC 0.51–0.54.

## 7. Limitations

- Env-S/Env-C are handcrafted; the contrast indicates the role of the state
  representation but is not a strict single-variable isolation.
- Simulator results are diagnostics, not evidence of real-user improvement.
- ML-1M lacks exposure labels; KuaiRand propensities are not verified, so no OPE claims.
- `V_logging` (logging-policy value) is never `V_target`; target-policy value remains
  unimplemented.
- The response model on KuaiRand uses basic features; the 3.4GB statistic feature file is
  unused.

## 8. Conclusion

The answer to our research question is conditional: **learning stochastic user dynamics
makes model-based planning better than optimistic or greedy recommendation if and only if
the state representation carries the dynamics.** When the state is faithful, a learned
response model plus exact feedback-conditioned transitions recovers the oracle; when the
state is a compressed MF posterior, the transition model's sophistication is irrelevant.
This shifts the design question from "how to plan deeper" to "what state to build".

## References

See `paper/references.bib`. Key: Trapeznikov & Utushkin 2026 (MF-MDP); Hu et al. 2008
(iALS); Rendle et al. 2022 (iALS practice); Steck 2011 (popularity); MuZero / Sampled /
Stochastic MuZero; MOReL; Value Equivalence (Grimm et al.); MCRL; ROLeR; R3S; KuaiSim;
RecSim NG; CIRS; Session-Level REINFORCE 2026; WMG-RL 2026.
