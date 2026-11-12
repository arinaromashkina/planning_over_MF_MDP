# Phase 5 formal problem statement (Gate A)

## Plain language

A recommender chooses items sequentially. The user responds stochastically (engage /
skip / terminate), and responses change what the user will enjoy next. We ask: when is it
worth planning over these stochastic, feedback-dependent dynamics — compared to the
optimistic "every recommendation is accepted" model — and when are learned reward/value
components worth their cost?

## Data-generating assumptions (controlled environment, Experiment A)

Two regimes with explicit generative equations (full spec in PHASE5_EXPERIMENT_PLAN.md):

- **Env-S (small, discrete, exact-DP oracle).** User state s ∈ {1..K} (interest mode) plus
  fatigue f ∈ {0,1,2}; items with mode-affinity and quality. Engagement probability
  p(engage | s, i) = σ(a(m_i[s] − θ0) + b q_i). Reward on engagement:
  r = q_i·m_i[s]·(1 − κ_sat f). Assimilation: engagement with item i moves the user's
  mode toward argmax m_i w.p. ρ. Termination hazard h = σ(c0 − c1·E[r | s, i]).
  Stochasticity: Bernoulli engagement + termination. Delayed structure: steering the
  mode toward high-quality clusters pays off over the horizon; fatigue punishes
  same-mode repetition; termination couples satisfaction to session length.
- **Env-C (continuous, MF-flavored, approximate oracle).** True latent θ_t ∈ R^d evolves
  as θ_{t+1} = normalize((1−λ)θ_t + α(y)·v_i + λ·θ_0), α(+) > 0, α(−) < 0; engagement
  p = σ(a·cos(θ_t, v_i) + b·q_i + ε); reward as in Env-S; termination hazard on mismatch.
  iALS factors are trained on logged trajectories from behavior policies; the learned
  model uses MF states (P, u) and exact weighted fold-ins.

## Notation

State s_t (MF posterior (P_t, u_t) or discrete (s, f)); action a_t (item);
response y_t ∈ {engage, skip}, termination d_t ∈ {0,1}; utility U(y_t, d_t);
policy π(a|s); planner π_plan; learned response p_φ(y, d | s, a);
transitions T0 (optimistic fold-in) / T1 (feedback-conditioned analytic) /
T2 (learned); reward r̂_ψ; value V_ω with the evaluated policy ALWAYS named
(V_logging vs V_π).

## Model class (this phase, small on purpose)

- Response: logistic / small MLP heads per outcome; termination as hazard.
- Transition: T1 — sample y ~ p_φ, apply exact weighted fold-in
  (alpha_w, beta_w from the response type: engage → (α, 1+α); skip → baseline confidence
  change with β=0; documented per response). Optional T2 GRU state dynamics (S1 variant).
- Reward: fixed transparent utility mapping U; optional learned expected utility.
- Value: ridge/MLP on state features; policy named.
- Planning: stochastic lookahead (chance nodes) over a bounded ANN candidate pool;
  budgets matched across methods.

## Internal objectives (training)

Response: negative log-likelihood of observed (y, d). Transition (T1): analytic, no
training. Reward: MSE/log-loss against U(y). Value: MC return regression under the NAMED
policy (logging on real data; target-policy only in the controlled simulator).

## External evaluation criteria

Primary: J(π) under the TRUE environment; regret vs oracle; error-vs-depth curves;
compute. Real data: predictive validity + (if propensities verify) OPE.

## Optimization problem

max_π E_π,T*[Σ γ^t r_t] subject to offline data support, candidate-pool restriction,
and compute budgets (matched across compared planners).

## Constraints

Mac M4 dev; L4 server for GRU/larger runs; KuaiRand-1K access (public download).
No fabricated labels; no future leakage; negative results reported.
