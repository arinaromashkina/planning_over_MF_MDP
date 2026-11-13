# H1 design — Separating Belief Updates from Preference Dynamics

Research design for hypothesis line **H1** on branch `h1/belief-vs-preference-dynamics`.
Owner prompt: research proposal "Separating Belief Updates from Preference Dynamics in
Model-Based Planning for Recommendation" (2026-11-13).

> Numbering note: H1/H1.1–H1.4 here are a NEW hypothesis line, unrelated to the
> "H1–H3" stochastic-world-model hypotheses in `docs/PHASE5_HYPOTHESES.md` (those
> concern stochastic response modeling, not belief-vs-preference decomposition).
>
> Naming note: proposal "Environment A/B/C" are renamed here to **Env-L / Env-P /
> Env-J** (Learning-only / Preference-shaping-only / Joint) to avoid collision with
> Phase-5 `Env-S`/`Env-C`, which keep their meaning.

---

## Part 1 — Scientific framing

### 1.1 Refined hypothesis

**H1.** The MF-MDP paper's transition — a fold-in of the recommended item into the
user's MF posterior — is an *epistemic* operation: it is the Bayes update of the
system's belief about a **static** latent user. When true user preferences also
*evolve* under consumption (preference dynamics), the planner that treats the fold-in
as the MDP transition silently assumes `T = identity` on user preferences. This
conflation is harmless in purely learning regimes and arbitrarily harmful in
shaping-dominated regimes; the harmfulness grows with planning horizon, preference-shift
strength, and the gap between myopic-optimal and steering-optimal actions.

Formally (Part 2): MF-MDP planning = POMDP planning under a specific wrong model class
(point-mass observation `y=+`, static-preference kernel). H1 is a claim about *when the
model-class error binds*, not a claim that a POMDP formulation exists (it obviously does).

### 1.2 What is already known (literature map)

| Area | Known result | Relation to H1 |
|---|---|---|
| Belief-state MDPs / POMDP-RS | Belief-state planning for recommendation is classical (Shani-style MDPs, POMDP variants, Bayes-adaptive MDPs) | H1 does NOT propose "a POMDP for recommendation". |
| Posterior-sampling exploration in RS | Thompson-style bandits over MF posteriors; exploration has information value | Env-L makes this the *only* active process; we test whether MF-MDP planning captures it. |
| Preference evolution / shaping | Assimilation, echo chambers, "rabbit hole" steering, dynamic preference MF (Koren temporal dynamics) | Provides the preference-dynamics kernel family; none isolates it against epistemic updates in planning. |
| Causal recommendation | Selection bias / unobserved confounding in logs; randomized exposure (KuaiRand) enables causal estimates | H1-E5: `α` is a causal quantity; identifiable only under randomization or ignorability + overlap. |
| MF-MDP paper (+ our Phase 4/5) | Fold-in transition, optimistic acceptance; Phase 5 added stochastic responses (T1 weighted fold-ins) | The conflated baseline under test. Phase 5's T1 is already "belief-only"; the joint filter is the missing piece. |

**Potentially new, narrow, checkable contribution:** a *diagnostic decomposition*, not a
new architecture: (i) fold-in is shown to be exactly the `T = δ` special case of a
Bayes filter over user latent state; (ii) controlled environments that separate the two
processes characterize the regime boundary (harmless / benign / harmful) via
`α × horizon × initial-uncertainty × response informativeness`; (iii) a closed-form
O(d²) joint belief filter + chance-node planner that matches the history-oracle upper
bound where MF-MDP regret grows with depth; (iv) a worked identifiability statement:
preference dynamics cannot be identified from organic logs without randomization or
assumptions.

### 1.3 Why this is not "just another POMDP"

The proposed object is a *factored diagnostic*: two kernels (epistemic update vs
preference push) composed in one filter. The scientific questions are comparative and
quantitative: when does dropping the push term change decisions and true return; when is
it free; when does deeper search make it worse. A generic POMDP baseline cannot answer
"was the conflation harmful here?" — only our decomposition with oracles can.

### 1.4 What cannot be shown without extra assumptions/data

- That **real** users have `α > 0` (causal claim; needs randomized exposure — KuaiRand
  random file is our one such source — or strong ignorability).
- Identifiability of `T` from **organic** logs (confounded with selection; Phase 5
  Gate E already documented policy-bound drift).
- Transfer of gains to real MF embedding geometry beyond the simulator's
  (mitigation: also run the MF-posterior belief variant, see §5 risks).

---

## Part 2 — Mathematical formulation

### 2.1 True environment (one generator, three regimes)

Latent user state `θ_t ∈ S^{d−1}`; items `v_i ∈ S^{d−1}` with quality `q_i` (reuse
Env-C catalog machinery, `src/mf_mdp/envs/continuous_env.py`).

- Response: `y_t ~ Bernoulli(σ(a·cos(θ_t, v_i) + b·q_i + ε_t))`, `ε ~ N(0, σ_obs²)`.
- Preference dynamics: `θ_{t+1} = normalize((1−λ)·θ_t + α(y_t)·v_i + λ·θ_0)` —
  assimilation family, `α(+) > 0`, `α(−) ≤ 0`. (Robustness variant: satiation /
  drift-to-anchor families, same interface.)
- Reward: `r_t = y_t·q_i·cos(θ_t, v_i)·(1 − κ·f_t)`; termination hazard
  `h = σ(c0 − c1·cos(θ_t, v_i))`; discount `γ`, horizon `H`.

Regimes (controlled by flags of ONE config, not three architectures):

| Regime | Proposal name | `α` | initial belief `b_0` | active process |
|---|---|---|---|---|
| Env-L | A — Learning Only | `≡ 0, λ = 0` | prior over `θ_0` (uncertainty knob `σ_0`) | epistemic only |
| Env-P | B — Preference Shaping Only | `> 0` | point mass at true `θ_0` | preference dynamics only |
| Env-J | C — Learning + Shaping | `> 0` | prior over `θ_0` | both |

### 2.2 Belief state and the correct filter

Belief `b_t ∈ Δ(S^{d−1})`, `b_t = P(θ_t | h_t)`, `h_t` = observed history of
`(recommendation, response)`. The correct Bayes filter factors as

```
b_{t+1} = Push_T( BayesUpdate(b_t, a_t, y_t) )
BayesUpdate:  b^+(θ) ∝ p(y_t | θ, a_t) · b_t(θ)          (epistemic)
Push_T:       b_{t+1}(θ') = ∫ T(θ' | θ, a_t, y_t) b^+(θ) dθ   (preference dynamics)
```

Exact for deterministic `T(·|θ,a,y)`; the factorization is the mathematical statement of
"separating belief updates from preference dynamics".

Belief representations (planner-configurable):

1. **MFPosteriorBelief** — Gaussian `N(u, P)`, `P = A^{−1}`; BayesUpdate = weighted
   fold-in from Phase 5 T1 (`src/mf_mdp/mdp.py:fold_in`); Push_T absent (static-θ
   assumption). O(d²).
2. **JointMFBelief** — same parameterization, WITH the push:
   `μ_d = normalize((1−λ)μ^+ + α(y)·v + λ·θ̄)`, `Σ_d = (1−λ)²Σ^+ + Q`. O(d²). This is
   the proposed closed-form joint filter.
3. **ParticleBelief** — SIR particle filter implementing the exact filter (N≈500–2000);
   used as the history-oracle belief and as a robustness variant for B4.

### 2.3 Where the original MF-MDP lands in this space

| Paper component | In filter terms |
|---|---|
| Fold-in of accepted item (unit confidence) | BayesUpdate under linear-Gaussian likelihood, `y=+` assumed observed |
| Paper's optimistic transition (always accept) | the above + observation model replaced by point mass at `y=1` |
| Using fold-in as the MDP transition | filter with `Push_T = identity`, i.e. kernel `T̂ = δ(θ'−θ)` — **static preferences assumed** |
| Phase-5 T1 (weighted fold-in by response) | BayesUpdate conditioned on realized `y`; still `Push_T = identity` |

So: the paper's MDP is exactly the belief-MDP restricted to `(T̂ = δ, ŷ = +)`; Phase 5
relaxed `ŷ` but kept `T̂ = δ`. **H1 tests what breaks when `T̂ = δ` is wrong.**

### 2.4 Belief-MDP value

`V*(b) = max_a [ r̄(b,a) + γ·E_{y∼p̄(·|b,a)} V*(B(b,a,y)) ]`,
`r̄(b,a) = E_{θ∼b} E_y r`, `p̄` the response predictive. Every baseline (Part 3) is an
approximation of this recursion differing in belief representation, `B̂`, `T̂`,
observation model, and search depth/budget.

### 2.5 Theoretical statements to develop

- **S1 (harmless regime).** If `T = δ` (Env-L), the Bayes filter reduces to the
  epistemic update; B2 with exact response model and exact belief + exact search is
  Bayes-optimal. The paper's optimistic variant overestimates Q on trajectories with
  `p < 1`; the optimism bias is non-negative and (to prove) monotone in depth for
  fixed `p<1`. Prediction for Env-L: B1 regret > 0 mild and growing with `K`;
  B2/B4 regret ≈ 0 — **conflation is free here** (Result B territory).
- **S2 (harmful regime).** In Env-P with exact response model, any planner whose belief
  update contains a nontrivial epistemic term maps observations to information about a
  fixed `θ` when in fact `θ` moves by `T`; its `μ_t` deviates from `θ_t` with
  `E‖μ_t − θ_t‖` growing in `t·|α|`. With a catalog containing an off-mode high-quality
  cluster (steering tension, as in Env-S's handcrafted catalog), the induced first
  action differs from the dynamics-aware optimum and true regret grows with `K`.
- **S3 (observational equivalence / Result E core).** Construct (to prove): two kernels
  `T_1` (assimilation toward consumed genre) and `T_2` (satiation away from it) and a
  logging policy such that the logged-trajectory distributions coincide while the
  optimal first actions are opposite. Conjectured 2-mode × 2-item construction; the
  simulator's Env-P/Env-J with swapped `α(−)` sign is the empirical version.
- **S4 (identifiability sketch).** Under organic logging, the map
  `(b_0, T) → P(logged traj)` is many-to-one (homophily vs assimilation are
  observationally equivalent); restricting to randomized exposure (propensity bounded
  away from 0/1) makes `α` identifiable. Formalize as an injectivity statement; the
  KuaiRand random file is the empirical anchor.

---

## Part 3 — Experimental protocol

Primary metric (frozen, Phase-5 convention): `J(π) = E[Σ γ^t r_t]` under the TRUE
environment; `γ = 0.9`; ≥ 5 seeds, paired bootstrap CIs; matched planning budgets;
HPs on validation only; no tuning on test seeds.

### Oracles (distinct, never mixed)

- **Oracle-state**: planner with direct `θ_t` access. Upper bound overall; its gap vs
  history-informed methods measures the cost of partial observability + representation.
- **Oracle-history**: particle filter with TRUE kernels + true response model, planning
  over beliefs; only history available to the agent. Upper bound for B4-class methods;
  its gap vs B4 measures planner/filter error alone.

### Baselines (all same reward, catalog, horizons, budgets)

| ID | Method | Belief | Kernel model | Observation model |
|---|---|---|---|---|
| B0 | Myopic / Static | MF posterior mean | — | — |
| B1 | Original MF-MDP (paper) | MF posterior | `T̂ = δ`, optimistic `ŷ=+` (T0) | point mass |
| B2 | Belief-only (Phase-5 T1) | MF posterior | `T̂ = δ` | stochastic `p̂(y)` |
| B3 | Preference-dynamics-only | point mass moved by `T̂` | `T̂ = true/estimated` | none (θ known) |
| B4 | **Joint (proposed)** | JointMF / particles | `T̂` | stochastic `p̂(y)` |

Crossed with belief representation where meaningful: B2/B4 × {MF, particles} — this
directly quantifies how much of Phase 5's negative Env-C result was *representation*
("MF planners far below true-θ greedy, representation-bound") vs *conflation*.

### Experiments

**H1-E1 — isolation (H1.1).** Env-L and Env-P side by side; response model exact by
construction (train to convergence; σ_obs small). Question: is B1/B2 regret ≈ 0 in
Env-L and > 0 in Env-P while B4 ≈ 0 in both? Measures: J, regret vs the appropriate
oracle, `‖μ_t − θ_t‖`, action agreement. **This is the minimal experiment that can
confirm or kill H1** — Env-P with exact response model isolates the transition-model
error completely.

**H1-E2 — horizon (H1.2).** Env-P and Env-J; `K = h ∈ {1, 2, 4, 8, 16}` (extend `H`).
Regret(h) curves per method; predicted-return vs actual-return gap per h (overconfidence
growth). Prediction: B1/B2 regret increasing in `h`; B4 flat; no monotonicity assumed a
priori for B1 in Env-L (optimism can be depth-monotone per S1).

**H1-E3 — factor matrix (H1.3).** Grid: `α ∈ {0, .05, …, .5}` × `σ_0 ∈ {point, small,
large}` × `σ_obs ∈ {0, .5}` × `h ∈ {1, 6, 16}` in Env-J. Output: heatmap of
`Δ = regret(B1) − regret(B4)` and of action-agreement(B4, oracle-history). Goal:
characterize the region where joint modeling pays; boundary in `(α·h, σ_0)` space.

**H1-E4 — oracle vs learned dynamics (postulation vs estimation error).** B4 with known
`(α, λ)` vs estimated from logged trajectories under randomized logging (uniform +
ε-greedy, reuse Phase-5 A2 logging); ablate learned vs oracle reward. Separates "wrong
postulation" from "wrong estimation".

**H1-E5 — predictive vs planning accuracy (H1.4) + offline identifiability.** (a) Train
response/transition variants spanning a log-loss range; scatter log-loss/Brier/ECE vs
true regret; exhibit model pairs with Δlog-loss ≈ 0 but Δregret > 0 differing only in
`T̂`. (b) KuaiRand randomized-exposure subset: estimate `α` vs logging-policy coverage;
document the assumptions under which organic logs could ever identify `T`.

### Paper exhibits (planned)

1. Regret-vs-horizon curves, 4 methods × {Env-L, Env-P, Env-J} (money figure, E2).
2. Heatmap `Δregret(Joint, MF-MDP)` over `(α, σ_0)` (E3).
3. Scatter: predictive log-loss vs true regret, labeled by `T̂` variant (E5a).
4. Belief trajectories: `μ_t` vs `θ_t` in Env-P for B2 vs B4 (conflation made visible).
5. Table: methods × envs × {J, regret, action-agreement, predicted-vs-actual gap, sims}.
6. (E5b) Estimated `α` vs coverage / vs random-vs-organic source.

---

## Part 4 — Implementation plan (mapped to repo)

Existing modules are untouched (all additions in new files) → Phase 1/4/5
reproducibility preserved; `pytest tests/` stays green.

| Reuse | From | Role in H1 |
|---|---|---|
| `mf.py:fold_in`, `mdp.py:MFState` | core | BayesUpdate of MFPosterior/JointMF beliefs |
| `envs/continuous_env.py` | Phase 5 | catalog, response, reward, termination patterns for `h1_envs.py` |
| `envs/discrete_env.py:MCTSPolicy` | Phase 5 | chance-node UCT pattern for the belief planner |
| `planning/candidates.py:exact_cosine_topl` | core | expansion pool near belief mean |
| `models/response.py` + metrics | Phase 4/5 | `p̂(y)`, log-loss/Brier/AUC/ECE |
| `experiments/phase5/run_env_a.py` | Phase 5 | rollout + paired-CI harness pattern |
| Phase-5 logging utilities | `envs/learned.py` | behavior policies, trajectory logging (E4/E5) |

New code:

1. **`src/mf_mdp/envs/h1_envs.py`** (Gate B) — one generative model,
   `H1EnvConfig{regime: L|P|J, alpha_pos, alpha_neg, lam, sigma_0, sigma_obs, ...}`;
   three preset configs. Catalog with off-mode high-quality cluster for steering tension.
2. **`src/mf_mdp/belief/`** (Gate B) — `Belief` ABC (`mean, cov, sample(rng),
   update(a, y)`); `MFPosteriorBelief`; `JointMFBelief` (§2.2); `ParticleBelief` (SIR).
3. **Oracles** (Gate B) — `OracleStatePlanner` (true-θ MCTS, Env-C style);
   `OracleHistoryPlanner` (particles + true kernels + chance-node search).
4. **`src/mf_mdp/planning/stochastic_belief.py`** (Gate C) — chance-node UCT over belief
   states; B0–B4 as `PlannerConfig`-style variants; matched-budget accounting.
5. **`experiments/h1/`** (Gate D/E) — `run_e1_isolation.py`, `run_e2_horizon.py`,
   `run_e3_grid.py`, `run_e4_learned_dynamics.py`, `run_e5_predictive_vs_planning.py`;
   configs in `configs/h1/*.yaml`.
6. **Tests** — filter correctness (JointMF vs ParticleBelief MC; fold-in vs exact
   Gaussian update in the linear-Gaussian special case); regime regression (Env-L with
   `α=0` ≡ Env-C fixed-θ semantics); **no-leakage test** (planner interface receives
   only `Belief`; no `θ` anywhere in the policy path — enforced by API + a test that
   passes a wrapper hiding the env); determinism/seed tests; budget-matching test.

**MVP = Gate B + Gate C + H1-E1** (isolation experiment in Env-P with exact response
model). Everything is CPU-scale per Phase-5 experience (no server needed unless
particle counts or grid sizes grow substantially).

**Sequence (cheapest decisive first):** Gate B (env+filters+oracles) → Gate C (planner)
→ E1 (kill/confirm) → E2 (horizon) → E3 (grid) → E4 (learned dynamics) → E5 (offline +
KuaiRand). E1 in Env-P is deliberately first: it needs no learned components at all.

---

## Part 5 — Critical evaluation

### 5.1 Real novelty assessment

Belief-MDP recommendation: not new. Novel and checkable here: (a) the *exact
characterization* of a published planner's transition as the `T̂=δ, ŷ=+` corner of the
Bayes filter; (b) the *regime boundary* — controlled evidence of when the conflation is
free vs harmful (the literature asserts both "explore, posterior = state" and
"preferences are shaped", rarely in one framework with shared oracles); (c) the O(d²)
joint filter that upgrades the existing Phase-5 T1 implementation rather than replacing
the stack. Target venue: RecSys/KDD-workshop-scale if Result A or C holds; strong
negative Result B (boundary characterization) is also publishable.

### 5.2 Weak points and alternative explanations (each with mitigation)

1. **Representation confound.** Phase 5 already showed MF posterior states can be
   binding (Env-C: MF planners ≪ true-θ greedy). If unread, H1's "conflation error" may
   be pure representation error. Mitigation: particle-belief variants crossed with MF
   beliefs; report oracle-state vs oracle-history gaps; make the crossing a headline
   exhibit, not a footnote.
2. **Optimism ≠ conflation.** The paper's planner is both optimistic about `y` AND
   conflated. Mitigation: B2 (stochastic response, static kernel) separates the two;
   all claims about conflation must hold with B2 as the reference, not B1.
3. **Simulator triviality.** Assimilation direction/form is our choice. Mitigation: ≥2
   dynamics families (assimilation, satiation, drift-to-anchor); E3 sweeps `α` broadly;
   S3 construction shows observational equivalence is not an artifact of one family.
4. **Compute unfairness.** Belief planners cost more per step. Mitigation: Phase-5
   matched-budget convention (sims × node cost reported; latency tables).
5. **Horizon semantics.** Planning depth vs real session length (already documented for
   Phase-4 value models) — keep readout at `k=K` and report both.

### 5.3 Continue / pivot criteria

- **E1 Env-P**: if B2 ≈ B4 ≈ oracle across horizons with exact response model →
  conflation harmless in the cleanest possible setting → pivot (document as negative
  Result B only if the boundary is otherwise informative).
- **E3**: if `Δregret(B1, B4) ≈ 0` across the whole grid → preference dynamics don't
  matter for planning here → pivot to E5-only (identifiability paper).
- **Proceed** if E1 shows B2 regret > 0 while B4 ≈ 0 in Env-P, and E2 shows the gap
  growing in `h` — that pair alone justifies the full program.

### 5.4 Strongest contribution statement (draft)

> Posterior-update transitions in MF-based recommendation planning are exact epistemic
> filters operating under a silently assumed static-preference kernel. We prove they are
> the `T = δ` corner of the Bayes filter over the user latent state, and we characterize —
> in controlled environments with history- and state-oracle upper bounds — the regimes
> where this conflation is harmless (learning-only), benign, and arbitrarily harmful
> (shaping-dominated), as a function of shift strength, horizon, and initial
> uncertainty. A closed-form O(d²) joint belief filter with a chance-node planner
> matches the history-oracle bound where the original MF-MDP's regret grows with depth,
> and we show preference dynamics are not identifiable from organic logs without
> randomized exposure — with a validating estimate on KuaiRand's randomized subset.

---

## Open items before Gate B sign-off

- [ ] Freeze `H1EnvConfig` numeric defaults by a small pilot (steering tension must make
      greedy ≠ oracle-history in Env-P, mirroring how Env-S was validated in Phase 5 A1).
- [ ] Decide particle counts / expansion widths so matched budgets are fair.
- [ ] Choose validation split convention (separate val seeds; no test-seed tuning).
