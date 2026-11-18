# H2 design — Hybrid Sequential Belief State for Model-Based Recommendation Planning

Research design for hypothesis line **H2** on branch `h2/hybrid-sequential-state`.
Owner hypothesis (2026-11-17, paraphrased): a hybrid belief state that combines the
MF posterior with a sequential (SASRec-style) encoder state should reduce state
representation error and make stochastic planning more effective than planning over a
single MF posterior.

> Numbering note: "H2" here is a NEW hypothesis line, distinct from the Phase-5 "H2"
> (learned reward/value, `docs/PHASE5_HYPOTHESES.md`) and from the `h1/…` line
> (belief updates vs preference dynamics, `docs/H1_BELIEF_VS_PREFERENCE_DESIGN.md`).
> The three lines are complementary: H1 asks *what the transition misses about the
> user*; H2 asks *whether the state itself is the bottleneck* and replaces/augments
> the MF posterior with a learned sequential state.

---

## Part 1 — Hypothesis and framing

### 1.1 Hypothesis

**H2.** The MF-MDP state `s = (P, u)` is an order-blind, response-blind sufficient
statistic: the fold-in posterior depends on the *set* of consumed items (and,
in Phase-5 T1, on their responses only through scalar confidence weights), not on
*which item was shown before which response*. Where true user dynamics depend on
order and on the last feedback (preference assimilation/satiation, session context,
short-term interest), a sequential encoder state `h_t` carries strictly more
decision-relevant information, and a hybrid state `b_t = (P_t, u_t, h_t)` dominates
either alone.

Formally: the MF posterior is a statistic of the history `H_t`; `h_t = f_eta(H_t)`
is a learned statistic; H2 claims that planning value `J(pi)` under the true
environment is ordered `J_hybrid >= J_seq >= J_mf` at matched reward, candidates and
search budget, in environments with feedback-conditioned dynamics.

**Falsification criteria** (pre-registered):
- If `J_mf` ≈ `J_seq` ≈ `J_hybrid` across regimes including strong preference
  dynamics → sequential state adds nothing for planning → H2 rejected for planning
  (it may still add predictive value — reported separately).
- If `J_seq` > `J_hybrid` → the MF component actively hurts the hybrid → the
  "hybrid" claim is rejected in favor of a pure sequential state.
- If the advantage survives the negative controls (§4) → the mechanism is NOT
  order/feedback/action conditioning (popularity or side features) → H2's mechanism
  claim is rejected even if the J-gap persists.

### 1.2 Why this is grounded in our own results

Phase 5 (Env-C, `docs/PHASE5_RESULTS.md`) found MF-based planners *representation-
bound*: all MF planners sat far below the true-`theta` greedy policy, and
stochastic-vs-optimistic transition modeling changed `J` by ≈ 0. That is exactly the
signature of a state that cannot express what the transition needs: better response
modeling over a poor statistic does not help. H2 tests the statistic directly, with
the response/transition learning problem held as comparable as possible across arms.

### 1.3 Scope of "SASRec"

The owner names SASRec. We operationalize it minimally and honestly: a *causal
sequential encoder* — GRU over events `(a_{t-1}, y_{t-1}, dt)` with a response head
`p(y|h,a)` and a next-item readout — trained by teacher forcing. Rationale: the
NumPy stack must stay CPU-reproducible (server SSH is currently unreachable; Phase 5
was entirely CPU-scale), and the scientific claim concerns *what the state can
represent* (order, last feedback, action conditioning), not the attention mechanism
per se. A transformer (SASRec-exact) variant is future work under the same
interface; the negative controls would be unchanged.

---

## Part 2 — Model

### 2.1 Events and state

History event at step `t`: `(a_t, y_t, dt_t)` — shown item, binary response
(engage/skip; labels the dataset actually supports), time gap. In Env-C, `dt ≡ 1`
(discrete steps; the feature exists for real-data transfer, e.g. KuaiRand).

States under test (factorial factor **S**):

| Arm | State | Maintained by |
|---|---|---|
| `mf` | `(P_t, u_t)` | iALS fold-in (`mf.py:user_state`), Phase-5 T1 weighted fold-in for stochastic transitions |
| `seq` | `h_t ∈ R^{dz}` | GRU over events |
| `hybrid` | `(P_t, u_t, h_t)` | both |

### 2.2 Feedback-conditioned dynamics (factorial factor **T**)

Per arm, two transition modes:

- **optimistic** (paper analog): assume engagement; MF arm uses unit fold-in
  (`mdp.py:fold_in`); seq/hybrid arms use `h' = GRU(h, [E_a, y=1, dt])`.
- **stochastic**: branch over `y ~ p_hat(y | state, a)`; MF arm uses Phase-5 T1
  weighted fold-in `(alpha_w, beta_w)` per branch; seq/hybrid arms step the GRU
  with the branch's `y`. Termination hazard folded in as `(1 - h_hat)` on the
  continuation, exactly the Phase-5 Bellman form.

`p_hat` is part of the state arm: MF arm — logistic on MF decision features
(Phase 4/5); seq/hybrid arms — logistic head on `[h, E_a, h*E_a]`. No popularity
term anywhere in the sequential model (popularity enters only through the shared
reward `q_hat`, §2.3) — this keeps the negative controls interpretable.

Factorization note: the world model is
`p_theta(h_{t+1}, y_t | h_t, a_t) = p_phi(y_t | h_t, a_t) * delta(h_{t+1} - g(h_t, a_t, y_t))`
with deterministic GRU `g` — the minimal faithful feedback-conditioned model; a
noisy `g` (latent stochasticity) is future work.

### 2.3 Shared reward and planning (held fixed across arms)

- Utility: `r_hat = p_hat(y=1|state,a) * q_hat_a * max(cos(u_state, v_a), 0) * (1 - kappa * f)`,
  where `u_state` is the arm's user-vector readout: `u` (mf), `W_r h` (seq),
  normalized sum `u/||u|| + (W_r h)/||W_r h||` (hybrid). `E`/`V` geometry: the seq
  readout uses the learned embedding table `E`; MF arms use iALS `V`.
- `q_hat`: engagement-shrunk empirical mean per item from training logs
  (`docs/DECISIONS.md`-style assumption; *not* the env's true `q` — less leaky than
  the Phase-5 Env-C convention, identical across arms). `kappa` from env config
  (shared, documented).
- Planner: depth-2 stochastic lookahead, candidate pool `L=8` by state relevance,
  both outcome branches, greedy leaf = max next expected utility; identical code
  path and budget for every arm (matched-budget convention, Phase 5).

### 2.4 Training objective (seq/hybrid world model)

Teacher forcing on logged trajectories:
`L = NLL(y_t | h_t, a_t) + w_next * ||W_r h_t - E[a_{t+1}]||^2 + l2`.
The next-item term both regularizes `h` and yields the next-item metric (§3).
Adam, full-BPTT, NumPy implementation with analytic GRU gradients (tested against
finite differences).

---

## Part 3 — Experiment H2-E1 (factorial) on Env-C

Env: `envs/continuous_env.py:EnvC` (true preference dynamics with assimilation
`alpha_pos>0`, satiation `alpha_neg<0`, drift `lam`; stochastic responses;
termination). This is the environment family where Phase 5 found MF planners
representation-bound, i.e. the strongest case for H2.

Protocol per seed (frozen before runs):
1. Log `n_train=400` trajectories under the Phase-5 behavior policy (eps-greedy on
   true match) — **logged, as in Phase 5: the planners never see `theta`**.
2. Train iALS factors; fit MF response/hazard logistics; train seq world model.
3. Evaluate `n_eval=200` held-out users closed-loop in the TRUE env (observed
   outcomes drive `f` and the next state), arms:
   `S x T = {mf, seq, hybrid} x {optimistic, stochastic}` plus `true_theta` greedy
   reference oracle (Phase-5 convention).
4. Predictive holdout metrics per arm: response log-loss / Brier / AUC;
   next-item Recall@10 (rank all items by `W_r h . E_a` for seq-type arms, by
   `u . v_a` for mf) at every history position.

Primary metric: `J(pi) = E[sum gamma^t r_t]` in the true env, `gamma=0.9`;
oracle regret `= J(oracle) - J(pi)`; paired bootstrap CIs over seeds; >= 5 seeds.

## Part 4 — Negative controls (mandatory)

Re-train the seq/hybrid world model with corrupted inputs; re-plan; re-evaluate:

| ID | Corruption | Mechanism destroyed |
|---|---|---|
| C1 | shuffle event order within each trajectory | temporal order / session structure |
| C2 | drop response from encoder inputs (`y_{t-1} := 0`) | feedback conditioning |
| C3 | replace transition action by a random item (`a_{t-1} := rand`) | action conditioning |

Response head always scores the TRUE `(h_t, a_t)`; labels unchanged.
Interpretation (owner's rule): if `hybrid+C1+C2+C3` still beats `mf` by ≈ the same
margin, the gain is not from order/feedback/action dynamics — it is popularity or
other side features — and the *mechanism* of H2 is rejected. If the margin
collapses to ≈ 0, each control contributes evidence that the corresponding
mechanism is real.

## Part 5 — Implementation (new files only; existing modules untouched)

| New | Role |
|---|---|
| `src/mf_mdp/sequential/__init__.py` | package |
| `src/mf_mdp/sequential/gru.py` | NumPy GRU cell + full-BPTT trainer, analytic gradients, Adam |
| `src/mf_mdp/sequential/world.py` | `SeqWorldModel`: embeddings, GRU, response/hazard heads, next-item readout; control modes `clean|shuffle|no_response|random_action` |
| `src/mf_mdp/planning/hybrid.py` | `HybridState`, arm configs, depth-2 stochastic/optimistic planner (matched budget) |
| `experiments/h2/run_h2_factorial.py` | E1 runner (CLI: seeds, users, controls on/off) |
| `tests/test_h2.py` | gradient check, feedback conditioning, transition correctness, optimism special case, determinism, no-leakage guard |

Reuse: `mf.py`, `mdp.py`, `envs/continuous_env.py`, `models/{features,response}.py`,
`planning/candidates.py` — unchanged.

## Part 6 — Risks and limits

1. **GRU vs SASRec faithfulness.** Mitigation: the claim under test is about state
   content, not architecture; documented above; interface allows a transformer swap.
2. **Different inductive biases per arm.** Unavoidable in any state comparison; we
   keep reward, candidates, budget, training data and the response-label stream
   identical, and report predictive metrics alongside J so a "better predictor but
   worse planner" split is visible.
3. **Env-C regime specificity.** Phase-5 conclusion that MF is representation-bound
   was itself Env-C-specific; the H2 conclusion is stated per-regime and the
   alpha sweep (follow-up, cheap) maps the boundary before any real-data claim.
4. **Estimated `q_hat`.** Shared across arms; if noisy it compresses J differences
   toward 0 (conservative for H2), not away.

---

## Part 7 — H2-E1 result (2026-11-17, branch h2/hybrid-sequential-state)

Output: `outputs/h2/h2_env_c_factorial.json` (5 seeds, 400 train users, 200 eval
users, 150 world-model epochs; runtime ~11 min local CPU).

**Verdict: H2 REJECTED on Env-C at this scale.** The hybrid state does not beat the
MF posterior, and the sequential state's distinctive information does not drive
planning value.

| Arm | J | Notes |
|---|---|---|
| mf + stochastic | **0.227 ± 0.045** | best learned arm |
| mf + optimistic | 0.157 ± 0.056 | |
| hybrid + optimistic | 0.153 ± 0.052 | ≈ mf |
| hybrid + stochastic | 0.148 ± 0.055 | significantly below mf |
| seq + stochastic | 0.097 ± 0.011 | worst |
| seq + optimistic | 0.083 ± 0.030 | worst |
| true-theta greedy (oracle ref) | 0.685 ± 0.082 | |

Contrasts (paired bootstrap over seeds): hybrid−mf @stochastic = **−0.080 CI95
[−0.155, −0.004]**; @optimistic = −0.003 ns. stoch−opt @mf = +0.071 [+0.027, +0.115]
(see below); @seq +0.014 ns; @hybrid −0.006 ns.

Predictive holdout (seed 0): mf response AUC 0.734 / ECE 0.013 vs seq AUC 0.577 /
ECE 0.073; next-item Recall@10 TIED (0.384 both) — the GRU learns item transitions
but not responses as well as the iALS-feature logistic.

Negative controls (ΔJ vs clean): seq no_response +0.018 [+0.003, +0.034] (stripping
feedback helps seq slightly), seq shuffle +0.017 ns, seq random_action −0.009 ns;
all hybrid controls ns/negative. No arm's advantage survives scrutiny because there
is no advantage to explain — the mechanism claim is moot, not supported.

**Interpretation.**
1. The MF posterior + logistic head is the better STATISTIC in this data regime
   (400 users x ~6 events): iALS gives it a strong inductive bias, the GRU trains
   from scratch. The seq arm plans badly precisely because its response model is
   poorly calibrated (planning weights branches by p_hat).
2. Hybrid ≈ mf under optimism and hybrid < mf under stochastic: adding h dilutes
   the response quality (hybrid uses the GRU head) without adding decision-relevant
   information — consistent with Env-C being "representation-bound" in a way that
   a from-scratch sequential state does NOT fix.
3. stoch−opt @mf is POSITIVE here (+0.071) whereas Phase 5 measured −0.004 under
   the TRUE env.q reward. The stochastic-transition advantage appears when item
   quality is estimated from logs, not given — a q_hat ablation is required before
   attributing this to the transition model (listed in MEMORY.md next steps).

**Caveats.** (a) Inductive-bias confound: iALS-structured features vs from-scratch
GRU; initializing E from iALS V is a cheap deconfounder. (b) Scale: 400 training
users; Env-S-style regimes with richer trajectories remain untested for H2.
(c) Single-generator regime (Env-C alpha_pos 0.35); the alpha sweep may move the
boundary, per Part 6 risk 3.

---

## Part 8 — H2-E2 grid result (2026-11-18, server, commit 424a4bd)

Output: `outputs/h2/grid/summary.json` (40 jobs = 8 cells x 5 seeds, 6 workers,
~25 min on adrianna). Cells: scale {400 (pilot_ref), 2000, 5000} x alpha_pos
{0.1, 0.35, 0.6} + einit2000_a035 (E-init-from-iALS arms) + ctrl2000_a035
(negative controls at scale) + qtrue_pilot (true-q ablation).

**Verdict: H2 is robustly REJECTED — at no scale and no dynamics strength does the
hybrid state beat the MF posterior, and the sequential state never beats either.**

1. **State factor**: hybrid−mf <= 0 in ALL cells (pilot stochastic −0.080
   [−0.156, −0.004]; every other cell ns, never positive). seq is the worst arm in
   every cell (0.09-0.15 vs mf 0.11-0.24).
2. **Scale does not rescue the sequential state**: seq response AUC falls with
   training users (0.577@400 -> 0.552@5000) while mf's rises (0.734 -> 0.765);
   next-item recall also degrades (0.385 -> 0.329 vs mf 0.383 -> 0.407). The GRU
   world model is not data-limited in a way more users fix — the iALS-feature
   logistic is simply the better statistic on this environment family.
3. **Dynamics strength**: at alpha_pos=0.6 all arms converge (0.22-0.24, ns) —
   strong assimilation homogenizes outcomes, no state representation wins. At
   alpha=0.1 everything is mediocre and equal.
4. **E-init deconfounder**: initializing the sequential embeddings from iALS V
   does not produce planning gains (seq +0.017 ns; hybrid+einit 0.176 < mf 0.186)
   and degrades the seq response model (AUC 0.477 vs 0.544 — note the hid_dim
   16-vs-32 capacity confound).
5. **Negative controls at scale (n=2000)**: all six control deltas ns — no hidden
   order/feedback/action-conditioning mechanism.
6. **q-mode ablation (the decisive side-finding)**: stoch−opt @mf = −0.002 ns under
   TRUE q vs +0.071 [+0.027,+0.115] under log-estimated q at pilot scale; and
   stoch−opt@mf shrinks monotonically with scale (+0.071 -> +0.016 -> −0.005 at
   400/2000/5000). The apparent stochastic-transition advantage is an interaction
   between OPTIMISTIC planning and ESTIMATED item quality — under the true reward
   weights we reproduce Phase 5's null exactly. Any future claim about stochastic
   transitions must control for reward-side estimation error.
7. **Ceiling**: every learned arm remains far below true-theta greedy (0.65-0.72)
   in all cells. Combined with H4 (response capacity null) and Phase 4 (leaf-value
   null), the evidence says the binding constraint on Env-C J is UPSTREAM of the
   state/response/leaf components — candidate pool, termination, horizon semantics,
   or logging-policy bias. That is the open question H2 leaves behind.

Methodological note: the H2 line's controls (C1-C3) and the q-mode cell are
reusable machinery for any follow-up line; the negative-controls protocol proved
its worth by ruling out mechanism claims cheaply.
