# H4 design — Cross-Feature Response Models for MF-MDP Planning

Research design for hypothesis line **H4** on branch `h4/cross-response-model`.

Owner hypothesis (2026-11-17, verbatim): *«Замена логистической регрессии на модель с
взаимодействиями. Нелинейные взаимодействия между состоянием пользователя, кандидатом и
контекстом должны улучшать модель отклика там, где один и тот же item по-разному
воспринимается в зависимости от истории и уверенности состояния. … Гипотеза должна
проверяться на двух уровнях: предсказательном (log-loss, Brier, ECE, AUC на временном
сплите и randomized-exposure subset) и decision-level (улучшает ли более точный отклик
истинный return планировщика). Улучшение AUC само по себе ещё не означает улучшения
планирования.»*

> Numbering note: "H4" is a NEW hypothesis line, distinct from the Phase-5 "H2"
> (learned reward/value), the Phase-5 "H3" (depth vs model accuracy), and from the
> `h1/…` (belief vs preference dynamics), `h2/…` (hybrid sequential state) and
> `h3/…` (sequential action proposals) lines. The five lines are complementary:
> H1 asks *what the transition misses about the user*; H2 asked *whether the state
> itself is the bottleneck* (rejected on Env-C at pilot scale); H3 asks *whether a
> sequential model helps most as a search-space reducer*; **H4 asks whether the
> response model itself — not the state, not the search — is under-expressive:**
> the current `p̂(y|s,a)` is logistic on 10 additive MF features, so it cannot
> represent "the same item is perceived differently depending on history and on
> state confidence".

---

## Part 1 — Hypothesis and framing

### 1.1 Hypothesis

**H4.** The response model enters the Phase-5/H2 planner both as the branch
probability of the stochastic transition and as a multiplicative factor of the
expected reward (`r̂ = p̂ · q̂ · cos⁺ · (1−κf)`). The logistic baseline is additive
in its features, so any modulation of candidate perception by state confidence or
history must be approximated linearly — e.g. `cos(u, v_a)` is shrunk toward the
prior for short histories (the iALS posterior mean is attenuated when `tr(P)` is
large), and the optimal de-attenuation factor is state-dependent, i.e. an
*interaction*. Models with grounded nonlinear interactions (named cross-features,
Factorization Machine, small MLP, conditionally DCN-v2) therefore predict held-out
responses better **and** improve closed-loop planner return `J(π)` in the true
environment, relative to the logistic baseline at matched training data, candidate
pools and search budget.

Formally, with `M` a response-model family and `π(M)` the resulting depth-2
stochastic planner, H4 claims:

1. (predictive) `LL_test(M*) < LL_test(lr)` and `AUC_test(M*) > AUC_test(lr)` for
   the best interaction model `M*`, on a chronological split AND on a
   randomized-exposure subset, where gains concentrate in the cold-start / low-
   confidence regime (early session positions, high `tr(P)`);
2. (decision) `J(π(M*)) > J(π(lr))` in the true environment with paired CI95
   excluding 0.

**Falsification criteria** (pre-registered):

- If no interaction model beats the logistic baseline on predictive metrics on
  the temporal split → H4 rejected at the predictive level; decision-level test
  becomes moot (nothing to transmit).
- If predictive gains exist but `J(π(M*)) − J(π(lr))` has CI95 covering 0 →
  H4 rejected at the decision level; the owner's caveat ("AUC ≠ planning") is
  supported, reported as the finding.
- If `BASE+HIST` linear additions explain the gains (i.e. `M1 ≈ M0h` on crosses,
  only `M0h > M0`) → the interaction claim specifically is rejected; only "more
  features" is supported.
- If ungrounded random crosses of matched count (`C1`) match the grounded crosses
  (`M1`) → the *grounding* claim is rejected (generic capacity is doing the work),
  even if both beat `M0h`.
- If FM/MLP beat `M1` while `M1 ≈ M0h` → generic learned capacity, not named
  interactions, drives any gain — the "meaningful interactions" motivation is
  rejected in favor of a capacity story.

### 1.2 Why this is grounded in our own results

- Phase 4 E2 (`docs/PHASE4_RESULTS.md`): the ML-1M response model is logistic on
  exactly the 10 additive `features_for` features; AUC ≈ 0.62 with ECE ≈ 0.007 —
  calibrated but weakly discriminative. Nothing in the model class lets perception
  of an item depend on how confident the posterior is about the user.
- Phase 5 Gate E (`outputs/phase5/kuairand_transition.json`): next-click AUC from
  the MF posterior is 0.592, worst at session position 0 (cold start, 0.511) and
  ECE grows with session depth — exactly the slices where posterior confidence
  varies most. This is the regime H4's interactions target.
- H2-E1 (`outputs/h2/h2_env_c_factorial.json`): the MF-state stochastic arm was
  the best learned planner on Env-C (`J=0.227`); its response model was the Phase-4
  logistic. The planner pipeline (`planning/hybrid.py`, matched-budget depth-2
  lookahead) is reused unchanged here so the ONLY moving part is `p̂`.
- The owner's caveat is itself testable here: Env-C gives the true `J(π)` in
  closed loop, so predictive-vs-decision transmission is directly measurable.

### 1.3 The "dot(u,v) is already bilinear" remark

`dot(u, v) = 1ᵀ(u ⊙ v)` is itself an aggregated bilinear term, so adding one more
generic product term has no scientific motivation. H4 therefore distinguishes:

- **Grounded crosses** — the named, mechanistically motivated interactions of
  §2.3. These are the hypothesis.
- **Generic capacity** — FM latent crosses / MLP units over the same base vector.
  These are controls on the capacity axis, not the hypothesis.
- **Random crosses** — products of randomly drawn base-feature pairs, matched in
  count to the grounded set. These isolate "any product terms at all" vs "these
  product terms". Per the owner's remark, `M1 > C1` is required to claim the
  grounded interactions matter.

---

## Part 2 — Model ladder and interfaces

### 2.1 Models

| Id | Model | Features | Role |
|---|---|---|---|
| `M0`  | Logistic L2 (`models/response.py`) | BASE | Phase-4/5 baseline, Gate-E comparable |
| `M0h` | Logistic L2 | BASE+HIST | linear history access (ablation isolator) |
| `M1`  | Logistic L2 | BASE+HIST+GROUNDED | **the hypothesis** |
| `C1`  | Logistic L2 | BASE+HIST+RAND×K | ungrounded control, K = #grounded |
| `M2`  | Factorization Machine, 2nd order | BASE+HIST | generic learned crosses |
| `M3`  | MLP (1 hidden, tanh) | BASE+HIST | generic nonlinearity |
| `M3'` | MLP | BASE+HIST+U⊙V | explicit `h_t ⊙ v_a` (interaction #5) |
| `M4`  | DCN-v2 | BASE+HIST | **conditional**, see §2.2 |

All models are NumPy CPU implementations (server SSH still unreachable; repo
convention). Common interface: `fit(X, y)`, `predict_proba(X)`, fitted inside the
same `FeatureEncoder` normalization pipeline as the baseline (fit on train only).

### 2.2 DCN-v2 conditional trigger

M4 is implemented **only if** both hold on the validation slice: (a) `M3` beats
`M1` on log-loss by more than the seed-to-seed std, and (b) the `M2`/`M3` gains
are not explained by any single grounded cross (checked by dropping one cross at
a time). Rationale: DCN-v2 adds an explicit cross-network layer; it is justified
only when there is evidence of interaction structure beyond both the named crosses
and FM's pairwise rank-k approximation. The trigger outcome is recorded in the log
whether or not M4 is built.

### 2.3 Feature sets and the grounded crosses

BASE (10, `models/features.py:features_for` order):
`[cos, dot, leverage ℓ, |Δs|, ‖v‖, log_pop, hist_len, tr(P), ‖P‖_F, ‖u‖]`.

HIST — history/context scalars (decision-time legal; leakage contract unchanged):
- KuaiRand: `[mean_wr_hist, log1p(recency_min+1), repeat_cnt, author_repeat_cnt,
  hour, log_duration, video_type, session_pos]`
- Env-C: `[mean_eng_hist, repeat_cnt]` (recency is uniform by construction —
  discrete steps, documented infeasible for interaction #2)

GROUNDED crosses (owner's list, operationalized):

| # | Owner's interaction | KuaiRand | Env-C |
|---|---|---|---|
| 1 | `cos(u,v) × hist_len` | ✔ | ✔ |
| 2 | relevance × recency | `cos × exp(−recency_min/τ)`, τ=30min — in addition to raw recency feature | infeasible (no continuous time), documented |
| 3 | watch ratio of past events × similarity | `mean_wr_hist × cos` | `mean_eng_hist × cos` |
| 4 | uncertainty × item popularity | `ℓ × log_pop` | `ℓ × log_pop` |
| 5 | `h_t ⊙ v_a` | via `M3'` (u⊙v dims appended); NOT an explicit LR cross (dot already aggregates it) | same |
| 6 | fatigue/repeatability × closeness to recent events | `repeat_cnt × cos`, `author_repeat_cnt × cos` | `repeat_cnt × cos` (constant 0 — no repeats in Env-C trajectories; kept for interface parity, flagged inert) |

U⊙V note: for `M3'` the raw `u ⊙ v` (d=16) is appended to the model input only
(non-additive access for the MLP); BASE/HIST models never see it, keeping the
ladder comparable.

RAND×K: K products of distinct uniformly-drawn BASE+HIST feature pairs (fixed
seed, pairs frozen between train/test).

---

## Part 3 — Level 1: predictive evaluation (real data)

### 3.1 Primary testbed: KuaiRand-1K

Why: the only dataset we hold with (a) confirmed-exposure semantics, (b) a
chronological two-week split, (c) a randomized-exposure log (the `random` file,
`is_rand=1`) — the owner's "randomized-exposure subset" — and (d) graded
engagement (watch ratio) for interaction #3.

Protocol (mirrors `experiments/phase5/run_kuairand_transition.py` for
comparability):

- iALS: `d=16, α=40, λ=1, it=10` on train-period click=1 positives restricted to
  the top-5000 exposure-popularity pool (paper convention; same as Gate E).
- Sessions ≥ 5 events, positions 0..7 (`MAX_POS=8`); train = standard_1 sessions
  (cap 40k), validation = last-two-days subset of standard_1 (20220420–21) for HP
  selection, test = standard_2 sessions (cap 25k), probe = `random` file sessions.
- Labels: primary `wr80` = `play_time/duration ≥ 0.8` over ALL confirmed
  exposures (graded engagement; non-clicks have play_time 0 → valid negatives);
  secondary `long_view`, `click` (Phase-5 targets, policy-bound by design).
- Metrics per model per subset: log-loss, Brier, ROC-AUC, ECE; ECE-by-date on
  test (drift); per-session-position AUC/ECE on test (mechanism slice: the
  interaction claim predicts the largest lift at positions 0–2).
- HPs on validation only: FM `k ∈ {4,8,16}`, MLP hidden ∈ {32,64}, lr/epochs for
  FM/MLP from a small grid; logistic `l2=1e-4, 400 steps` (Phase-5 default).
- C1 control and M0h/M1 decomposition per §2.1.

### 3.2 Secondary testbed: ML-1M (Phase-4 E2 protocol)

`build_decision_dataset` (LLN prefixes, user-disjoint split, label rating=5) with
crosses #1, #3 (`mean_hist_rating × cos`), #4; #2 timestamps exist but the Phase-4
protocol is order-based — documented, not built; #6 mostly inert (re-consumption
is rare). Purpose: check the interaction value is not KuaiRand-specific. Quick
(CPU seconds).

### 3.3 Decision rule (level 1)

`M1 > M0h` on test log-loss with paired bootstrap over users AND `M1 ≥ C1`
(non-inferiority margin = bootstrap std); mechanism slice consistent (largest lift
at low-confidence positions). Otherwise the specific interaction claim fails even
if generic models win.

---

## Part 4 — Level 2: decision evaluation (Env-C, true `J(π)`)

Protocol is the H2-E1 pipeline with the state arm FIXED to `mf` and transition
FIXED to `stochastic` (the mode where `p̂` actually drives planning; under
`optimistic`, `_p ≡ 1` and the response model is unused — verified in
`planning/hybrid.py:_p`). Everything else identical to `experiments/h2/run_h2_factorial.py`:

- 5 seeds; 400 train users (behavior-logged trajectories), 200 eval users;
  iALS `d=16, α=10, λ=0.1, it=15`; `q_hat` engagement-shrunk from logs; hazard
  model shared LR across ALL arms (hypothesis is about the response model only);
  reward `r̂ = p̂ · q̂_a · cos⁺(u, v_a) · (1−κf)`; depth-2 lookahead, pool `L=8`,
  matched budget by construction (same planner code path).
- Arms: `resp ∈ {M0 lr, M1 lr_x, C1 lr_randx, M2 fm, M3 mlp, M3' mlp_uv}`; plus
  `optimistic+lr` (Phase-5 reference) and `true_greedy` (oracle reference).
- The planner-side feature builder recomputes BASE+HIST+GROUNDED from the live
  fold-in state each call (subclass of `HybridPlanner`, h4-local; existing
  modules untouched).
- Primary metric: closed-loop `J(π)` in the TRUE env; per-arm mean±std over
  seeds; contrasts `arm − lr` with paired bootstrap CI95 over seeds. Env-C
  response-model holdout AUC/ECE recorded per arm per seed (transmission check).

### 4.1 Pre-registered reading

- **H4 supported**: level-1 decision rule passes AND at least one interaction
  model has `J − J_lr` CI95 > 0.
- **Transmission gap (owner's caveat)**: level-1 passes, `J` CIs cover 0 →
  predictive accuracy does not transmit to planning value at this scale; reported
  as the main finding.
- **J moves without level-1 support** → treated as suspect (noise / leakage /
  planner-specific quirk), investigated before any claim.

### 4.2 Honest limitations

- Env-C's true response surface is `sigmoid(a·cos(θ,v) + b·q + ε)` — interactions
  enter only through posterior shrinkage of `u`, so the decision-level testbed
  rewards exactly the confidence-modulation mechanism, not arbitrary crosses.
  KuaiRand is the arbiter of whether such structure exists in real data.
- `M3'` appends `u ⊙ v`; with `d=16` this is 16 extra inputs — capacity differs
  from `M3` slightly; reported alongside, not laundered into the main contrast.
- 5 seeds × 200 eval users is the H2-E1 budget; power is what it was there.

---

## Part 5 — Implementation plan

New code (existing modules untouched, per repo convention):

- `src/mf_mdp/interactions/__init__.py`
- `src/mf_mdp/interactions/fm.py` — `FactorizationMachine` (log-loss, minibatch
  SGD, fixed seed; vectorized pairwise term)
- `src/mf_mdp/interactions/mlp.py` — `MLPClassifier` (1 hidden layer, tanh)
- `src/mf_mdp/interactions/cross.py` — grounded-cross builders (Env-C / KuaiRand),
  `random_crosses`, feature-name registry
- `src/mf_mdp/interactions/planner.py` — `CrossPlanner(HybridPlanner)` overriding
  the mf feature path with BASE+HIST(+crosses)(+U⊙V)
- `experiments/h4/run_kuairand_cross.py` — level 1 (KuaiRand)
- `experiments/h4/run_ml1m_cross.py` — level 1 (ML-1M secondary)
- `experiments/h4/run_env_c_cross.py` — level 2 (Env-C)
- `tests/test_h4.py` — FM closed-form term vs brute force; FM/MLP finite-difference
  gradient checks; determinism under fixed seed; cross-builder sanity (no NaN,
  correct count, leakage-free by construction); `CrossPlanner` parity with
  `HybridPlanner` when crosses are empty and model is LR; `random_crosses` pair
  freezing between fit/predict.

Outputs: `outputs/h4/kuairand_cross.json`, `outputs/h4/ml1m_cross.json`,
`outputs/h4/env_c_cross.json`.

Docs: this file; `EXPERIMENT_LOG.md` row; `DECISIONS.md` entries for every new
assumption; `MEMORY.md`; `AGENTS.md` research-branch section.

## Part 6 — Risks

1. **iALS retrain per run** — Gate-E-scale iALS (≈1.9M clicks, d=16) is minutes on
   the M4; acceptable, time-boxed.
2. **Feature-build runtime** — capped sessions (40k/25k) keep the walk at Gate-E
   scale (~2–4 min/log).
3. **FM/MLP optimization fragility** — mitigated by gradient tests and a small
   validation grid; failures are reported, not silently re-tuned on test.
4. **Multiple comparisons across arms** — pre-registered primary contrasts
   (`M1−M0h`, `M1−C1`, `arm−lr` on J); secondary slices are labelled exploratory.
5. **Negative result** — if nothing moves, the report states which rung of the
   ladder failed (features vs crosses vs capacity vs transmission), matching the
   pre-registered falsification structure.
