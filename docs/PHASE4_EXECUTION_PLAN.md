# Phase 4 execution plan — reward/value foundation

Branch: `phase4/reward-value-v1` (from `fb0411b`). Prompt: `../Phase4_Reward_Value_Model_Execution_Prompt.md`.
Design contract: `docs/PHASE4_DESIGN.md` (labeling contracts enforced). Diagnostics: `docs/PLANNING_DIAGNOSTICS.md`.

## 0. Gate: mathematical audit of reward semantics (§4 of prompt)

- Inventory which component uses `u` vs `u+` (code inspection, table, tests).
- Monotonicity of each reward component in leverage (fixed residual q = 1 - u'v):
  alignment 1/(1+|Delta|) is DECREASING in l; post-transition relevance u'v + Delta is
  INCREASING in l; paper reward (fixed sim) is decreasing in l.
- Component-wise correlations (u'v vs ||v||, popularity, l; relevance vs alignment).
- Attribute the full-catalog Plan-1 degradation to the exact code path ('post' variant,
  post-transition relevance) with a controlled rerun ('paper'/'root' on full catalog).
- Dated addendum to PLANNING_DIAGNOSTICS.md; no overwrite.

## 1. Data audit and label semantics (§5.1)

MovieLens-1M LLN provides per user: chronologically ordered positive interactions
(rating >= 4), ratings 4/5, timestamps. NO exposure fields, NO watch ratio, NO skips,
NO session boundaries. Consequences (documented, not fabricated):
- Feedback label for response modeling: y = rating in {4, 5} on CONFIRMED interactions
  (consumed items). These are exposed-and-accepted events; nothing is known about
  exposures or rejections. Response model p(y=5 | s,a) is defined ONLY on the
  consumption-support of the logging policy — selection bias is inherent and reported.
- No skip/dislike/termination labels -> those response types are infeasible on ML-1M
  (marked infeasible; KuaiRec watch-ratio planned for a later phase).
- Session boundary: none in data; returns are computed per user over the full observed
  sequence (truncated at the LLN cut). Document censoring.

## 2. Reward variants (§5.2)

- R0 — paper heuristic, PAPER-EXACT semantics: r(s,a) = sim(u,v_a) + eta/(1+|Delta_s(a)|),
  sim at the CURRENT state. Plan-1 canonical = 'paper' variant (renamed from 'root').
  The 'post' variant (post-transition relevance) is kept, named 'post_transition'.
- R1 — response distribution + fixed utility: p(y=5|s,a) logistic regression on
  decision-time features (see §4); U(5)=1, U(4)=u4 with u4 in {0, 0.25, 0.5} swept on
  validation only; expected reward = p*1 + (1-p)*u4, optionally + alignment term
  (ablated). Expected-utility estimates logging-conditional satisfaction, not exposure.
- R2 — direct learned expected utility: ridge regression of observed U(y) on the same
  decision-time features. Compared against R1's distribution-then-expectation.
- Features (decision-time only, no leakage): cos(u,v_a), dot(u,v_a), l = v'Pv,
  |Delta_s|, ||v_a||, log popularity, user's history length; plus V-state features
  (tr(P), ||P||_F) for the state-dependent parts. Normalization fitted on train only.
- Utility weights: fixed a priori, sensitivity reported; chosen on validation only.

## 3. V_logging v1 (§6)

- Logged trajectories: per user, states s_t re-folded along the ACTUAL consumed sequence
  (prefix of the LLN prefix). MC return-to-go G_t = sum_k gamma^k r_{t+k} with r = R0
  paper-exact reward of the realized (s_t, a_t); gamma = 0.9 (configurable).
- Split: users disjoint train/valid/test (e.g. 70/15/15, seeded); normalization and
  ridge lambda fitted/selected on train/valid only. Temporal order preserved within users.
- Estimator: ridge regression (closed form). MAE/RMSE/R^2 on held-out users; reliability
  plot data saved. Named V_logging everywhere; never V_target.
- Known biases (documented in every result): logging-policy conditional value;
  consumption-support only; optimistic fold-in states are OFF-distribution for V_logging
  (measured, see extrapolation diagnostic E4).

## 4. MCTS leaf integration (§7)

- `leaf_evaluator` config: `original` (default, current behavior, bit-exact) |
  `zero` | `heuristic` (greedy rho, current explicit) | `v_logging_ridge`.
- G_leaf = sum_{t<h} gamma^t r_t + gamma^h V(s_h); terminal = session end adds nothing.
- Disabled-by-default; disabling reproduces current Plan-K (test).
- Value scale: R0 reward is unbounded-ish (cos in [-1,1] + eta); V_logging trained on the
  same reward scale -> compatible by construction; normalization stored, applied at predict.

## 5. Experiments (§8), all tuned on validation only

E1 Reward ablation: Plan-1 with R0 / R1(u4 grid) / R2 (+/- alignment term), valid 500
  users x 3 seeds -> test for the chosen variant. Baselines: static dot/cos, paper Plan-1.
E2 Response model quality: AUC/log-loss/Brier/ECE of p(y=5|s,a) on held-out users;
  calibration by history-length segment.
E3 Value quality: V_logging MAE/RMSE/R^2 vs closed-form leaf rho heuristic as a predictor
  of realized returns (diagnostic comparison, labeled as logging-policy estimate).
E4 Leaf ablation in Plan-K: leaf_evaluator in {original, zero, v_logging_ridge} x
  horizon {1, 3, 5, 10} at fixed simulations (subset of users, 1-3 seeds, labeled);
  quality + latency + leaf-eval counts. Extrapolation check: feature-space distance of
  MCTS-visited states to logged training states, reported with E4.
E5 (report-only) Off-policy limitations section: IPS/SNIPS/DR discussion deferred —
  no exposure support on ML-1M; labeled as diagnostic, not policy improvement.

## 6. Interfaces (§9)

ResponseModel.predict_distribution / UtilityModel.utility / ExpectedRewardModel.predict_reward /
ValueModel.predict_value — plain dataclass-backed callables in src/mf_mdp/models/, numpy-only,
type-hinted, config-selected. No new heavy dependencies.

## 7. Tests (§10)

Reward audit tests (u vs u+ per component), monotonicity tests (alignment decreasing in l;
post-transition relevance increasing in l), utility mapping, expected-reward discrete math,
return-to-go incl. truncation/terminal, leakage (features use only past), user-split
isolation, ridge normalization/regularization, value interface, MCTS leaf accumulation and
discounting, terminal leaf, disabled=original bit-exact, determinism. Existing 12 tests
must keep passing unmodified.

## 8. Acceptance criteria (§13)

Audit table + addendum committed; R0/R1/R2 implemented with labels feasible on ML-1M
(skip/dislike explicitly infeasible); V_logging leakage-safe and never named V_target;
leaf integration opt-in with bit-exact original default; ablations run with quality+compute;
all tests pass; PHASE4_RESULTS.md documents limitations.

## 9. Files

Add: src/mf_mdp/models/{__init__,response,utility,reward,value}.py,
experiments/phase4/{run_reward_ablation.py,run_value_model.py,run_leaf_ablation.py},
tests/test_reward_audit.py, tests/test_reward_models.py, tests/test_value_logging.py,
tests/test_mcts_leaf.py, docs/PHASE4_RESULTS.md (later).
Change: src/mf_mdp/planning/planners.py (paper-exact variant naming, leaf_evaluator hook),
src/mf_mdp/mdp.py (none expected), configs (new phase4 configs only),
docs/{PLANNING_DIAGNOSTICS.md addendum, DECISIONS.md, EXPERIMENT_LOG.md, MEMORY.md}.
