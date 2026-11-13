# Phase 5 results — stochastic world models for sequential recommendation

Branch history: `phase5/stochastic-world-model` (merged to `main` at fbcf9af).
Prompt: `../Phase5.md`. Hypotheses: `docs/PHASE5_HYPOTHESES.md`.
Plan: `docs/PHASE5_EXPERIMENT_PLAN.md` (frozen). Audit: `docs/PHASE5_RESEARCH_AUDIT.md`.
Artifacts: `outputs/phase5/*.json`, `figures/*.png|.pdf`, article: `paper/article.tex|.pdf`.

## Gate status

| Gate | Content | Status |
|---|---|---|
| A | research validity (audit, literature, H1–H3, frozen protocol) | PASS |
| B | Env-S correctness (greedy≠optimal, MCTS convergence, Bellman checks) | PASS |
| C | learned world model (response/transition training, leakage-safe) | PASS |
| D | planning experiment (matched budgets, true-env J, ablations, error-vs-depth) | PASS |
| E | KuaiRand-1K (inspection, response models, transition validation) | PASS (with documented limitations) |

## H1 — stochastic vs optimistic planning

Controlled environments with known dynamics and oracles; `J` measured in the TRUE
environment; 5 seeds; paired bootstrap CIs.

| Environment | State | Optimistic `J` | Stochastic `J` | Δ (CI95) | Verdict |
|---|---|---|---|---|---|
| Env-S (12-state, exact DP oracle) | faithful | 0.523 ± 0.012 | **0.645 ± 0.007** | **+0.122 [0.115, 0.129]** | oracle regret 0.006 — **H1 supported** |
| Env-C (θ-dynamics, MF posterior) | compressed | 0.217 ± 0.059 | 0.213 ± 0.054 | **−0.004 [−0.010, 0.002]** | **H1 not supported** |

Interpretation: stochastic feedback-conditioned dynamics pay off **only when the state
representation carries the dynamics**. Over the compressed MF posterior the transition
model's sophistication is irrelevant — the representation dominates the error.

## Gate B details (planning correctness)

Greedy ≠ DP-optimal (J gap 0.084, CI95 [0.069, 0.099]); chance-node MCTS converges to
the DP choice as budget grows (residual = statistically tied states). Cross-checks
caught two real bugs: (i) the DP oracle initially ignored assimilation failure;
(ii) an earlier candidate-generation leak (consumed items in ANN pools) — both fixed
and regression-tested.

## H2 — components

On ML-1M (Phase 4, feeding Phase 5): learned rewards R1/R2 lose to the paper heuristic
(response signal AUC 0.62, popularity-flavored); `V_logging` on deployable features ≈
mean predictor (R² 0.019); in Env-S the fixed transparent utility suffices because the
utility function is known by construction; learned value as MCTS leaf was not binding in
Env-S (exact DP leaf available). Components are justified where labels carry signal
(KuaiRand watch-ratio) — see Gate E.

## H3 — depth × model error

Env-S grid (response logit noise σ ∈ {0.5, 1.5, 3.0}; train trajectories {250, 1000,
3000}; depths {1, 2, 3, 6}):

| σ | depth-2 regret | depth-6 regret (3000 traj) | depth-6 regret (250 traj) |
|---|---|---|---|
| 0.5 | 0.110 | 0.013 | 0.025 |
| 1.5 | 0.124 | 0.001 | 0.026 |
| 3.0 | 0.134 | 0.009 | 0.041 |

Deeper planning cuts regret 5–15× once the model is reasonable; estimation error stops
binding once the response model saturates (transition structure exactly known here);
no "depth-hurts" regime appears in this family. The curve supports **adaptive depth** as
the next direction (shallow default, deepen when the model is reliable), not deeper
search unconditionally.

## Gate E — KuaiRand-1K

Inspection (documented in `outputs/phase5/kuairand_inspection.json`): 5.06M + 6.66M
standard rows (click ≈ 0.38), 43k randomized-exposure rows (`is_rand=1`; click 0.174 —
selection bias in standard logs confirmed); watch-ratio median 0.87 on clicks; no
missing values; no session id (30-min gap rule documented).

Response models (chronological split; `outputs/phase5/kuairand_response.json`):

| Target | Test AUC | Test ECE | Random-subset AUC | Note |
|---|---|---|---|---|
| click | 0.607 | 0.234 | 0.515 | policy-bound; temporal miscalibration |
| long_view | 0.627 | 0.104 | 0.542 | |
| wr80 on clicks | **0.839** | **0.043** | **0.826** | graded engagement transfers |

Transition validation, full scale on the L4 server (all eligible sessions, no caps;
`outputs/phase5/kuairand_transition_full.json`): MF posterior states predict the next
in-session click with test AUC 0.586 (train 0.659), log-loss 0.647, BCE 0.227, ECE
0.044 (base rate 0.635). Per-position AUC is flat in 0.57–0.61 (cold start is not
special at full scale); calibration degrades late in sessions (ECE 0.099 at position 7).
On real data, as in Env-C, the MF representation is the bottleneck.

## Limitations (explicit)

- Env-S/Env-C are handcrafted; the regime contrast indicates the role of the state
  representation but is not a strict single-variable isolation.
- Simulator results are diagnostics, not evidence of real-user improvement.
- No OPE claims: KuaiRand propensities unverified; ML-1M has no exposure labels.
- `V_logging` never `V_target`; target-policy value unimplemented.
- The transition validation is full-scale on the server (43.7k train / 5.0k test pool
  examples); the per-position analysis uses the first 8 positions of each session.

## Conclusions

1. **H1 is conditionally true**: stochastic planning beats optimistic planning where the
   state is faithful (+0.122, significant), and is indistinguishable from it over the
   MF posterior (−0.004).
2. **H3**: depth helps once the model is reasonable; adaptive depth is the supported
   next step, not unconditional deeper search.
3. On real KuaiRand data the MF posterior carries weak next-response signal — the same
   representation bottleneck the controlled experiments isolate.

## Next steps

- Adaptive-depth planning (Phase 5+): confidence-based deepening with the H3 curve as
  the prior.
- Server-scale runs (SSH currently unavailable from the dev machine): full KuaiRand
  sessions, GRU state encoder (S1), MCTS at scale.
- V_target via planner-rollout distillation in the validated controlled simulator.
