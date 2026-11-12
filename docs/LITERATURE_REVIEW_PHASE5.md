# Literature review — Phase 5 (stochastic world models for sequential recommendation)

Scope per prompt §2. Methods-level reading of abstracts and available full texts;
comparison table at the end. Accessed 2026-11-12.

## Closest recommender-systems work

**MCRL (Model-enhanced Contrastive RL for Sequential Recommendation, 2023)** —
[arXiv:2310.16566](https://arxiv.org/abs/2310.16566). Value function for long-term
engagement + conservative value learning against offline overestimation; contrastive
learning of reward/transition structure from state-action pairs. Neural backbone.
Overlap: learned reward/transition + value, offline. Gap: no analytic state geometry, no
controlled-environment oracle, no planning-depth/error analysis.

**ROLeR (Reward Shaping in Offline RL for RecSys, 2024)** —
[arXiv:2407.13163](https://arxiv.org/abs/2407.13163). Model-based offline RL with
non-parametric reward shaping and uncertainty penalties. Overlap: reward-model refinement,
uncertainty-aware planning. Gap: reward-model accuracy focus, no feedback-conditioned
analytic transitions, no controlled env.

**R3S (Reallocated Reward for RecSys, 2025)** —
[arXiv:2506.22112](https://arxiv.org/abs/2506.22112). Reward reallocation under world-model
uncertainty + diversity penalties (local/global state-variety decay). Overlap:
uncertainty-aware model-based offline RL. Gap: uncertainty handling is penalty-based, not
planning-depth adaptation; no analytic posterior.

**KuaiSim (2023)** — [arXiv:2309.12645](https://arxiv.org/abs/2309.12645). Comprehensive
simulator on KuaiRand: multi-behavior feedback, cross-session responses, three task levels
with protocols/baselines. Overlap: simulator-based evaluation. Gap: simulator is the
evaluation target, not a learned model of the MF posterior; no optimistic-vs-stochastic
planning comparison.

**RecSim NG (2021)** — [arXiv:2103.08057](https://arxiv.org/abs/2103.08057). Probabilistic
programmable recommender-ecosystem simulator (Edward2/TF), latent-variable agent behavior.
Overlap: controlled environments with known generators. Gap: not tied to MF posteriors;
does not study when planning helps vs greedy.

**CIRS (2022/2023)** — [arXiv:2204.01266](https://arxiv.org/abs/2204.01266),
[ACM TOIS](https://dl.acm.org/doi/10.1145/3594871). Counterfactual user model disentangling
intrinsic interest from over-exposure; KuaiEnv built on KuaiRand; offline RL against filter
bubbles. Overlap: feedback-shaped latent state, simulator-based policy evaluation.
Gap: causal deconfounding focus; no analytic MF transitions; no depth/error trade-off.

**Session-Level Optimization with REINFORCE + multi-step off-policy correction (2026)** —
[arXiv:2607.02818](https://arxiv.org/abs/2607.02818). Two-tower retriever trained
autoregressively with off-policy REINFORCE; user-feedback model for doubly-robust
sequential OPE; test-time scaling by simulating future responses. CLOSEST to our
setting: learned feedback model + simulated futures at inference. Overlap: feedback-model
rollouts at test time, long-term return objective. Gap: no tree search over posterior
states; no controlled true-environment oracle; no error-vs-depth analysis; the feedback
model is a black box, not tied to an analytic posterior.

**WMG-RL (2026)** — [arXiv:2609.01067](https://arxiv.org/abs/2609.01067). Frozen user
engagement world model (UEWM) providing reward supervision to a student policy via
counterfactual engagement simulation; LLM-scale. Overlap: world-model-generated rewards.
Gap: reward supervision for policy training, not planning; no posterior geometry.

## Model-based RL / planning

**MuZero (2019)** — [arXiv:1911.08265](https://arxiv.org/abs/1911.08265). Learned
latent dynamics/reward/value/policy; MCTS with learned model; value-equivalence in
practice. **Sampled MuZero (2021)** — [arXiv:2104.06303](https://arxiv.org/abs/2104.06303):
sampled action sets for large discrete spaces (the Wolpertinger-style reduction our
starting paper uses at expansion). **Stochastic MuZero (2021/22)** — OpenReview
X6D9bAHhBQ1: afterstate dynamics model + chance nodes + per-chance values for stochastic
environments; the reference for chance-node search we adapt. **MOReL (2020)** —
[NeurIPS](https://proceedings.neurips.cc/paper/2020/hash/f7efa4f864ae9b88d43527f4b14f750f-Abstract.html):
pessimistic MDP from offline data; performance lower-bound logic. **Value Equivalence
(2021)** — [arXiv:2011.03506](https://arxiv.org/abs/2011.03506): models need only be
value-equivalent for planning, not observation-equivalent — the theoretical justification
for planning over the MF posterior (a compressed, value-oriented state).

## Synthesis — the defensible gap

Everything above either (a) learns a fully neural world model and plans/optimizes with it
(MCRL, ROLeR, R3S, Session-Level REINFORCE, WMG-RL), or (b) builds hand-crafted simulators
(KuaiSim, RecSim NG, KuaiEnv/CIRS). The starting MF-MDP paper sits alone in planning over
the ANALYTIC iALS posterior with closed-form transitions — and our reproduction/diagnostics
showed its optimistic transition makes planning quality-inert in practice (leverage is a
popularity proxy; alignment term inert or harmful).

**Gap:** nobody has measured, in a controlled environment with a known true dynamics and
an oracle, whether FEEDBACK-CONDITIONED STOCHASTIC dynamics over the analytic MF posterior
(hybrid: learned stochastic response model + exact weighted fold-in transitions) creates
useful long-horizon planning behavior where optimistic MF planning provably fails — and how
model error accumulates with planning depth. That is our H1/H3 territory.

## Comparison table

| Paper | State | Dynamics | Feedback model | Reward | Value | Policy learning | Planning | Uncertainty | Datasets | Key limitation vs us |
|---|---|---|---|---|---|---|---|---|---|---|
| MCRL 2023 | neural seq | contrastive | implicit | contrastive R | long-term V | offline RL (conservative) | none | conservative value | 2 real | no analytic state, no oracle |
| ROLeR 2024 | neural | learned world model | implicit | non-parametric shaping | implicit | offline RL | none | penalty | 4 real | reward-model focus |
| R3S 2025 | neural | world model | implicit | reallocated | implicit | offline RL | none | uncertainty penalty | real | no depth analysis |
| KuaiSim 2023 | simulator state | hand-crafted env | multi-behavior rules | hand-crafted | n/a | n/a | n/a (evaluation) | n/a | KuaiRand/ML-1M | simulator, not learned model |
| RecSim NG 2021 | ecosystem | probabilistic program | probabilistic | configurable | n/a | n/a | n/a | Bayesian agents | synthetic | not MF-based |
| CIRS 2022 | latent interest | counterfactual causal | exposure-aware | shaped | implicit | offline RL | n/a | causal | KuaiRand | no planning-depth study |
| Session-REINFORCE 2026 | two-tower | feedback model | learned p(y) | feedback utility | implicit (DR-OPE) | REINFORCE | test-time rollout scaling | via OPE | Yambda-5B | no tree search, no true-env oracle |
| WMG-RL 2026 | LLM/UEWM | learned env model | engagement | world-model reward | implicit | distillation | none | frozen model | cross-domain | reward supervision, no planning |
| MuZero 2019 | latent | learned | env reward | learned | learned | MCTS policy improvement | MCTS | none | Atari/go | full reimplementation discouraged |
| Stochastic MuZero 2021 | latent + afterstate | learned stochastic | env reward | learned | learned | MCTS | chance-node MCTS | none | board games | no recsys, no offline |
| MOReL 2020 | given | pessimistic learned | env | env | model-based | offline RL | none | pessimism | D4RL etc. | needs offline coverage |
| **This work (Phase 5)** | **MF posterior (P,u) [+ optional GRU]** | **hybrid: learned stochastic response + exact weighted fold-in** | **learned p(y,d)** | **fixed utility / learned expected** | **V_logging vs V_π separated** | **MCTS action distributions / BC** | **stochastic lookahead + chance nodes** | **error-vs-depth analysis** | **controlled env (oracle) + KuaiRand** | — |
