# Phase 4 design — learned reward model and value model

Owner-chosen extension, aligned with the paper's future work ("learn stochastic dynamics
and values from logs, including rejection, session termination, and delayed satisfaction"
and "use a sequence model as a value function inside the MF-MDP"). This doc is the design
contract for Phase 4; nothing here is implemented until Phase 1 reproduction is complete.

## 4.1 Learned reward model  r_hat_psi(s, a, y, c)

Separate response prediction from utility (brief §4.4). In the MF-MDP, s = (P, u), a = item.

Available signals per dataset (from DATASETS.md):
- ML-1M: rating value (>= 4 defines positives), timestamps (inter-event gaps as weak
  session/engagement proxy). No watch time, no skips.
- KuaiRec: watch ratio (graded engagement — the interesting target), timestamps.

V1 data spec (ML-1M): examples are (state, action) pairs where action = the item consumed
next in the log. Features from (P, u, v_a): cos(u, v_a), dot(u, v_a), leverage l = v'Pv,
self-shift Delta_s, ||v_a||, item popularity (train count), history length. Responses y:
for ML-1M the rating bucket (4/5) or >=4 binary; graded utility only possible on KuaiRec.

Three reward variants to compare (brief §4.4):
1. paper heuristic: r = sim + eta/(1+|Delta_s|);
2. transparent fixed mapping from feedback to utility, e.g. utility(y) with y = rating or
   watch-ratio bucket, combined with the posterior-alignment term;
3. learned reward r_hat_psi(s, a, y): predict expected utility E[utility(y) | s, a] with a
   calibrated probabilistic model (logistic / beta head), multiplied by or added to the
   alignment term — ablate the combination.

Reporting: sensitivity to utility weights; calibration (ECE, Brier) of the response head,
including temporal calibration (GTS).

## 4.2 Learned value model  V_omega(s, c)

V_omega(s) ~ E[ sum_k gamma^k r_{t+k} | s ] (brief §4.5).

V1 training data (honest about policy mismatch, brief §4.5): logged trajectories — for each
user, the sequence of consumed items defines states s_t (refold-in along the ACTUAL history,
not optimistic rollouts) and realized rewards r(s_t, a_t) under the paper's reward. MC
return-to-go provides regression targets. => V_hat estimates the LOGGING POLICY's value.
This is a known, documented bias when used as an MCTS leaf value: it biases search toward
logged-behavior-like continuations. State it in every result; do not claim unbiasedness.

V1 model: ridge/kernel regression on state features (numpy-only, no torch needed):
u (d), plus posterior-geometry features: tr(P), ||P||_F, catalog-mean leverage, history
length. V2 (server, torch): small MLP over the same features + item-popularity context.

Plug-in points (all behind config flags):
- MCTS leaf evaluation replacing closed-form rho (measure depth reduction, RQ6/H6);
- Plan-1 value: r(s+, a) + gamma * V_hat(s+) (one-step lookahead with learned leaf);
- (Phase 5) uncertainty: ensemble of V_hat for epistemic-risk leaves.

## 4.3 Experiments (after reproduction)

E1. Response-model quality on ML-1M (and KuaiRec when available): AUC/log-loss/Brier/ECE,
    temporal calibration under a GTS probe.
E2. Reward ablation: planner with heuristic vs transparent-utility vs learned expected
    utility. Metric: test Recall@10/nDCG@10 vs Static/Plan-1 baselines.
E3. Value leaf: MCTS with rho = V_hat vs rho = greedy closed form: quality vs #simulations
    and vs depth (H6: learned value reduces required search depth at comparable quality).
E4. Depth x error: |V_hat(s_k) - realized return| as a function of rollout depth k in
    optimistic rollouts vs logged rollouts (RQ3, model-error accumulation).

## 4.4 Open questions

- Is optimistic-rollout state distribution close enough to logged states for V_hat to
  transfer at all? Measure feature-space discrepancy first (cheap, decisive).
- ML-1M lacks negatives/skips: response model is "positive-only" -> consider exposure
  modeling only when KuaiRand-style data is available.
