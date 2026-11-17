# Decision log

Format: date — decision — context/assumption — consequence.

- 2026-11-11 — Reproduction targets the paper text of arXiv:2607.02115v2 (6 pages, incl.
  appendices A/B). The paper does NOT report MF hyperparameters (d, alpha, lambda, iterations),
  LLN holdout count n, or planner hyperparameters (gamma, eta, c_uct, expansion width, #simulations).
  These must be chosen by us and recorded here; paper-reported values are used verbatim wherever
  they exist.
- 2026-11-11 — Reward formula is the paper's eq. (8): `r(s,a) = sim(u,v_a) + eta * 1/(1+|Delta_s(a)|)`
  (SUM form). The workspace AGENTS.md summary describes a product form `sim * (1-|Delta|)`; that
  is an error in the summary — the paper's sum form is authoritative.
- 2026-11-11 — UCT prior per paper eq. (10): `Q + c_uct * sqrt(log N(s) * (2 + cos(u_s,v_a)) / (N(s,a)+1))`.
  AGENTS.md's `2*c_uct*sqrt(U_i*logN/N)` rendering is imprecise; eq. (10) is authoritative.
- 2026-11-11 — Fold-in convention per paper eq. (2): the per-user precision sums over ALL catalog
  items; unobserved items contribute baseline confidence c=1 with preference p=0. Consequence:
  `A = lam*I + sum_all c_i v_i v_i^T` can be computed as `lam*I + V^T V + alpha * (V_H^T V_H)` where
  V_H are embeddings of the observed history; same factorization on the item side during ALS training.
- 2026-11-11 — Paper eq. (13) transition semantics: the optimistic fold-in layers a NEW
  unit-confidence observation on the baseline, b+ = b + v (NOT the naive re-fold b + (1+alpha)v
  that adding the item to history under eq. (2) would give). Precision A+ = A + vv^T matches a
  re-fold, the user vector does not. Test `test_fold_in_paper_semantics` locks this in.
- 2026-11-11 — Leverage regime: for isotropic item embeddings, A ≈ (n_items/d)·||v||^2·I in the
  typical direction, so ell = v'Pv ≲ d/n_items (ML-1M d=64: ~0.017). |Delta_s| ~ 1e-2 makes the
  alignment term eta/(1+|Delta_s|) nearly constant — Plan-1 degenerates to cosine static ranking.
  Whether the paper operated in a higher-leverage regime is unknowable from the text (MF HPs
  unreported); our HP search must acknowledge this structural bound.
- 2026-11-11 — Item-solve bug found and fixed during validation: consumer indices for the item
  ALS block were read from the wrong column of the swapped pairs array (used item ids as user
  row indices). Objective now decreases monotonically under alternation; 12/12 tests green.
- 2026-11-17 (H2) — "SASRec" operationalized as a causal GRU encoder over events
  (a_{t-1}, y_{t-1}, dt) with response/hazard heads on [h, e_a, h*e_a] and a next-item
  readout W_r h; NumPy full-BPTT (analytic gradients, finite-difference checked).
  Rationale: keep the stack CPU-reproducible; the hypothesis concerns state CONTENT
  (order/feedback/action conditioning), not attention; negative controls are
  architecture-agnostic. (docs/H2_HYBRID_SEQUENTIAL_STATE_DESIGN.md §1.3)
- 2026-11-17 (H2) — Shared reward for all factorial arms: r_hat = p_hat(y|s,a) *
  q_hat_a * max(cos(u_state, emb_a),0) * (1-kappa*f); q_hat = engagement-shrunk
  empirical per-item rate from TRAINING logs (shrink 5), NOT the env's true q —
  strictly less leaky than the Phase-5 Env-C convention (true env.q), identical
  across arms; kappa taken from env config (shared). Oracle = true-theta greedy
  (Phase-5 convention). Cold-start pool: q_hat top-L fallback.
- 2026-11-17 (H2) — MF state rebuild from observed history folds ALL shown items as
  positives (user_state convention, same as Phase-5 Env-C); the stochastic MF
  transition INSIDE planning uses Phase-5 T1 weighted fold-ins (engage (alpha,1+alpha),
  skip (-0.5*alpha,0)); sequential part always steps the GRU with the true observed y.
- 2026-11-17 (H3) — "Transformer as action generator" operationalized as SASRec proper
  (PyTorch 2.15 in .venv, CPU/MPS): causal MHSA, learned positional embeddings, pre-LN
  blocks, full-softmax next-item CE with tied embeddings (catalog 3.4k -> full softmax
  cheap, no sampling bias). Left-pad to maxlen; pad keys masked; loss masked to real
  inputs (stricter than the reference: all-pad positions never predict the first real
  item); all-pad query rows zeroed post-softmax (NaN guard). HPs: d=64 (matches MF),
  2 blocks, 1 head, maxlen 200, dropout 0.2, Adam 1e-3 (0.9, 0.98), early stop on
  validation next-item Recall@10. Torch is optional: import-guarded inside
  mf_mdp.proposal only. (docs/H3_LEARNED_ACTION_PROPOSALS_DESIGN.md §1.3, §2.1)
- 2026-11-17 (H3) — Pool-restricted Plan-K: root pool C_t from a proposer (MF exact
  top-M cosine / SASRec top-M / union / MF+rho-random); expansion, leaf value AND
  readout tail all stay inside C_t \ consumed(path) — "MCTS ищет только внутри C_t".
  INSIDE C_t the paper's exact top-l cosine rule still picks the expansion set at each
  node (branching factor = expansion_width preserved): per-sim fold-in work identical
  to the unrestricted planner, so the pool-vs-full compute comparison is fair.
  Implemented as a mirrored planK_topk in mf_mdp/proposal/planning.py so existing
  planning modules stay untouched. Planner oracle = first action of the UNrestricted
  full-catalog Plan-K (n_sim=100): the planner's own argmax, not a true-reward optimum.
  J(pi) = raw eq. (8) discounted return of the recommended slate rolled out in the
  MF-MDP model (unnormalized rewards).
