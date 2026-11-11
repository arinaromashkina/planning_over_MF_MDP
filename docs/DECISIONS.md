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
