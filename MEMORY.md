# MEMORY.md

## Current objective

H2 (hybrid sequential belief state) on branch `h2/hybrid-sequential-state` —
**line COMPLETE, hypothesis REJECTED at every scale/dynamics regime tested**
(E1 pilot + E2 server grid). Design + full results:
`docs/H2_HYBRID_SEQUENTIAL_STATE_DESIGN.md` (Parts 7-8).
Open lines: H1 (pilot sign-off open), H3 (WIP snapshot), H4 (REJECTED, both levels).

## Status (updated 2026-11-18)

- H2-E1 (2026-11-17, local): factorial on Env-C, 5 seeds, 400 train users. H2 rejected
  at pilot scale (mf+stochastic 0.227 best; hybrid+stochastic 0.148, hybrid-mf -0.080
  CI95 [-0.155,-0.004]; seq worst 0.097; controls ns).
- H2-E2 grid (2026-11-18, server adrianna, commit 424a4bd): scale {400,2000,5000} x
  alpha_pos {0.1,0.35,0.6} + einit arms + controls + q_mode ablation, 5 seeds, 40 jobs.
  - **H2 robustly REJECTED**: hybrid-mf <= 0 in ALL 8 cells (never significantly
    positive); seq is the worst arm everywhere; seq response AUC FALLS with scale
    (0.577@400 -> 0.552@5000) while mf's rises (0.734 -> 0.765) — more data widens,
    not closes, the gap. At alpha=0.6 all arms converge (~0.23), none near oracle.
  - E-init-from-iALS deconfounder: seq +0.017 ns planning, but its AUC drops to 0.477
    (hid_dim 16 vs 32 confound noted); no arm beats mf+stochastic.
  - Controls (n=2000): all ns — no hidden order/feedback/action mechanism.
  - **q_hat ablation: stoch-opt@mf = -0.002 ns under TRUE q vs +0.071 under
    log-estimated q (pilot scale). The stochastic-transition advantage is a
    q-ESTIMATION artifact; under true q we reproduce Phase 5's null exactly.**
  - stoch-opt@mf shrinks monotonically with scale: +0.071 (400) -> +0.016 (2000)
    -> -0.005 (5000).
  - Big picture (with H4 + Phase-4 leaves): NO state/response/leaf upgrade moves
    Env-C J at any tested scale; all learned arms stay far below true-theta greedy
    (0.65-0.72). The bottleneck is upstream of the response model — open.
- Infra: server adrianna reachable again (2026-11-18; was down since 2026-11-12);
  ~/work/mf-mdp venv (numpy 2.5.4) intact; grid ran ~25 min on 6 workers.
- Repo fix on h2: .gitignore `data/` pattern silently swallowed `src/mf_mdp/data/`
  (Phase-5 Gate-E loaders) — clean checkouts could not import mf_mdp.models.value;
  anchored to `/data/`, package tracked in 424a4bd. Merge this to main eventually.
- Tests: tests/test_h2.py 10/10 green on h2; pre-existing UNRELATED failure
  tests/test_h1.py::test_greedy_differs_from_oracle_in_envP (H1 pilot open item).

## Next steps (owner decides)

1. H2 is done unless a qualitatively different regime is proposed; write up as a
   strong negative result (the q-estimation artifact finding is the publishable bit,
   together with H4's and Phase-5's nulls: "the bottleneck is upstream of the response
   model").
2. Candidate follow-ups for the upstream bottleneck: candidate-pool/termination
   modeling, horizon/readout semantics, or behavior-policy logging bias (all arms
   trained on the same eps-greedy logs).
3. If continuing H2 mechanically: KuaiRand real-data check (predictive only — H2
   already has next-item/response metrics); SASRec-exact attention under the same
   controls (low expected value given E1+E2).

## Open questions

- What DOES move Env-C J toward the oracle? Three null lines (Phase-4 leaf, H2 state,
  H4 response) + q-artifact finding suggest reward-side/quality-side estimation or
  logging-policy bias, not model capacity.
- Whether the q-estimation artifact generalizes beyond Env-C (KuaiRand has no true q;
  ML-1M rating >= 4 IS the q proxy — artifact may not apply there).
