# Planning diagnostics — leverage / alignment mechanism

Time-boxed diagnostic phase after the ML-1M LLN reproduction (see REPRODUCTION.md).
Everything below is produced by `experiments/diagnostics/run_diagnostics.py`; artifacts:
`outputs/diagnostics/task1_leverage.json`, `task2_alignment.json`, `task3_sensitivity.json`,
`plots/*.png`. Dataset: MovieLens-1M LLN (test users; 300 users per task unless noted,
seed 0). Runtime: < 10 min total on the M4.

## TL;DR

No meaningful planning regime was found among documented iALS variants, and the mechanism is
now understood: **leverage `ℓ_i = v_iᵀPv_i` is an item-popularity/norm proxy** (Spearman up
to 1.0 / 0.96), only weakly related to user relevance (|r| ≤ 0.2). The reward's
posterior-alignment term `η/(1+|Δs|)` is therefore either **inert** (in every config where
cosine retrieval works well) or **actively harmful** (when it does move the ranking, it
re-orders by popularity, collapsing Recall@10). This closes the question the reproduction
left open: the missing planning gain is a property of the MF-MDP geometry, not a bug.

---

## Task 1 — Leverage analysis

### 1.1 Catalog leverage by MF configuration (unconsumed items, per-user states)

| config | mean | median | p90 | p95 | p99 | max |
|---|---|---|---|---|---|---|
| d64 α40 λ1 (reproduction) | 1.13e-2 | 9.0e-3 | 2.4e-2 | 2.8e-2 | 3.6e-2 | 6.6e-2 |
| d64 α10 λ1 | 1.29e-2 | 7.7e-3 | 3.3e-2 | 4.0e-2 | 5.6e-2 | 1.1e-1 |
| d64 α1 λ1 | 1.47e-2 | 3.3e-3 | 4.2e-2 | 6.6e-2 | 1.4e-1 | 3.8e-1 |
| d32 α10 λ1 | 6.4e-3 | 3.7e-3 | 1.7e-2 | 2.1e-2 | 2.7e-2 | 4.7e-2 |
| d128 α10 λ1 | 2.6e-2 | 1.5e-2 | 6.6e-2 | 8.3e-2 | 1.2e-1 | 2.4e-1 |
| d64 α10 λ0.1 | 1.3e-2 | 7.9e-3 | 3.3e-2 | 4.1e-2 | 5.6e-2 | 1.1e-1 |

Plot: `plots/task1_leverage.png` (histogram + per-config bars).

### 1.2 Trace identity and the average bound — VERIFIED

With `A = λI + Σ_{catalog} v vᵀ + α Σ_{history} v vᵀ` and `P = A⁻¹`:

```
tr(PA) = d   ⇒   λ·tr(P) + Σ_i ℓ_i + α·Σ_{i∈H} ℓ_i = d
```

Numerically exact for all 6 configs (total/d = 1.0000). Budget split (d=64, α=40):
λ·tr(P)=2.06, Σ_catalog ℓ=41.5, α·Σ_history ℓ=20.4. Because the λ- and history-terms are
nonnegative, the catalog mean satisfies **mean_i ℓ_i ≤ d/n_items** (0.0173 at d=64;
observed 0.011–0.015 i.e. 65–85% of budget). At d=128 the bound is 0.035, observed 0.026.

### 1.3 Average bound ≠ per-item bound — VERIFIED

The average bound limits the *mean*; individual items can exceed it by orders of magnitude
(max/mean ≈ 6–25× across configs). Per-item, `ℓ_i ≤ ‖v_i‖²/σ_min(A)`; measured utilization
of this bound (reproduction config): median 0.50, p90 0.57, max 0.85. So high-leverage items
exist (e.g. α=1: max ℓ = 0.38) but they are a thin tail — the alignment term acts on the
bulk of candidates where ℓ ≈ 0.01.

### 1.4 Candidate pools vs catalog

| config | catalog mean | ANN top-20 mean | top-100 mean | top-10 mean |
|---|---|---|---|---|
| d64 α40 (reproduction) | 1.13e-2 | — | 1.18e-2 | 7.2e-3 |
| d64 α10 | 1.29e-2 | — | 2.17e-2 | 2.28e-2 |
| d64 α1 | 1.47e-2 | — | 2.47e-2 | 3.43e-2 |
| d128 α10 | 2.6e-2 | — | 4.0e-2 | 4.7e-2 |

At low α, relevance-selected candidates carry ~2× the catalog-average leverage (relevance
and ℓ both load on item norm); at α=40 (homogenized norms) the effect vanishes. Task 3
shows candidate *restriction* shifts the leverage of selected actions strongly (selected
top-10 mean ℓ ranges 5.7e-3 … 4.3e-2 across variants) **without touching the MF state** —
the state `(P,u)` is built from the full-catalog precision and verified pool-invariant
(asserted in code).

### 1.5 Correlations of leverage (Pearson / Spearman, averaged over users)

| config | ‖v‖ | popularity | cos(u,v) |
|---|---|---|---|
| d64 α40 | +0.95 / +0.98 | +0.46 / +0.90 | +0.12 / +0.14 |
| d64 α10 | +0.96 / +1.00 | +0.78 / +0.96 | +0.19 / +0.17 |
| d64 α1 | +0.92 / +1.00 | +0.97 / +0.93 | +0.11 / +0.11 |
| d128 α10 | +0.95 / +1.00 | +0.85 / +0.95 | +0.14 / +0.13 |

ℓ is essentially a norm/popularity proxy; user relevance barely matters. Scatter plots:
`plots/task1_correlations.png`.

## Task 2 — Alignment contribution

Top-100 cosine pool per user; η-sweep on 500 validation users.

| config | ‖Δs‖ mean (p99/max) | std cos | std align (η=1) | ratio | Spearman(plan1, cos) | top-10 overlap (η=1) |
|---|---|---|---|---|---|---|
| d64 α40 (reproduction) | 0.004 (0.016/0.03) | 0.074 | 0.0038 | **0.051** | 0.999 | 0.994 |
| d64 α10 | 0.009 (0.035/0.06) | 0.067 | 0.0081 | **0.121** | 0.987 | 0.980 |
| d128 α1 | 0.022 (0.129/0.25) | 0.061 | 0.0251 | **0.415** | 0.810 | 0.818 |

η-sweep (top-10 overlap vs η=0 ordering / valid Recall@10):

| η | repro overlap / recall | α10 overlap / recall | d128α1 overlap / recall |
|---|---|---|---|
| 0.01 | 1.000 / 0.0860 | 1.000 / 0.0940 | 0.998 / 0.0780 |
| 0.1 | 0.999 / 0.0860 | 0.999 / 0.0940 | 0.973 / 0.0800 |
| 1 | 0.994 / 0.0860 | 0.980 / 0.0900 | 0.818 / 0.0600 |
| 10 | 0.938 / 0.0860 | 0.843 / 0.0800 | 0.352 / 0.0120 |
| 100 | 0.733 / 0.0800 | 0.399 / 0.0420 | 0.136 / **0.0000** |

Reading: the alignment term only changes ordering where its spread rivals the cosine
spread (ratio ≳ 0.4, i.e. d128α1). Where it does, quality collapses — because the term
re-orders by ℓ, i.e. by popularity (§1.5). In quality-optimal configs the term is inert at
any η that preserves recall; recall degradation begins exactly where the overlap drops
(η≈10 for α≥10 configs, η≈1 for d128α1). The tiny d128α1 bump at η=0.01–0.1 (+0.002–0.004)
is within noise (se ≈ 0.013 at 500 users). Plot: `plots/task2_alignment.png`.

## Task 3 — Embedding-geometry sensitivity (EXPLORATORY, not reproduction)

Validation Recall@10, 500 users; `lev_sel` = mean leverage of the top-10 Plan-1 actions.

| variant | lev catalog mean | lev_sel | R@10 static_cos | R@10 plan1 |
|---|---|---|---|---|
| baseline d64 α40 | 1.15e-2 | 7.3e-3 | 0.0860 | 0.0860 |
| normalized V | 1.25e-2 | 8.8e-3 | 0.0880 | 0.0880 |
| d32 α10 | 6.4e-3 | 1.1e-2 | 0.0860 | 0.0840 |
| d128 α10 | 2.6e-2 | **4.3e-2** | 0.0840 | **0.0780** |
| λ0.1 α10 | 1.3e-2 | 2.3e-2 | **0.0960** | 0.0960 |
| λ10 α10 | 1.2e-2 | 2.0e-2 | 0.0880 | 0.0900 |
| pool=20 | 1.15e-2 | 7.3e-3 | 0.0860 | 0.0860 |
| pool=1000 | 1.15e-2 | 7.3e-3 | 0.0860 | 0.0860 |
| pool=full (3706) | 1.15e-2 | 5.7e-3 | 0.0860 | **0.0440** |

Findings:
- **No variant produces a planning gain.** Best cosine quality is λ0.1 α10 (0.096) — and
  plan1 = static there too. d128 (highest leverage) makes planning *hurt*.
- **Candidate restriction matters, and cleanly**: pools up to ~1000 ANN candidates leave
  results identical; the full catalog crashes Plan-1 to 0.044. Mechanism (see math below):
  over the full catalog, thousands of low-relevance/high-ℓ items enter the pool, and
  `r(s+, a)` ranks by self-score shift Δs > 0, which is largest for high-ℓ items.
- Normalizing V does not unlock planning (cosine was already norm-free; the norm channel
  just moves into P).
- Pool restriction never changed the MF state (assert-verified): selected-action leverage
  varies 6× across variants while `(P,u)` stays identical.

## Mathematical interpretation

1. **Why ℓ is a popularity proxy.** `P = A⁻¹` with `A = λI + VᵀV + αV_HᵀV_H`. For an item
   whose embedding carries large norm or aligns with dense catalog directions, `vᵀAv` is
   large exactly where the quadratic form is large — and since the catalog contributes the
   dominant share of `A`'s spectrum, `ℓ_i = v_iᵀPv_i` measures how much energy item i
   *contributes to the geometry that contains it* — i.e. norm/popularity (§1.5).
2. **The self-boost identity.** The folded item's own score always rises when it is not yet
   fully endorsed: `Δs(a) = ℓ/(1+ℓ)·(1−uᵀv) > 0` for `uᵀv < 1`. Hence `r(s+, a)` — the
   Plan-1 'post' value — rewards items for moving *their own* score, which is monotone in
   ℓ. Over the full catalog this admits a long tail of high-ℓ/low-relevance items; over an
   ANN pool they are pre-filtered. This is a concrete, reproducible **popularity-amplification
   failure mode of the optimistic value**, complementary to the paper's popularity warning
   for dot-product *similarity*.
3. **Alignment ≠ relevance.** `|Δs|` is controlled by ℓ (popularity) times `(1−uᵀv)`
   (relevance gap). In quality-optimal configs the relevance gap is small for the top
   candidates and ℓ is uniformly tiny, so the term cannot discriminate good continuations.
   It discriminates only when ℓ is large — where it sorts by popularity.

## Computational cost

Whole diagnostic suite < 10 min on the M4 (task1 ≈ 2 min incl. 6 IALS fits; task2 ≈ 2 min;
task3 ≈ 4 min for 9 variants × fits + 500-user evaluations). No GPU used; per-user cost is
O(catalog·d²) for leverage, O(pool·d²) for Plan-1 values.

## Conclusions

1. The reproduction gap is explained: within documented iALS variants, the MF-MDP's
   alignment term is structurally inert where embeddings are good and popularity-biased
   where it is active. **No meaningful planning regime was found** (time-box honored).
2. The leverage mechanism is now fully characterized: bounds verified (average `≤ d/n`,
   per-item `≤ ‖v‖²/σ_min(A)`), correlations measured, pools analyzed, state-invariance of
   candidate restriction verified.
3. Implication for Phase 4 (owner's direction): a *learned value model* is exactly what
   this geometry cannot provide — a user-relevance-dependent estimate of future utility.
   The diagnostics give it a well-defined job: replace the inert/biased alignment term and
   the greedy closed-form leaf `ρ` with an estimate that depends on the user state, and
   measure depth-reduction (H6) under the labeling contract in PHASE4_DESIGN.md.

## Artifacts

- `outputs/diagnostics/task1_leverage.json`, `task2_alignment.json`, `task3_sensitivity.json`
- `outputs/diagnostics/plots/task1_leverage.png`, `task1_correlations.png`,
  `task2_alignment.png`, `task3_sensitivity.png`
- Runner: `experiments/diagnostics/run_diagnostics.py` (tasks selectable: `1 2 3`)

---

## Addendum 2026-11-11 (Phase 4 reward audit, prompt §4) — CORRECTION

This addendum corrects an error in the Task-3 interpretation above. The original report is
preserved unmodified above; this addendum supersedes the affected claims.

### 1. Component semantics (u vs u+) — established by code inspection + tests

| component | state used | paper-exact? |
|---|---|---|
| `reward()` (mdp.py) | the `u` argument (call-site dependent) | eq. (8) as written |
| Plan-1 canonical `paper` (was 'root') | current state `u`, Δs from fold-in | YES |
| Plan-1 `post_transition` (was 'post') | post-transition `u+` for relevance | NO — variant |
| Plan-1 `bellman` leaf term | `u+` for the leaf reward | NO — variant |
| Plan-K edge reward | edge-origin node `u` | YES |
| Plan-K closed-form leaf rho | node `u` | heuristic (documented) |
| Plan-K tail padding (readout extension) | current state (fixed in audit) | extension, not paper |

The pre-audit canonical Plan-1 ('post') used `sim(u+, v_a)` — a deviation from the paper's
`sim(u, v_a)`. Phase 4 renames variants (`paper` = exact, `post_transition` = variant,
aliases kept), defaults to `paper`, and adds tests (`tests/test_reward_audit.py`).

### 2. Monotonicity in leverage ℓ (fixed residual q = 1 − uᵀv > 0)

- Δ(ℓ) = ℓq/(1+ℓ): increasing in ℓ.
- **Paper alignment 1/(1+|Δ|): DECREASING in ℓ** — the paper term penalizes large posterior
  shifts; it does not amplify leverage/popularity.
- Post-transition relevance uᵀv + Δ: INCREASING in ℓ — this popularity-amplifying property
  belongs to the `post_transition` variant only, not to the paper reward.
- Paper reward at fixed q: decreasing in ℓ.
Tests: `test_alignment_decreases_in_leverage_fixed_residual`,
`test_post_transition_relevance_increases_in_leverage`,
`test_paper_reward_decreases_in_leverage_fixed_residual`,
`test_monotonicity_via_states_fixed_udotv`.

### 3. CORRECTION — the "full-catalog crash" was a candidate-generation bug

The Task-3 row "pool=full → 0.044" and its attribution to a "self-boost popularity
amplification" mechanism were WRONG. Root cause: `exact_cosine_topl` capped `l` at the raw
catalog size instead of the number of UNCONSUMED items, so consumed (history) items leaked
into the pool and crowded the top-10 (identical 0.0440 for both variants — the signature).
After capping `l` at n_unconsumed, full-catalog Plan-1 = **0.0860** (both variants), equal to
pool-100 and static_cosine. The "self-boost" discussion is retained only as a property of the
`post_transition` variant (§2 above), not as an explanation of any measured degradation.

### 4. Component correlations (reproduction config, 300 users, unconsumed catalog)

corr(uᵀv, ‖v‖) = +0.29, corr(uᵀv, popularity) = +0.48, corr(uᵀv, ℓ) = +0.19,
corr(uᵀv, alignment) = +0.23. Relevance loads on popularity more than on leverage; the
alignment term is mildly positively correlated with relevance (via the (1−uᵀv) factor), i.e.
it mildly favors already-endorsed items — at the catalog scale it is a tiebreaker, not a
rank driver (consistent with Task 2).
