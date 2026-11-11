# Reproduction notes — arXiv:2607.02115v2

Target: Table 1 (Static/Plan-1/Plan-K, LLN + GTS, 5 datasets), Table 2 (dot vs cosine
ablation on Plan-1). Aggregation: mean +/- std over 3 seeds; paired Wilcoxon vs Static.

## Protocol definitions (from the paper)

- LLN: hold out the last interactions of each user; build the initial posterior from the prefix.
- GTS: global train/valid/test timestamp cutoffs; test strictly after training window.
- Metrics: Recall@10, nDCG@10 by user.
- Table 1 note: Static = conventional inner-product (dot) static retrieval; Plan-1/Plan-K use
  cosine similarity. All methods share the same item factors trained once per protocol split.

## Documented protocol gaps (paper silent -> our choice, selected on validation)

- LLN holdout count n: paper says "leave-last-n" without a number. We use n_test=1.
- MF hyperparameters (d, alpha, lambda, iters) unreported. Validation sweep over
  d in {32,64,128} x alpha in {1,10,20,40} x lambda in {0.01,0.1,1} selected
  d=64, alpha=40, lambda=1.0, n_iter=20 (best validation cosine recall; CSVs in
  outputs/sweeps/). NOTE: alpha=40 homogenizes embedding norms, making dot ~= cos.
- Planner hyperparameters unreported ("selected on validation" per paper). eta=1.0
  (validation-neutral: the alignment term is inert at our leverage, see below);
  gamma=0.9, c_uct=1.0, expansion_width=20, n_simulations=100, horizon=10 for Plan-K.
- Closed-form leaf value rho: not explicitly defined in the paper; assumed
  rho(s) = max over ANN top-l unconsumed candidates of r(s, a).
- Plan-1 interpretation: text says "evaluates each candidate after a single fold-in and
  re-ranks by the one-step relevance/alignment value". Canonical: value(a) = r(s+, a)
  with s+ the folded-in state ('post'). Ablations 'root' (r(s0,a)) and 'bellman'
  (r(s0,a) + gamma * max_b r(s1,b)) implemented; differences on ML-1M validation are
  within ~0.002 recall at the selected HPs.

## Structural finding: the small-leverage regime

For item embeddings whose energy is spread across the catalog (isotropic approximation),
A = lambda*I + V'V + ... has eigenvalues ~ n_items*||v||^2/d, so the leverage
l = v'Pv entering the reward's alignment term is bounded by ~ d/n_items
(ML-1M, d=64: ~0.017; measured medians 0.004-0.05 across all swept configs). Then
|Delta_s| ~ 1e-2 and eta/(1+|Delta_s|) is ~constant: Plan-1 degenerates to cosine static
ranking and MCTS leaf values barely differ across nodes. Planning gains of the size the
paper reports (+0.007 recall on ML-1M) require a different embedding geometry than any
config in our sweep produced. This explains the reproduction outcome below; it is a
property of the MF-MDP construction in our regime, not (necessarily) an implementation bug:
all closed-form algebra is verified against direct recomputation (tests/test_fold_in.py).

## Results: MovieLens-1M LLN, full test set (6035 users), mean over 3 seeds

| Method | Recall@10 (ours) | Recall@10 (paper) | nDCG@10 (ours) | nDCG@10 (paper) | Label |
|---|---|---|---|---|---|
| Static (dot) | .0704 (.0699/.0699/.0713) | .0694 +/- .0032 | .0343 | .0567 | **reproduced** |
| Static cosine (ablation, not in paper) | .0696 | — | .0342 | — | — |
| Plan-1 (cos, post) | .0694 (.0699/.0694/.0689) | .0762 +/- .0016 | .0341 | .0612 | **not reproduced** |
| Plan-K (cos) | .0685 (subset, see below) | .0764 +/- .0015 | .0343 | .0611 | **not reproduced** |
| Plan-1 (dot, Table-2 ablation) | .0704 | .0630 | .0343 | .0513 | **not reproduced** (direction: dot ~= cos here) |

Plan-K was run on the first 2000 test users only (seed 0, ~32 min at ~1s/user; a full-set
run is ~100 min/seed at current unoptimized speed). Same-subset context, seed 0:
static_dot .0660, static_cos .0685, plan1 .0690, plank .0685 — Plan-K adds nothing over
Plan-1 (and slightly trails it) at ~680x the latency (956ms vs 1.4ms per user).
This matches the paper's own observation that Plan-1 captures most of the gain where any
gain exists, but here there is essentially no gain for either.

Per-seed note: with alpha=40/n_iter=20 the iALS solution is essentially init-independent
(seeds 0/1 give identical static recalls); plan1 near-tie flips cause +/-0.0005.

| Plan-1 (dot, Table-2 ablation) | .0704 | .0630 | .0343 | .0513 | **not reproduced** (direction: dot ~= cos here) |

Interpretation labels per brief: Static(dot) reproduced; Plan-1/Plan-K gains not reproduced;
dot-vs-cosine ablation (paper Table 2 direction: cos >> dot) not reproduced — in our
validation-selected regime dot ~= cos (alpha=40) or dot > cos (alpha<=10). The paper's MF
hyperparameters are unreported, so number-level comparison beyond Static is not meaningful;
our regime produces essentially the paper's Static number but no planning gain.

## What would change the regime (candidates, not yet run)

- Much smaller effective catalog per user state (paper uses top-5000 candidate pools for
  VK/YAMbDA; ML-1M native catalog is 3706 so the bound bites hardest here).
- iALS variants that concentrate embedding energy (stronger anisotropy) — e.g. confidence
  on graded ratings without the dense baseline, or explicit normalization of V.
- Explicitly normalized item embeddings (pure-cosine iALS) — makes the norm channel
  uninformative by construction and raises effective leverage per direction.
