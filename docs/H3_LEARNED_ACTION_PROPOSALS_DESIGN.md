# H3 design — Sequential Model as Action Proposer for the Planner

Research design for hypothesis line **H3** on branch `h3/sequential-action-proposals`.

Owner hypothesis (2026-11-17, verbatim): *"Трансформер как генератор действий, а не как
ещё один рекомендатель. Последовательная модель может сокращать пространство действий
для планировщика, сохраняя перспективные с точки зрения долгосрочной награды кандидаты
и уменьшая стоимость поиска. То есть трансформер используется не для финального next-item
ranking, а для построения candidate set: C_t = TopM_a p_theta(a|h_t). Далее MCTS ищет
только внутри C_t. Основной риск — candidate bottleneck: фильтр может удалить действие,
которое не выглядит релевантным сейчас, но открывает хорошую последующую траекторию.
Поэтому сравнивать нужно: MF ANN top-M; SASRec top-M; объединение MF и SASRec top-M;
случайное расширение части пула для exploration. Метрики: попадание оптимального
oracle-действия в пул; Recall@M; итоговый J(pi) при фиксированном числе симуляций;
качество при фиксированной задержке. Это отдельная гипотеза о quality–compute
trade-off, а не обязательно о повышении качества рекомендаций."*

> Numbering note: "H3" is a NEW hypothesis line, distinct from the Phase-5 "H3"
> (depth vs model accuracy, `docs/PHASE5_HYPOTHESES.md`), from `h1/…` (belief vs
> preference dynamics) and `h2/…` (hybrid sequential state). The four lines are
> complementary: H1 asks *what the transition misses*; H2 asks *whether the state is
> the bottleneck*; **H3 asks whether a sequential model is more useful as a search
> space reducer than as a recommender** — it is a hypothesis about the
> quality–compute trade-off of planning, not about final recommendation quality.

---

## Part 1 — Hypothesis and framing

### 1.1 Hypothesis

**H3.** Let `C_t = TopM_a p_theta(a | h_t)` be the candidate pool proposed by a
sequential model trained for next-item prediction, and let the MF-MDP planner (Plan-K
MCTS) search **only inside** `C_t`. Then, at a fraction of the per-user compute of
full-catalog MCTS:

- (H3-a, coverage) `C_t` retains the planner-relevant actions: the action an
  unrestricted planner would choose ("planner oracle") and the ground-truth next
  item both land in `C_t` with high probability, at pool sizes `M << n_items`;
- (H3-b, trade-off) MCTS-over-`C_t` reaches planning quality (`J(pi)`) and final
  recommendation quality (Recall@10 / nDCG@10) statistically indistinguishable from
  full-catalog MCTS at fixed simulation budget, while its latency is lower by roughly
  the pool-size ratio;
- (H3-c, complementarity) the sequential proposer contributes candidates the MF ANN
  pool misses (and vice versa), so `union(MF, SASRec)` covers strictly more than
  either alone at matched pool budget — and this complementarity is NOT reproduced by
  replacing the SASRec half with uniformly random exploration items.

**Falsification criteria** (pre-registered):
- If planner-oracle coverage of the SASRec pool at matched `M` is ≤ MF-ANN coverage
  AND final quality is lower → the sequential proposer is not a better action
  generator → H3-a/b rejected (the model may still be a fine recommender — reported
  separately as next-item Recall@10 of the raw model).
- If `union ≈ MF` at matched pool budget → no complementarity → H3-c rejected.
- If `MF + rho·M random` recovers union-level coverage → complementarity is plain
  exploration, not sequential signal → H3-c mechanism rejected.
- If restricted MCTS quality at fixed latency is WORSE than full-catalog MCTS at the
  same latency → the pool restriction destroys search efficiency → H3-b rejected.

### 1.2 Why this is grounded in our own results

1. **Planning is compute-bound.** Reproduction: Plan-K at `n_sim=100` costs ~0.96 s/user
   (EXPERIMENT_LOG "plank test") — ~100 min per full LLN evaluation per seed, ~680×
   Plan-1. Any candidate reduction that preserves search value has direct practical
   value at VK-catalog scale (top-5000 pool, paper §4).
2. **Planning gain on ML-1M is absent in our leverage regime** — Plan-1 ≈ Static cosine
   (REPRODUCTION.md). Therefore ML-1M cannot demonstrate *quality gains* from better
   pools; it CAN demonstrate the quality–compute trade-off (H3-b), which is the
   owner's stated primary question. Quality-gain claims are deferred to a catalog/
   regime where planning itself moves metrics (explicitly out of scope here).
3. **H2's seq arm** (`src/mf_mdp/sequential/`, numpy GRU) showed a sequential state
   adds nothing as a *belief state* on Env-C. H3 tests the orthogonal use of sequence
   models: as *action proposers*. Different mechanism, different claim.

### 1.3 Scope of "SASRec" and the transformer claim

The owner names SASRec. Unlike H2 (which stayed NumPy-only because SSH to the GPU
server was down), H3 installs **PyTorch** into `.venv` (CPU/MPS) and implements
SASRec proper: causal self-attention over item embeddings, learned positional
embeddings, pre-LN residual blocks, dropout, full-softmax next-item cross-entropy
with tied input/output embeddings (Kang & McAuley, ICDM'18; Wang et al., SIGIR'21
for the CE variant). Catalog is 3.4k items, so full softmax is cheap and sampling
bias (BPR/popularity-sampled) is avoided. Torch is an *optional* dependency: imported
only inside `mf_mdp.proposal`, tests skip when absent.

---

## Part 2 — Model and candidate pools

### 2.1 SASRec training setup (fair to the MF side)

- Data: ML-1M LLN protocol identical to the reproduction (`min_rating=4`,
  `n_test=1`, `n_valid=1`). Training sequences = each user's `prefix`
  (chronological positives); teacher-forcing next-item CE. No valid/test leakage:
  valid/test items are never inputs or targets during training.
- HPs (SASRec paper defaults for ML-1M): `d=64` (matches MF dim), 2 blocks, 1 head,
  `maxlen=200`, dropout 0.2, Adam lr 1e-3 (β=(0.9, 0.98)), batch 128, ~200 epochs max
  with early stopping on **validation next-item Recall@10** computed exactly like the
  eval states (prefix → score last position, rank against unconsumed catalog).
- Checkpoint to `checkpoints/sasrec_ml1m_seed{S}.pt` (git-ignored); training script
  `experiments/h3/train_sasrec.py`.

### 2.2 Candidate pools (all exclude consumed items; sizes `M ∈ {25, 50, 100, 200}`)

| Key | Definition |
|---|---|
| `full` | no pool — unrestricted MCTS (the paper's top-l ANN per node; reference) |
| `mf_ann` | exact top-M by cosine(u_t, e_i) — the paper's ANN step made exact (catalog 3.4k) |
| `sasrec` | top-M by p_θ(i \| h_t) from the last prefix position |
| `union`  | `TopM(mf_ann) ∪ TopM(sasrec)`, deduped (size ≤ 2M) |
| `mf_rand`| top-`⌈(1−ρ)M⌉` by MF cosine ∪ `⌊ρM⌋` uniform over remaining unconsumed, ρ ∈ {0.1, 0.25} |

`M ≥ expansion_width l=20` so the root can branch fully. Pool is computed **once per
user at the root** (cost included in latency); the whole tree searches inside
`C_t \ consumed(path)` — "MCTS ищет только внутри C_t" (owner's formulation). Inside
`C_t`, expansion still applies the paper's exact top-`l` cosine rule at the node's
current `u` (branching factor = `expansion_width` preserved), so per-simulation
fold-in work is identical across arms — the compute effect comes solely from the
shrunken candidate domain, and the iso-compute comparison stays fair.

### 2.3 Planner (fixed across arms)

`PlannerConfig` reproduction defaults: `eta=1`, `gamma=0.9`, `c_uct=1`,
`expansion_width l=20`, `horizon K=10`, `n_simulations=100`, cosine similarity.
Pool-restricted MCTS is a new module mirroring `planning/planners.py:planK_topk`
(expansion provider injected); existing modules stay untouched.

---

## Part 3 — Metrics and experiment stages

**Pool metrics** (cheap, all ~6k test users):
- `gt_coverage@M` — P(test item ∈ C_t). This is the proposer's Recall@M against the
  LLN test holdout (owner's "Recall@M").
- `valid_coverage@M` — same against validation (used only for HP sanity).

**Planner-oracle coverage** (expensive, subset n=300, computed once per user):
- `planner_oracle` = first action of the **unrestricted** full-catalog Plan-K policy
  (n_sim=100, seed 0). `oracle_coverage@M` = P(planner_oracle ∈ C_t). This is the
  owner's "попадание оптимального oracle-действия в пул" — the action the planner
  itself considers optimal must survive the filter (candidate-bottleneck check).
- Reported per pool and per M; also `oracle_rank_in_pool` (position of the oracle
  action in pool order, when present).

**Policy metrics** (fixed n_sim=100, subset n=1000 test users, 1 seed initially):
- `J(pi)` — trajectory return of the recommended slate rolled out in the MF-MDP
  model: `R(tau) = Σ_{k=0..9} γ^k r(s_k, a_k)`, eq. (8) raw (unnormalized) rewards.
- `recall_at_10`, `ndcg_at_10` — paper metrics vs the LLN test holdout.
- `latency_mean_s`, `latency_p95_s` — wall-clock per user, pool construction included
  for proposer arms.

**Fixed-latency iso-compute** (same subset):
- Measure quality-vs-`n_sim` for each arm; then report the quality of each arm at the
  latency of the *slowest* arm (i.e. give every arm the same wall-clock budget by
  scaling its n_sim; full-catalog baseline gets its native budget). Also the inverse:
  n_sim each arm needs to match full-catalog quality (if it does).

**Compute note.** Local Mac only (server SSH down). Stages are sized so the full H3
suite finishes in a few hours: pools ~minutes; policy n=1000 × 6 arms ≈ 1–2 h at the
reproduction per-sim cost (pool-restricted arms are strictly cheaper than full);
oracle subset n=300 ≈ 5 min extra per configuration of full-catalog runs (reused
across pools/M).

## Part 4 — Implementation (new files only; existing modules untouched)

```
src/mf_mdp/proposal/__init__.py     # package docstring
src/mf_mdp/proposal/sasrec.py       # torch SASRec + fit/eval loops (torch optional)
src/mf_mdp/proposal/pools.py        # proposer functions + coverage metrics
src/mf_mdp/proposal/planning.py     # pool-aware Plan-K MCTS + J(pi) rollout
experiments/h3/train_sasrec.py      # training entrypoint (checkpoint)
experiments/h3/run_proposals.py     # stages: pools | oracle | policy | latency
tests/test_h3.py                    # unit tests (skip torch-less)
docs/H3_LEARNED_ACTION_PROPOSALS_DESIGN.md   (this file)
```

Reused unchanged: `data/movielens.py`, `mf.py`, `mdp.py`, `planning/candidates.py`,
`planning/planners.py`, `evaluation/metrics.py`.

## Part 5 — Risks and limits

- **ML-1M regime**: planning gain is absent here (leverage bound, REPRODUCTION.md), so
  pools cannot improve final quality; the H3 question on this dataset is purely
  "does search survive shrinkage, and at what compute saving". Generalizing
  quality-gain claims to planning-active regimes needs a regime where Plan-K ≠
  Static (future work; KuaiRec is the natural next candidate once the server is back).
- **Oracle definition**: "planner oracle" is the planner's own argmax under unlimited
  search, NOT the true-reward optimum (unknowable without env model). It is exactly
  the candidate-bottleneck quantity the owner cares about; ground-truth coverage is
  reported alongside.
- **Union pool size**: union doubles the budget (≤2M). We report it both raw and
  size-matched (union@M vs each source @M, plus union of top-M/2 each — noted in the
  results table).
- **Torch dependency**: optional, contained in `mf_mdp.proposal`; CI-without-torch
  skips SASRec tests.

## Part 6 — Results

(filled in after the runs; see EXPERIMENT_LOG.md)
