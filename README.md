# Planning over MF-MDPs: reproduction + reward/value model research

Reproduction and extension of **"Planning over Matrix-Factorization MDPs for Candidate Generation"**
(Trapeznikov & Utushkin, arXiv:2607.02115v2, KDD 2026 End-to-End Customer Journey Optimization workshop).

Paper: <https://arxiv.org/abs/2607.02115> (local copy: `../2607.02115v2.pdf`)

## Status

- Phase 0 (bootstrap): **in progress** — see `docs/INFRASTRUCTURE.md`, `docs/DECISIONS.md`.
- Phase 1 (reproduction): pending.
- Phase 4 (learned reward & value models): research direction — see `docs/RESEARCH_PLAN.md`.

## Layout

See `AGENTS.md`. Core logic lives in `src/mf_mdp/`, experiments in `experiments/`,
tests in `tests/`, configs in `configs/`, documentation in `docs/`.

## Quick start

```bash
python3.11 -m venv .venv && .venv/bin/pip install -e '.[dev]'
.venv/bin/pytest
.venv/bin/python experiments/run_reproduction.py --config configs/reproduction/movielens_lln.yaml
```

Data and outputs are git-ignored. Roots can be overridden with
`MFMDP_DATA_ROOT`, `MFMDP_OUTPUT_ROOT`, `MFMDP_CHECKPOINT_ROOT` (see `.env.example`).
