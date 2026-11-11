# Infrastructure

## Local (primary development workspace)

- Apple Silicon Mac M4, macOS arm64.
- Python 3.11.14 at /opt/homebrew/bin/python3.11 (system python3 is 3.9, unsuitable).
- Project venv: `.venv/` (git-ignored) with numpy, scipy, pyyaml, pytest, pypdf.
- All core code is CPU/NumPy; no CUDA/MPS dependency for the reproduction.

## Remote compute server

- Connection (from the research brief): `ssh root@176.53.248.106` (8 vCPU, 32 GB RAM,
  200 GB disk, 1 x NVIDIA L4 24 GB). Optional alias `mf-mdp-gpu` (needs owner approval).
- Status: SSH OK (BatchMode key auth, 2026-11-11). Inspected without changes:
  hostname `adrianna`, Linux x86_64, 31 GiB RAM (30 available), 184 GiB disk free,
  NVIDIA L4 23034 MiB, driver 615.71.09, system Python 3.14.4 (use a project venv
  instead for torch compatibility).
- Intended use: response/reward/value model training, ensembles, KuaiRec-scale experiments.

## Sync and remote runs

- `scripts/sync_to_server.sh` — rsync code/config to server (no --delete; data/outputs excluded).
- `scripts/run_remote.sh` — placeholder workflow: sync -> env check -> run in tmux -> run ID.
- `scripts/fetch_results.sh` — copy selected outputs back.

## Experiment tracking

Outputs under `MFMDP_OUTPUT_ROOT` (default `./outputs`), layout:
`outputs/<experiment>/<dataset>_<split>/run_<id>/{config.yaml,metrics.json,environment.json,stdout.log,stderr.log}`.
Never overwrite prior runs silently; >= 3 seeds where feasible.
