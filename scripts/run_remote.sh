#!/usr/bin/env bash
# Sync, then launch an experiment on the remote server inside tmux and print a run ID.
# Usage: ./scripts/run_remote.sh CONFIG_PATH [extra args for run_reproduction.py]
set -euo pipefail
REMOTE="${MFMDP_REMOTE:-mf-mdp-gpu}"
DEST="${MFMDP_REMOTE_DIR:-~/work/mf-mdp/}"
CONFIG="$1"; shift || true
RUN_ID="run_$(date +%Y%m%d_%H%M%S)"
./scripts/sync_to_server.sh
ssh "${REMOTE}" "bash -lc '
  set -e
  cd ${DEST}
  source .venv/bin/activate 2>/dev/null || true
  mkdir -p outputs logs
  tmux new-session -d -s mfmdp_${RUN_ID} \
    \"python experiments/run_reproduction.py --config ${CONFIG} $* > logs/${RUN_ID}.log 2>&1\"
  echo started ${RUN_ID}
'"
