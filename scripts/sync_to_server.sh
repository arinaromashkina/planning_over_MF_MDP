#!/usr/bin/env bash
# Sync code/config to the remote compute server (no --delete; data/outputs excluded).
# Usage: MFMDP_REMOTE=user@host ./scripts/sync_to_server.sh
set -euo pipefail
REMOTE="${MFMDP_REMOTE:-mf-mdp-gpu}"
DEST="${MFMDP_REMOTE_DIR:-~/work/mf-mdp/}"
cd "$(dirname "$0")/.."
rsync -avz \
  --exclude='.git' \
  --exclude='.venv' \
  --exclude='data/' \
  --exclude='outputs/' \
  --exclude='checkpoints/' \
  --exclude='logs/' \
  --exclude='__pycache__/' \
  --exclude='.DS_Store' \
  ./ "${REMOTE}:${DEST}"
echo "synced to ${REMOTE}:${DEST}"
