#!/usr/bin/env bash
# Fetch selected outputs from the remote server.
# Usage: ./scripts/fetch_results.sh [remote_subdir, default "outputs/"]
set -euo pipefail
REMOTE="${MFMDP_REMOTE:-mf-mdp-gpu}"
DEST="${MFMDP_REMOTE_DIR:-~/work/mf-mdp/}"
SUB="${1:-outputs/}"
cd "$(dirname "$0")/.."
rsync -avz "${REMOTE}:${DEST}${SUB}" "./${SUB}"
