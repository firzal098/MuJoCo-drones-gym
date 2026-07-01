#!/usr/bin/env bash
# RunPod Pod start script for JAX training.
#
# Set in RunPod template "Container Start Command":
#   bash /app/docker/runpod-start.sh
#
# Optional env vars:
#   TRAIN_SCRIPT   default: train_single_rl_jax.py
#   RUNPOD_GIT_REPO  if set, clone/pull repo into /app before training

set -euo pipefail

cd /app

if [[ -n "${RUNPOD_GIT_REPO:-}" ]]; then
  if [[ -d /app/.git ]]; then
    git pull
  else
    git clone "${RUNPOD_GIT_REPO}" /tmp/repo
    cp -a /tmp/repo/. /app/
  fi
  pip install -e ".[all,fork]"
fi

mkdir -p "${JAX_COMPILATION_CACHE_DIR:-/app/.jax_cache}" /app/results

echo "=== GPU ==="
nvidia-smi || true
echo "=== JAX devices ==="
python -c "import jax; print(jax.devices())"

python docker/verify_install.py

TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_single_rl_jax.py}"
echo "=== Starting ${TRAIN_SCRIPT} ==="
exec python "${TRAIN_SCRIPT}" "$@"
