#!/usr/bin/env bash
# ============================================================
# RunPod auto-start script for MuJoCo-drones-gym
# ============================================================
# Runs automatically on container start (set as CMD in Dockerfile).
# Clones the repo then idles — run training scripts manually via SSH.
#
# PYTHONPATH=/workspace/MuJoCo-drones-gym is set in the image,
# so `import multi_drone_mujoco` always uses the live cloned source.
#
# Optional env var override:
#   RUNPOD_GIT_REPO — override the repo URL
# ============================================================

set -euo pipefail

DEFAULT_GIT_REPO="https://github.com/firzal098/MuJoCo-drones-gym.git"
REPO_DIR="/workspace/MuJoCo-drones-gym"
GIT_REPO="${RUNPOD_GIT_REPO:-${DEFAULT_GIT_REPO}}"

# ── 1. Clone or pull project source ──────────────────────────
if [[ -d "${REPO_DIR}/.git" ]]; then
    echo "=== Pulling latest source ==="
    git -C "${REPO_DIR}" pull
else
    echo "=== Cloning ${GIT_REPO} ==="
    git clone "${GIT_REPO}" "${REPO_DIR}"
fi

# ── 2. Ensure runtime directories exist ──────────────────────
mkdir -p /workspace/results /workspace/.jax_cache

# ── 3. Quick sanity check ────────────────────────────────────
echo ""
echo "=== GPU ==="
nvidia-smi || echo "(nvidia-smi not available)"
echo ""
echo "=== Python & JAX ==="
python --version
python -c "import jax; print('JAX devices:', jax.devices())"
echo ""
echo "=== multi_drone_mujoco source ==="
python -c "import multi_drone_mujoco; print(multi_drone_mujoco.__file__)"

# ── 4. Idle — run training manually via SSH ───────────────────
echo ""
echo "=========================================================="
echo "  Container ready. SSH in and run your training script:"
echo "    cd /workspace/MuJoCo-drones-gym"
echo "    python train_single_rl_sbx_mjx.py"
echo "=========================================================="
sleep infinity
