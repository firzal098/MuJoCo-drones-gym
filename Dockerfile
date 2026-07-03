# ============================================================
# MuJoCo-drones-gym — Pure dependency image (GPU / RunPod)
# ============================================================
# Sets up Python 3.10.12 + all ML libraries + JupyterLab + SSH.
# Nothing from the repo is copied. Clone manually after starting.
#
# Ports:
#   22   — SSH  (VS Code Remote SSH / Antigravity IDE)
#   8888 — JupyterLab
#
# Workflow:
#   1. Pod starts → SSH & JupyterLab auto-start
#   2. Connect via VS Code Remote SSH or open JupyterLab in browser
#   3. git clone https://github.com/firzal098/MuJoCo-drones-gym.git
#   4. python train_single_rl_sbx_mjx.py
#
# Build:
#   docker build --network host -t lolplomer/mujoco-drones-deps:latest .
# ============================================================

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

# ── Environment ──────────────────────────────────────────────
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl \
    XLA_PYTHON_CLIENT_PREALLOCATE=true \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.30 \
    JAX_COMPILATION_CACHE_DIR=/workspace/.jax_cache \
    # Once you clone the repo here, multi_drone_mujoco is importable immediately
    PYTHONPATH=/workspace/MuJoCo-drones-gym \
    UV_NO_CACHE=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/root/.local/bin:/root/.cargo/bin:$PATH

# ── System packages ───────────────────────────────────────────
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
        git \
        ca-certificates \
        # SSH server (for VS Code Remote SSH / Antigravity)
        openssh-server \
        # MuJoCo rendering
        libgl1-mesa-glx \
        libglew2.2 \
        libosmesa6 \
        libegl1 \
        libglfw3 \
        libgomp1 \
        build-essential \
    && rm -rf /var/lib/apt/lists/*

# ── Configure SSH ─────────────────────────────────────────────
RUN mkdir -p /run/sshd /root/.ssh \
    && chmod 700 /root/.ssh \
    # Allow root login with key (no password)
    && sed -i 's/#PermitRootLogin.*/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && sed -i 's/#PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config \
    && sed -i 's/#PubkeyAuthentication.*/PubkeyAuthentication yes/' /etc/ssh/sshd_config \
    && echo "AuthorizedKeysFile /root/.ssh/authorized_keys" >> /etc/ssh/sshd_config \
    # Generate host keys
    && ssh-keygen -A

# ── Install UV ────────────────────────────────────────────────
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# ── Python 3.10.12 ───────────────────────────────────────────
RUN uv python install 3.10.12 \
    && uv python pin 3.10.12

# ── Create virtual environment ────────────────────────────────
RUN uv venv /opt/venv

# ── Install all libraries ─────────────────────────────────────
RUN uv pip install \
    # Core
    "gymnasium>=0.29.0" \
    "mujoco>=3.0.0" \
    "numpy>=1.21.0" \
    # RL
    "stable-baselines3>=2.0.0" \
    "sbx-rl>=0.20.0" \
    "pettingzoo>=1.24.0" \
    # JAX / MJX / GPU physics
    "jax[cuda12]" \
    "mujoco-mjx>=3.0.0" \
    "brax>=0.14.0" \
    # Warp
    "warp-lang" \
    "mujoco-warp" \
    # Viz / media
    "matplotlib>=3.5.0" \
    "Pillow>=9.0.0" \
    "mediapy>=1.2.0" \
    "opencv-python-headless>=4.8.0" \
    # Logging / export
    "tensorboard>=2.15.0" \
    "onnxruntime>=1.16.0" \
    # Deep learning backend
    "torch>=2.0.0" \
    # JupyterLab
    "jupyterlab>=4.0.0" \
    "ipywidgets"

# ── Workspace dirs ────────────────────────────────────────────
RUN mkdir -p /workspace/results /workspace/.jax_cache

# ── Startup script (written inline — no file copy needed) ─────
RUN printf '%s\n' \
    '#!/usr/bin/env bash' \
    'set -e' \
    '' \
    '# Inject RunPod SSH public key if provided' \
    'if [[ -n "${RUNPOD_PUBLIC_KEY:-}" ]]; then' \
    '    echo "${RUNPOD_PUBLIC_KEY}" >> /root/.ssh/authorized_keys' \
    '    chmod 600 /root/.ssh/authorized_keys' \
    'fi' \
    '' \
    '# Start SSH daemon' \
    '/usr/sbin/sshd' \
    'echo "SSH started on port 22"' \
    '' \
    '# Start JupyterLab (no token — secured by SSH tunnel or RunPod proxy)' \
    'jupyter lab \' \
    '    --ip=0.0.0.0 --port=8888 --no-browser --allow-root \' \
    '    --ServerApp.token="" --ServerApp.password="" \' \
    '    --notebook-dir=/workspace &' \
    'echo "JupyterLab started on port 8888"' \
    '' \
    'echo ""' \
    'echo "=================================================="' \
    'echo "  Ready. Connect via:"' \
    'echo "    SSH     : ssh root@<pod-ip> -p <ssh-port>"' \
    'echo "    Jupyter : http://<pod-ip>:<jupyter-port>"' \
    'echo "    Then: git clone https://github.com/firzal098/MuJoCo-drones-gym.git"' \
    'echo "=================================================="' \
    '' \
    'sleep infinity' \
    > /start.sh && chmod +x /start.sh

WORKDIR /workspace

EXPOSE 22 8888

CMD ["/start.sh"]
