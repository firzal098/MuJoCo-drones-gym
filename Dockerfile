# GPU image for JAX/MJX training (CUDA 12).
# Requires: Docker with NVIDIA Container Toolkit (--gpus all).
#
# Build:  docker build -t mujoco-drones-gym .
# Run:    docker compose run --rm mujoco-drones-gym python train_single_rl_jax.py

FROM nvidia/cuda:12.6.3-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MUJOCO_GL=egl \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    XLA_PYTHON_CLIENT_ALLOCATOR=platform \
    JAX_COMPILATION_CACHE_DIR=/app/.jax_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    git \
    libgl1-mesa-glx \
    libglew2.2 \
    libosmesa6 \
    libegl1 \
    libglfw3 \
    libgomp1 \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install package deps first (better layer caching).
COPY pyproject.toml setup.py README.md ./
COPY multi_drone_mujoco/ multi_drone_mujoco/
RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install -e ".[all,fork]"

# Copy training scripts and remaining project files.
COPY *.py ./
COPY docker/ docker/

RUN mkdir -p /app/results /app/.jax_cache \
    && chmod +x /app/docker/runpod-start.sh

# Default: interactive shell (local dev). On RunPod, override with:
#   bash /app/docker/runpod-start.sh
CMD ["bash"]
