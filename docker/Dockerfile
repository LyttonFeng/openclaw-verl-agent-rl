# =============================================================================
# naive_meeting_analysis — reproducible CUDA + torch + vLLM + OpenClaw image
# =============================================================================
# Pinned to the live pod environment (probed 2026-06-10):
#   Ubuntu 22.04 · Python 3.10 · torch 2.11.0+cu130 · vllm 0.22.0
#   transformers 5.9.0 · flashinfer 0.6.11 · OpenClaw 2026.4.5 (node 22)
#
# Model weights are NOT baked in (7.6 GB). Mount them at runtime:
#   -v /host/qwen_models:/workspace/qwen_models
# or hf-download on first boot into a persistent volume.
# =============================================================================

# CUDA 13.0 to match torch 2.11.0+cu130. `devel` keeps nvcc for flashinfer JIT.
# Slim to `...-runtime-...` if you confirm flashinfer never JIT-compiles on your GPU.
FROM nvidia/cuda:13.0.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# ---- system deps + python 3.10 + node 22 -----------------------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3.10-dev python3-pip \
        git curl ca-certificates build-essential \
    && curl -fsSL https://deb.nodesource.com/setup_22.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# ---- OpenClaw (npm global, exact pod version) ------------------------------
RUN npm install -g openclaw@2026.4.5 \
    && openclaw --version

# ---- python venv -----------------------------------------------------------
RUN python3.10 -m venv /opt/venv
ENV PATH="/opt/venv/bin:${PATH}" \
    VIRTUAL_ENV="/opt/venv"
RUN pip install --upgrade pip setuptools wheel

# ---- torch stack from the cu130 wheel index (must precede vllm) -------------
RUN pip install \
        --index-url https://download.pytorch.org/whl/cu130 \
        torch==2.11.0 torchvision==0.26.0 torchaudio==2.11.0

# ---- the rest of the locked deps (vllm, transformers, flashinfer, ...) ------
# requirements.lock.txt is the pod's `pip freeze` with the torch stack and any
# private accel wheels commented out. PyPI resolves everything else.
COPY requirements.lock.txt /tmp/requirements.lock.txt
RUN pip install -r /tmp/requirements.lock.txt

# ---- workspace layout (weights + repo mounted at runtime) ------------------
RUN mkdir -p /workspace/qwen_models /workspace/repo
WORKDIR /workspace

COPY serve_vllm.sh /usr/local/bin/serve_vllm
RUN chmod +x /usr/local/bin/serve_vllm

# Default: serve the base model on :8021 (override CMD for training/rollouts).
EXPOSE 8021
CMD ["serve_vllm"]
