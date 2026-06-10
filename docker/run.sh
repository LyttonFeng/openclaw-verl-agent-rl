#!/usr/bin/env bash
# Run the image on a GPU host. Weights + repo are mounted, never baked in.
#   QWEN_MODELS=/data/qwen_models  REPO=/data/openclaw-naive-meeting-analysis  ./run.sh
set -euo pipefail

REGISTRY="${REGISTRY:-ghcr.io/lyttonfeng}"
IMAGE="${IMAGE:-naive-meeting-analysis}"
TAG="${TAG:-latest}"
REF="${REGISTRY}/${IMAGE}:${TAG}"

QWEN_MODELS="${QWEN_MODELS:-/workspace/qwen_models}"   # host dir holding qwen3-4b/ etc.
REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"      # this trainer repo
PORT="${PORT:-8021}"

if [[ ! -d "${QWEN_MODELS}/qwen3-4b" ]]; then
  echo "WARN: ${QWEN_MODELS}/qwen3-4b not found."
  echo "      Mount existing weights, or hf-download first, e.g.:"
  echo "      huggingface-cli download Qwen/Qwen3-4B --local-dir ${QWEN_MODELS}/qwen3-4b"
fi

exec docker run --rm -it \
  --gpus all \
  --shm-size=16g \
  -p "${PORT}:8021" \
  -v "${QWEN_MODELS}:/workspace/qwen_models" \
  -v "${REPO}:/workspace/repo" \
  -e MODEL_DIR=/workspace/qwen_models/qwen3-4b \
  "${REF}" "${@:-serve_vllm}"

# Examples:
#   ./run.sh                                  # serve base model on :8021
#   LORA_ADAPTER=/workspace/repo/results/.../lora_adapter ./run.sh
#   ./run.sh bash                             # interactive shell for training/rollouts
