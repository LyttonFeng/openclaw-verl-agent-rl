#!/usr/bin/env bash
# Exact vLLM serve command from the pod (probed 2026-06-10).
# Override any var via the environment, e.g.:
#   MODEL_DIR=/workspace/qwen_models/qwen3-4b-instruct-2507 serve_vllm
#   LORA_ADAPTER=/workspace/repo/results/.../lora_adapter serve_vllm
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/workspace/qwen_models/qwen3-4b}"
SERVED_NAME="${SERVED_NAME:-Qwen3-4B-base}"
HOST="${HOST:-0.0.0.0}"          # 0.0.0.0 so the container port is reachable
PORT="${PORT:-8021}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
GPU_UTIL="${GPU_UTIL:-0.85}"
DTYPE="${DTYPE:-bfloat16}"

ARGS=(
  --model "${MODEL_DIR}"
  --served-model-name "${SERVED_NAME}"
  --host "${HOST}" --port "${PORT}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_UTIL}"
  --dtype "${DTYPE}"
  --trust-remote-code
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3
)

# Optional LoRA adapter (set LORA_ADAPTER=/path[:name] to enable).
if [[ -n "${LORA_ADAPTER:-}" ]]; then
  LORA_NAME="${LORA_NAME:-r2-lora}"
  ARGS+=( --enable-lora --max-loras 1 --max-lora-rank "${LORA_RANK:-16}"
          --lora-modules "${LORA_NAME}=${LORA_ADAPTER}" )
fi

echo "[serve_vllm] python -m vllm.entrypoints.openai.api_server ${ARGS[*]}"
exec python -m vllm.entrypoints.openai.api_server "${ARGS[@]}"
