#!/usr/bin/env bash
# Serve a trained LoRA adapter through vLLM for manual eval/debug.

set -euo pipefail

BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-1.7B}"
LORA_PATH="${LORA_PATH:?set LORA_PATH to an actor/lora_adapter checkpoint directory}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-1.7B-task16-lora}"
PORT="${PORT:-8000}"

python3 -m vllm.entrypoints.openai.api_server \
  --model "${BASE_MODEL}" \
  --served-model-name "${SERVED_MODEL_NAME}" \
  --enable-lora \
  --lora-modules "task16=${LORA_PATH}" \
  --port "${PORT}"
