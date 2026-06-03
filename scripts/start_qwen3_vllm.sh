#!/usr/bin/env bash
# Start canonical Qwen3-4B vLLM serving for OpenClaw rollouts/benchmarks.
#
# Qwen3-4B does not provide a robust native OpenAI tool-call path for this
# workflow. We serve it with vLLM's hermes tool parser and apply the OpenClaw
# hermes fallback patch so <tool_call> text emitted in later turns is still
# executed as an OpenClaw tool call.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/root/openclaw-venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen3-4B}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8021}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"
DTYPE="${DTYPE:-bfloat16}"
ROPE_SCALING="${ROPE_SCALING:-{\"type\":\"dynamic\",\"factor\":2.0}}"
ENABLE_LORA="${ENABLE_LORA:-1}"
MAX_LORAS="${MAX_LORAS:-1}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
APPLY_OPENCLAW_HERMES_PATCH="${APPLY_OPENCLAW_HERMES_PATCH:-1}"

if [ "$APPLY_OPENCLAW_HERMES_PATCH" = "1" ]; then
  bash "$REPO_ROOT/scripts/apply_oc_hermes_patch.sh"
fi

ARGS=(
  -m vllm.entrypoints.openai.api_server
  --model "$MODEL_PATH"
  --served-model-name "$SERVED_MODEL_NAME"
  --host "$VLLM_HOST"
  --port "$VLLM_PORT"
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
  --dtype "$DTYPE"
  --trust-remote-code
  --enable-auto-tool-choice
  --tool-call-parser hermes
  --reasoning-parser qwen3
)

if [ -n "$ROPE_SCALING" ]; then
  ARGS+=(--rope-scaling "$ROPE_SCALING")
fi

if [ "$ENABLE_LORA" = "1" ]; then
  ARGS+=(--enable-lora --max-loras "$MAX_LORAS" --max-lora-rank "$MAX_LORA_RANK")
fi

export CUDA_VISIBLE_DEVICES
export VLLM_ALLOW_RUNTIME_LORA_UPDATING="${VLLM_ALLOW_RUNTIME_LORA_UPDATING:-True}"

echo "Starting Qwen3-4B vLLM"
echo "  python:       $PYTHON_BIN"
echo "  model:        $MODEL_PATH"
echo "  served name:  $SERVED_MODEL_NAME"
echo "  endpoint:     http://$VLLM_HOST:$VLLM_PORT/v1"
echo "  hermes patch: $APPLY_OPENCLAW_HERMES_PATCH"
echo "  lora:         $ENABLE_LORA"

exec "$PYTHON_BIN" "${ARGS[@]}"
