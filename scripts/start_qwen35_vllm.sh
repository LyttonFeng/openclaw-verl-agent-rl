#!/usr/bin/env bash
# Start canonical Qwen3.5-4B vLLM serving for OpenClaw rollouts/benchmarks.
#
# Qwen3.5-4B differs from Qwen3-4B in ways that change every serving flag:
#
#   * NATIVE TOOL CALLS (XML): Qwen3.5 emits <tool_call><function=NAME>
#     <parameter=P>...</parameter></function></tool_call>, parsed by vLLM's
#     `qwen3_coder` parser. Do NOT reuse Qwen3-4B's `hermes` parser or the
#     OpenClaw hermes fallback patch.
#   * NON-THINK by default: this workflow runs thinking OFF (cleaner output).
#     => Do NOT pass `--reasoning-parser qwen3`. With non-think the reasoning
#     parser misroutes the whole answer into `message.reasoning` and leaves
#     `content` empty. We instead patch the chat template so the DEFAULT is
#     non-think (OpenClaw/pi-ai only sends chat_template_kwargs.enable_thinking
#     when model.reasoning=true; otherwise the stock template defaults to
#     thinking-ON, which is degenerate/runaway on this model).
#   * 256K NATIVE CONTEXT: max_position_embeddings=262144, rope_scaling=None.
#     No rope/yarn needed (and rope hurt Qwen3-4B), so there is no rope flag.
#   * LOCAL WEIGHTS ONLY: the model MUST live on local disk (/tmp). The MFS
#     /workspace network disk silently corrupts large downloads (Errno5 short
#     writes) AND loads ~20x slower (21min vs ~2s). A corrupted /workspace copy
#     is exactly what made an earlier run score 0 (garbled file paths).
#     Use scripts/download_qwen35_4b.sh to fetch to /tmp first.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-/root/openclaw-venv/bin/python}"
if [ ! -x "$PYTHON_BIN" ]; then
  PYTHON_BIN="${PYTHON_BIN_FALLBACK:-python3}"
fi

MODEL_PATH="${MODEL_PATH:-/tmp/qwen3.5-4b}"   # LOCAL disk only — never MFS /workspace
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-qwen35-4b}"
VLLM_HOST="${VLLM_HOST:-127.0.0.1}"
VLLM_PORT="${VLLM_PORT:-8023}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"      # native 256K, no rope
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.90}"
DTYPE="${DTYPE:-bfloat16}"
TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-qwen3_coder}"
PATCH_TEMPLATE_NONTHINK="${PATCH_TEMPLATE_NONTHINK:-1}"
ENABLE_LORA="${ENABLE_LORA:-0}"
MAX_LORAS="${MAX_LORAS:-1}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
LORA_MODULES="${LORA_MODULES:-}"

if [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "[ERROR] $MODEL_PATH not found. Run scripts/download_qwen35_4b.sh first (local disk only)." >&2
  exit 1
fi

# Make non-think the chat-template default (idempotent).
if [ "$PATCH_TEMPLATE_NONTHINK" = "1" ]; then
  "$PYTHON_BIN" "$REPO_ROOT/scripts/patch_qwen35_template_nothink.py" "$MODEL_PATH/chat_template.jinja"
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
  --tool-call-parser "$TOOL_CALL_PARSER"
  # NOTE: deliberately NO --reasoning-parser and NO rope flag (see header).
)

if [ "$ENABLE_LORA" = "1" ]; then
  ARGS+=(--enable-lora --max-loras "$MAX_LORAS" --max-lora-rank "$MAX_LORA_RANK")
  if [ -n "$LORA_MODULES" ]; then
    ARGS+=(--lora-modules "$LORA_MODULES")
  fi
  export VLLM_ALLOW_RUNTIME_LORA_UPDATING="${VLLM_ALLOW_RUNTIME_LORA_UPDATING:-True}"
fi

export CUDA_VISIBLE_DEVICES

echo "Starting Qwen3.5-4B vLLM (non-think, native 256K, qwen3_coder tools)"
echo "  python:       $PYTHON_BIN"
echo "  model:        $MODEL_PATH"
echo "  served name:  $SERVED_MODEL_NAME"
echo "  endpoint:     http://$VLLM_HOST:$VLLM_PORT/v1"
echo "  max-model-len:$MAX_MODEL_LEN (native, no rope)"
echo "  tool parser:  $TOOL_CALL_PARSER (no reasoning-parser)"
echo "  template:     non-think default patch=$PATCH_TEMPLATE_NONTHINK"
echo "  lora:         $ENABLE_LORA"

exec "$PYTHON_BIN" "${ARGS[@]}"
