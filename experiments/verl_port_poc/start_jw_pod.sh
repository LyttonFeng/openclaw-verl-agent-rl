#!/usr/bin/env bash
# Pod-adapted JiuwenClaw stack launcher.
# Differences from colleague's pinchbench/scripts/start_jiuwenclaw_stack.sh:
#   - Use Qwen3-4B (not Qwen3-4B-Thinking-2507)
#   - Resolve model via HF cache at /root/hf_cache/...
#   - Only 2 A100-80GB available → VLLM_TP=2, VLLM_GPU_IDS=0,1
#   - Run inside jiuwenclaw uv venv so openjiuwen is importable
#   - DISABLE_ONLINE_TRAINING=1 (we only want bench, no training scheduler)

set -euo pipefail

# ─── Paths ────────────────────────────────────────────────
REFACTOR_ROOT=/root/jiuwen_work
AGENT_CORE_ROOT=$REFACTOR_ROOT/agent-core
JIUWENCLAW_ROOT=$REFACTOR_ROOT/jiuwenclaw
VENV_PY=$JIUWENCLAW_ROOT/.venv/bin/python
RUN_ONLINE_RL_PY=$AGENT_CORE_ROOT/examples/jiuwenrl_online/run_online_rl.py

# Resolve Qwen3-4B local snapshot (HF cache layout)
HF_HOME=/root/hf_cache
HF_HUB_CACHE=$HF_HOME/hub
MODEL_SNAPSHOT_DIR=$(find $HF_HUB_CACHE/models--Qwen--Qwen3-4B/snapshots -mindepth 1 -maxdepth 1 -type d | head -1)
if [[ -z "$MODEL_SNAPSHOT_DIR" || ! -f "$MODEL_SNAPSHOT_DIR/config.json" ]]; then
  echo "[start] cannot find Qwen3-4B local snapshot under $HF_HUB_CACHE" >&2
  exit 1
fi

MODEL_PATH="${MODEL_PATH:-$MODEL_SNAPSHOT_DIR}"
MODEL_NAME="${MODEL_NAME:-Qwen3-4B}"

# ─── Ports / GPU ──────────────────────────────────────────
WS_PORT="${WS_PORT:-611}"
WEB_PORT="${WEB_PORT:-612}"
GATEWAY_PORT="${GATEWAY_PORT:-613}"
INFERENCE_PORT="${INFERENCE_PORT:-614}"
JUDGE_PORT="${JUDGE_PORT:-615}"
APP_HOST="${APP_HOST:-0.0.0.0}"
WEB_HOST="${WEB_HOST:-0.0.0.0}"
VLLM_GPU_IDS="${VLLM_GPU_IDS:-0,1}"
VLLM_TP="${VLLM_TP:-2}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
VLLM_GPU_MEMORY_UTILIZATION="${VLLM_GPU_MEMORY_UTILIZATION:-0.7}"
VLLM_TOOL_PARSER="${VLLM_TOOL_PARSER:-hermes}"
TRAIN_GPU_IDS="${TRAIN_GPU_IDS:-0,1}"  # not used since DISABLE_ONLINE_TRAINING=1
SKIP_JUDGE_VLLM="${SKIP_JUDGE_VLLM:-1}"
DISABLE_ONLINE_TRAINING="${DISABLE_ONLINE_TRAINING:-1}"
RUN_LOG_DIR="${RUN_LOG_DIR:-/tmp/pinchbench_jw/logs}"
REUSE_EXISTING_STACK="${REUSE_EXISTING_STACK:-1}"

# ─── HF cache + venv ──────────────────────────────────────
export HF_HOME HF_HUB_CACHE
export TRANSFORMERS_CACHE=$HF_HUB_CACHE
export PYTHONPATH=""    # rely on venv site-packages

mkdir -p "$RUN_LOG_DIR"
RUN_TS=$(date +%Y%m%d_%H%M%S)
ONLINE_RL_LOG=$RUN_LOG_DIR/online_rl_${RUN_TS}.log
WS_URL="ws://127.0.0.1:${WS_PORT}/ws"
WEB_URL="http://127.0.0.1:${WEB_PORT}"
LATEST_META_FILE=$RUN_LOG_DIR/stack_latest.env

ws_ready() {
  "$VENV_PY" - <<PY "$WS_URL" >/dev/null 2>&1
import asyncio, sys, websockets
async def main():
    try:
        async with websockets.connect(sys.argv[1], max_size=1_000_000):
            return 0
    except Exception:
        return 1
raise SystemExit(asyncio.run(main()))
PY
}

if ws_ready; then
  if [[ "$REUSE_EXISTING_STACK" == "1" ]]; then
    echo "[start] reuse existing ws: $WS_URL"
    {
      echo "WS_URL=$WS_URL"
      echo "WEB_URL=$WEB_URL"
      echo "MODEL_PATH=$MODEL_PATH"
      echo "MODEL_NAME=$MODEL_NAME"
    } >"$LATEST_META_FILE"
    exit 0
  fi
  echo "[start] ws already in use, set REUSE_EXISTING_STACK=1 to reuse" >&2
  exit 1
fi

# ─── Build launcher args ──────────────────────────────────
REDIS_URL="${REDIS_URL:-redis://127.0.0.1:6379/0}"

LAUNCH_ARGS=(
  "--model-path" "$MODEL_PATH"
  "--model-name" "$MODEL_NAME"
  "--vllm-gpu" "$VLLM_GPU_IDS"
  "--vllm-tp" "$VLLM_TP"
  "--vllm-port" "$INFERENCE_PORT"
  "--gateway-port" "$GATEWAY_PORT"
  "--redis-url" "$REDIS_URL"
  "--jiuwen-agent-server-port" "$WS_PORT"
  "--jiuwen-ws-port" "$WS_PORT"
  "--jiuwen-web-port" "$WEB_PORT"
  "--jiuwen-web-host" "$WEB_HOST"
  "--train-gpu" "$TRAIN_GPU_IDS"
)

if [[ "$DISABLE_ONLINE_TRAINING" == "1" ]]; then
  LAUNCH_ARGS+=("--threshold" "999999999" "--scan-interval" "3600")
fi

TMP_CFG=$(mktemp /tmp/pb_jw_online_rl_XXXX.yaml)
cat >"$TMP_CFG" <<CFG
jiuwen:
  app_host: ${APP_HOST}
inference:
  extra_args:
    - --enable-lora
    - --max-loras
    - "4"
    - --max-lora-rank
    - "32"
    - --enable-auto-tool-choice
    - --tool-call-parser
    - ${VLLM_TOOL_PARSER}
    - --lora-modules
    - ${LORA_NAME}=${LORA_PATH}
    - --lora-modules
    - ${LORA_NAME}=${LORA_PATH}
    - --lora-modules
    - ${LORA_NAME}=${LORA_PATH}
    - --max-model-len
    - "${VLLM_MAX_MODEL_LEN}"
    - --gpu-memory-utilization
    - "${VLLM_GPU_MEMORY_UTILIZATION}"
CFG
LAUNCH_ARGS=("--config" "$TMP_CFG" "${LAUNCH_ARGS[@]}")

if [[ "$SKIP_JUDGE_VLLM" == "1" ]]; then
  # Reuse inference vLLM as judge; still pass --judge-port (validator requires it)
  LAUNCH_ARGS+=("--judge-model-name" "$MODEL_NAME" "--judge-port" "$INFERENCE_PORT")
else
  LAUNCH_ARGS+=("--judge-model-name" "$MODEL_NAME" "--judge-model-path" "$MODEL_PATH" "--judge-port" "$JUDGE_PORT")
fi

cd "$REFACTOR_ROOT"

echo "[start] launching stack:"
echo "  model: $MODEL_PATH ($MODEL_NAME)"
echo "  vllm: gpu=$VLLM_GPU_IDS tp=$VLLM_TP port=$INFERENCE_PORT mem=$VLLM_GPU_MEMORY_UTILIZATION"
echo "  ws=$WS_PORT web=$WEB_PORT gateway=$GATEWAY_PORT"
echo "  log=$ONLINE_RL_LOG"

nohup "$VENV_PY" "$RUN_ONLINE_RL_PY" "${LAUNCH_ARGS[@]}" >"$ONLINE_RL_LOG" 2>&1 &
ONLINE_RL_PID=$!
echo "$ONLINE_RL_PID" >"$RUN_LOG_DIR/online_rl_latest.pid"
ln -sfn "$ONLINE_RL_LOG" "$RUN_LOG_DIR/online_rl_latest.log"

echo "[start] pid=$ONLINE_RL_PID, waiting for ws ready (~3-5 min for vLLM CUDA graph)..."

READY=0
for i in $(seq 1 300); do
  if ! kill -0 "$ONLINE_RL_PID" 2>/dev/null; then
    echo "[start] online-rl exited early, log tail:" >&2
    tail -n 60 "$ONLINE_RL_LOG"
    exit 1
  fi
  if ws_ready; then
    READY=1
    break
  fi
  sleep 2
done

if [[ "$READY" != "1" ]]; then
  echo "[start] ws not ready after 10 min, see $ONLINE_RL_LOG" >&2
  exit 1
fi

{
  echo "ONLINE_RL_PID=$ONLINE_RL_PID"
  echo "WS_URL=$WS_URL"
  echo "WEB_URL=$WEB_URL"
  echo "MODEL_PATH=$MODEL_PATH"
  echo "MODEL_NAME=$MODEL_NAME"
  echo "GATEWAY_PORT=$GATEWAY_PORT"
  echo "INFERENCE_PORT=$INFERENCE_PORT"
} >"$LATEST_META_FILE"

echo "[start] OK pid=$ONLINE_RL_PID ws=$WS_URL web=$WEB_URL"
