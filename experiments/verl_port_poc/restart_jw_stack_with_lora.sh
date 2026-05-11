#!/bin/bash
# Restart jiuwenclaw stack with a LoRA hot-loaded as the primary served model.
# Usage: bash restart_stack_with_lora.sh <lora_name> <lora_path>
#
# This works around colleague's stack hardcoding MODEL_NAME everywhere:
# - vLLM gets --lora-modules <lora_name>=<lora_path> as extra_args (so the
#   LoRA is registered as model_id=<lora_name>)
# - MODEL_NAME=<lora_name> so jiuwenclaw gateway uses that as model in its
#   /v1/chat/completions calls (and therefore the LoRA-augmented outputs)
# - MODEL_PATH stays as base Qwen3-4B (vLLM loads base weights, applies LoRA on top)
set -euo pipefail

LORA_NAME="${1:?lora_name required, e.g. step8-lora}"
LORA_PATH="${2:?lora_path required, e.g. /workspace/verl_port/ckpt_openclaw/global_step_8/actor/lora_adapter}"

# Kill current stack
pkill -9 -f run_online_rl 2>/dev/null || true
pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
pkill -9 -f 'jiuwenclaw.gateway' 2>/dev/null || true
sleep 8
nvidia-smi --query-gpu=memory.used --format=csv,noheader

# Inject lora-modules into the yaml the launcher writes
# Simplest: set env vars consumed by our start_jw_pod.sh, then post-edit the temp yaml after writing
export MODEL_NAME="$LORA_NAME"
export VLLM_LORA_ARGS="--lora-modules ${LORA_NAME}=${LORA_PATH}"

# Patch start_jw_pod.sh to splice VLLM_LORA_ARGS into extra_args (one-shot, only this run)
sed -i 's|VLLM_TOOL_PARSER}|VLLM_TOOL_PARSER}\n    - --lora-modules\n    - ${LORA_NAME}=${LORA_PATH}|' /root/jiuwen_work/start_jw_pod.sh || true
# Actually safer: just write a fresh launch yaml below via a temp script

# Better approach: write an inline yaml with lora-modules + invoke launcher directly
REFACTOR_ROOT=/root/jiuwen_work
VENV_PY=$REFACTOR_ROOT/jiuwenclaw/.venv/bin/python
RUN_LAUNCHER=$REFACTOR_ROOT/agent-core/examples/jiuwenrl_online/run_online_rl.py
MODEL_SNAPSHOT_DIR=$(find /root/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots -mindepth 1 -maxdepth 1 -type d | head -1)
RUN_LOG_DIR=/tmp/pinchbench_jw/logs
mkdir -p "$RUN_LOG_DIR"

CFG=$(mktemp /tmp/pb_jw_lora_XXXX.yaml)
cat >"$CFG" <<EOF
jiuwen:
  app_host: 0.0.0.0
inference:
  extra_args:
    - --enable-lora
    - --max-loras
    - "4"
    - --max-lora-rank
    - "32"
    - --enable-auto-tool-choice
    - --tool-call-parser
    - hermes
    - --max-model-len
    - "65536"
    - --gpu-memory-utilization
    - "0.7"
    - --lora-modules
    - ${LORA_NAME}=${LORA_PATH}
EOF

export HF_HOME=/root/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub

RUN_TS=$(date +%Y%m%d_%H%M%S)
LOG=$RUN_LOG_DIR/online_rl_lora_${LORA_NAME}_${RUN_TS}.log

nohup "$VENV_PY" "$RUN_LAUNCHER" \
  --config "$CFG" \
  --model-path "$MODEL_SNAPSHOT_DIR" \
  --model-name "$LORA_NAME" \
  --vllm-gpu 0,1 \
  --vllm-tp 2 \
  --vllm-port 614 \
  --gateway-port 613 \
  --redis-url redis://127.0.0.1:6379/0 \
  --jiuwen-agent-server-port 611 \
  --jiuwen-ws-port 611 \
  --jiuwen-web-port 612 \
  --jiuwen-web-host 0.0.0.0 \
  --train-gpu 0,1 \
  --threshold 999999999 --scan-interval 3600 \
  --judge-model-name "$LORA_NAME" \
  --judge-port 614 \
  >"$LOG" 2>&1 &
echo PID=$!
echo "log=$LOG"

# Wait for ws ready
echo "[lora-stack] waiting for ws to come up..."
for i in $(seq 1 180); do
  if "$VENV_PY" -c "import asyncio, websockets; asyncio.run(websockets.connect('ws://127.0.0.1:611/ws').close())" 2>/dev/null; then
    echo "[lora-stack] ws ready after ${i}s"
    break
  fi
  sleep 2
done

{
  echo "WS_URL=ws://127.0.0.1:611/ws"
  echo "MODEL_NAME=$LORA_NAME"
  echo "MODEL_PATH=$MODEL_SNAPSHOT_DIR"
  echo "GATEWAY_PORT=613"
  echo "INFERENCE_PORT=614"
} >/tmp/pinchbench_jw/logs/stack_latest.env

echo "[lora-stack] done: MODEL_NAME=$LORA_NAME"
