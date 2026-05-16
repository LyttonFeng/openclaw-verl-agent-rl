#!/usr/bin/env bash
# Launch the standalone vLLM for the jiuwenclaw + Qwen3-4B benchmark / training
# rollout link. The args here matter — they were settled after multiple bench
# iterations on pod 154.54.102.42:17755.
#
# Key knobs:
#   - rope_scaling=yarn factor=2.0  → 81920 context.
#     Qwen3-4B's native max_position_embeddings is 40960, but jiuwen's system
#     prompt + tool defs + long transcripts (e.g. council_votes) hit ~80k. We
#     patch the model's config.json to advertise yarn at load time.
#   - --reasoning-parser deepseek_r1  → strip <think>...</think> from the
#     conversation history sent back on the next turn. Without this, the
#     thinking text accumulates and (a) blows up context, (b) pollutes
#     subsequent reasoning. deepseek_r1 parser handles Qwen3's <think> tag too.
#   - --tool-call-parser hermes      → required by jiuwen's tool prompt format.
#   - --enable-auto-tool-choice      → required for tool-using chat completions.
#
# Usage:
#   bash launch_vllm_qwen3_base.sh
#
# Stops if a vllm is already bound to $PORT.

set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/root/hf_cache/Qwen3-4B}"
PORT="${PORT:-8123}"
SERVED_NAME="${SERVED_NAME:-Qwen3-4B}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-81920}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-8}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.85}"
LOG_FILE="${LOG_FILE:-/tmp/vllm_qwen3_base.log}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
ROPE_FACTOR="${ROPE_FACTOR:-2.0}"          # yarn factor; 2.0 → 81920, 3.0 → 122880

if [ ! -f "${MODEL_DIR}/config.json" ]; then
  echo "[vllm] FATAL: model config not at ${MODEL_DIR}/config.json"
  echo "[vllm] Download first: hf download Qwen/Qwen3-4B --local-dir ${MODEL_DIR}"
  exit 1
fi

# Patch the model's config.json in-place to enable yarn rope scaling. vLLM 0.21
# rejects --rope-scaling on the CLI; the supported path is via config.
python3 - <<PY
import json
p = "${MODEL_DIR}/config.json"
d = json.load(open(p))
target = {"rope_type": "yarn", "factor": float(${ROPE_FACTOR}), "original_max_position_embeddings": 40960}
if d.get("rope_scaling") != target:
    d["rope_scaling"] = target
    json.dump(d, open(p, "w"), indent=2)
    print(f"[vllm] patched rope_scaling={target}")
else:
    print(f"[vllm] rope_scaling already {target}")
PY

if ss -tln 2>/dev/null | awk 'NR>1 {print $4}' | grep -qE ":${PORT}\$"; then
  echo "[vllm] port ${PORT} already bound — assuming live vllm; not restarting"
  exit 0
fi

echo "[vllm] starting vllm serve $SERVED_NAME on port $PORT (max_model_len=$MAX_MODEL_LEN)"
CUDA_VISIBLE_DEVICES="$CUDA_VISIBLE_DEVICES" \
  setsid nohup vllm serve "$MODEL_DIR" \
    --port "$PORT" --host 0.0.0.0 \
    --served-model-name "$SERVED_NAME" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization "$GPU_MEM_UTIL" \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --reasoning-parser deepseek_r1 \
    < /dev/null > "$LOG_FILE" 2>&1 &

PID=$!
echo "[vllm] PID=$PID log=$LOG_FILE — waiting up to 10min for /v1/models …"

for _ in $(seq 1 60); do
  if curl -sf -m 2 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null | grep -q "$SERVED_NAME"; then
    echo "[vllm] READY"
    exit 0
  fi
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "[vllm] FATAL: vllm crashed; last 30 lines of $LOG_FILE:"
    tail -30 "$LOG_FILE"
    exit 1
  fi
  sleep 10
done
echo "[vllm] FATAL: vllm did not become ready in 10min"
tail -30 "$LOG_FILE"
exit 1
