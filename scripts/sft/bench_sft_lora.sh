#!/bin/bash
# Bench Qwen3-4B + SFT LoRA on PinchBench val_5 × 3 runs.
#   - Launches vLLM with the LoRA adapter
#   - Runs scripts/benchmark.py with localhost vLLM as the model
#   - Compares to base (0.49) and DSv4 Flash teacher (0.89)
#
# Usage:
#   bash bench_sft_lora.sh <lora_dir> <out_dir>
set -euo pipefail

LORA_DIR="${1:?Usage: bench_sft_lora.sh <lora_dir> <out_dir>}"
OUT_DIR="${2:?Usage: bench_sft_lora.sh <lora_dir> <out_dir>}"
BASE_MODEL="${BASE_MODEL:-/workspace/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c}"
VLLM_PORT="${VLLM_PORT:-8765}"
SERVED_NAME="${SERVED_NAME:-qwen3-sft}"

mkdir -p "$OUT_DIR"
LOG_DIR="$OUT_DIR/logs"
mkdir -p "$LOG_DIR"

# Kill any existing vLLM
pkill -9 -f vllm 2>/dev/null || true
pkill -9 -f VLLM::EngineCore 2>/dev/null || true
sleep 3

echo "[$(date +%T)] Starting vLLM on port $VLLM_PORT with LoRA $LORA_DIR"
nohup python3 -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL" \
  --served-model-name "$SERVED_NAME" \
  --port "$VLLM_PORT" \
  --max-model-len 65536 \
  --rope-scaling '{"rope_type":"yarn","factor":2.0,"original_max_position_embeddings":32768}' \
  --enable-lora \
  --lora-modules "${SERVED_NAME}-lora=${LORA_DIR}" \
  --max-lora-rank 64 \
  --gpu-memory-utilization 0.85 \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice --tool-call-parser hermes \
  > "$LOG_DIR/vllm.log" 2>&1 &
VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"

# Wait for vLLM health
echo "Waiting for vLLM to be ready..."
for i in $(seq 1 120); do
  if curl -s "http://localhost:$VLLM_PORT/v1/models" 2>/dev/null | grep -q "$SERVED_NAME"; then
    echo "vLLM ready (after ${i}×5s)"
    break
  fi
  sleep 5
done

if ! curl -s "http://localhost:$VLLM_PORT/v1/models" | grep -q "$SERVED_NAME"; then
  echo "ERROR: vLLM didn't come up in 10min — see $LOG_DIR/vllm.log"
  exit 1
fi

# Source bench env
set -a; source /root/.pinchbench_env; set +a
export PINCHBENCH_FORCE_LOCAL_OPENCLAW=1
export OPENCLAW_HOST=localhost

SUITE="task_meeting_advisory_stakeholders,task_meeting_council_votes,task_meeting_gov_speaker_summary,task_meeting_tech_action_items,task_meeting_sentiment_analysis"

cd /workspace/openclaw-verl-agent-rl
for r in r1 r2 r3; do
  echo "[$(date +%T)] === SFT LoRA $r ==="
  mkdir -p "$OUT_DIR/$r"
  python3 scripts/benchmark.py \
    --model "${SERVED_NAME}-lora" \
    --base-url "http://localhost:$VLLM_PORT/v1" \
    --api-key dummy \
    --judge deepseek-chat \
    --suite "$SUITE" \
    --timeout-multiplier 3 --no-upload \
    --output-dir "$OUT_DIR/$r" \
    > "$OUT_DIR/${r}.log" 2>&1
done

# Summary
python3 - <<PYEOF | tee "$OUT_DIR/summary.txt"
import json, glob, statistics
scores = []
per_task = {}
for f in glob.glob("$OUT_DIR/r*/0*.json"):
    d = json.load(open(f))
    scores.append(d["category_scores"]["MEETING"]["pct"] / 100)
    for p in d["efficiency"]["per_task"]:
        per_task.setdefault(p["task_id"], []).append(p["score"])
print(f"SFT LoRA OVERALL mean: {statistics.mean(scores):.4f}")
print(f"SFT LoRA OVERALL std:  {statistics.stdev(scores) if len(scores)>1 else 0:.4f}")
print(f"n runs: {len(scores)}")
print(f"BASELINE base 4B: ~0.49 / DSv4 Flash teacher: ~0.89")
print()
print("Per-task:")
for t in sorted(per_task):
    s = per_task[t]
    print(f"  {t:50s} mean={statistics.mean(s):.3f} n={len(s)}")
PYEOF

# Cleanup
kill $VLLM_PID 2>/dev/null || true
echo "DONE — SFT bench results: $OUT_DIR"
