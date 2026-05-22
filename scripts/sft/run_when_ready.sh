#!/bin/bash
# Wait for DSv4 Flash train_23 bench to finish, then auto-run SFT data prep +
# training + bench. Designed to be nohup'd overnight.
set -euo pipefail

TRAIN_BENCH_DIR="${1:?Usage: run_when_ready.sh <train_bench_dir> [val_bench_dir]}"
VAL_BENCH_DIR="${2:-}"
SFT_BASE="${SFT_BASE:-/workspace/verl_port/sft_v1}"

mkdir -p "$SFT_BASE"
LOG="$SFT_BASE/runner.log"
exec > >(tee -a "$LOG") 2>&1

echo "=========================================="
echo "[$(date +%T)] SFT auto-runner starting"
echo "  train bench: $TRAIN_BENCH_DIR"
echo "  val bench:   $VAL_BENCH_DIR"
echo "  out base:    $SFT_BASE"
echo "=========================================="

# 1. Wait for all 5 runs to have a *.json (each run finishes by writing the result JSON)
echo "[$(date +%T)] Waiting for train_23 r1..r5 to finish..."
while true; do
  done_count=$(ls "$TRAIN_BENCH_DIR"/r*/*.json 2>/dev/null | wc -l)
  echo "[$(date +%T)] runs done: $done_count / 5"
  [ "$done_count" -ge 5 ] && break
  sleep 600  # check every 10 min
done

echo "[$(date +%T)] All 5 runs complete. Starting SFT pipeline."

# 2. Run data pipeline (include val bench too if available)
SFT_DATA_DIR="$SFT_BASE/data"
mkdir -p "$SFT_DATA_DIR"

BENCH_DIRS=("$TRAIN_BENCH_DIR")
[ -n "$VAL_BENCH_DIR" ] && BENCH_DIRS+=("$VAL_BENCH_DIR")

cd /workspace/openclaw-verl-agent-rl
TOP_K=2 MIN_SCORE=0.5 MAX_TOOL_CHARS=6000 DROP_ABOVE=22000 \
  bash scripts/sft/run_pipeline.sh "$SFT_DATA_DIR" "${BENCH_DIRS[@]}"

DATA_FILE="$SFT_DATA_DIR/chatml.jsonl"
N_TRAIN=$(wc -l < "$DATA_FILE")
echo "[$(date +%T)] Final training data: $DATA_FILE ($N_TRAIN records)"

if [ "$N_TRAIN" -lt 10 ]; then
  echo "ERROR: only $N_TRAIN training records — too few. ABORTING."
  exit 1
fi

# 3. Train LoRA
SFT_CKPT_DIR="$SFT_BASE/ckpt"
echo "[$(date +%T)] Starting LoRA training → $SFT_CKPT_DIR"

# Kill anything on GPU first
pkill -9 -f vllm 2>/dev/null || true
pkill -9 -f VLLM::EngineCore 2>/dev/null || true
sleep 5

QWEN3_DIR="/workspace/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"
python3 scripts/sft/train_qwen3_lora.py \
  --data "$DATA_FILE" \
  --model "$QWEN3_DIR" \
  --out "$SFT_CKPT_DIR" \
  --epochs 2 \
  --lr 1e-4 \
  --lora-r 32 \
  --max-len 22000 \
  --batch-size 1 \
  --grad-accum 8 \
  2>&1 | tee "$SFT_BASE/train.log"

LORA_DIR="$SFT_CKPT_DIR/final_lora"
if [ ! -d "$LORA_DIR" ]; then
  echo "ERROR: LoRA not saved to $LORA_DIR"
  exit 1
fi

# 4. Bench SFT'd model
echo "[$(date +%T)] Benching SFT LoRA on val_5"
bash scripts/sft/bench_sft_lora.sh "$LORA_DIR" "$SFT_BASE/bench"

echo "=========================================="
echo "[$(date +%T)] SFT v1 ALL DONE."
echo "Data: $DATA_FILE"
echo "LoRA: $LORA_DIR"
echo "Bench summary: $SFT_BASE/bench/summary.txt"
cat "$SFT_BASE/bench/summary.txt" 2>/dev/null || true
echo "=========================================="
