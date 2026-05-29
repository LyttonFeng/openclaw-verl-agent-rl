#!/bin/bash
# SFT v2: more samples (TOP_K=3 MIN_SCORE=0.4), longer context (DROP_ABOVE=32000),
# more training (15 epoch + lr 5e-5). val_5 still in train for fit-ceiling test.
set -euo pipefail

SFT_BASE="${SFT_BASE:-/workspace/verl_port/sft_v2}"
QWEN3_DIR="/workspace/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"

mkdir -p "$SFT_BASE"
LOG="$SFT_BASE/runner.log"
exec > >(tee -a "$LOG") 2>&1

cd /workspace/openclaw-verl-agent-rl

echo "=========================================="
echo "[$(date +%T)] SFT v2 starting"
echo "  out base: $SFT_BASE"
echo "  config:   TOP_K=3 MIN_SCORE=0.4 DROP_ABOVE=32000 MAX_TOOL_CHARS=8000"
echo "            epochs=15 lr=5e-5 max-len=32000"
echo "=========================================="

# 1. Data pipeline (re-use already-collected bench data)
SFT_DATA_DIR="$SFT_BASE/data"
mkdir -p "$SFT_DATA_DIR"

TOP_K=3 MIN_SCORE=0.4 MAX_TOOL_CHARS=8000 DROP_ABOVE=32000 \
  bash scripts/sft/run_pipeline.sh "$SFT_DATA_DIR" \
    /workspace/verl_port/bench/dsv4flash_train_20260521_053546 \
    /workspace/verl_port/bench/dsv4flash_20260521_052408

DATA_FILE="$SFT_DATA_DIR/chatml.jsonl"
N_TRAIN=$(wc -l < "$DATA_FILE")
echo "[$(date +%T)] Final training data: $DATA_FILE ($N_TRAIN records)"

# 2. Train LoRA
SFT_CKPT_DIR="$SFT_BASE/ckpt"
echo "[$(date +%T)] Starting LoRA v2 → $SFT_CKPT_DIR"

pkill -9 -f vllm 2>/dev/null || true
pkill -9 -f VLLM 2>/dev/null || true
sleep 5

python3 scripts/sft/train_qwen3_lora.py \
  --data "$DATA_FILE" \
  --model "$QWEN3_DIR" \
  --out "$SFT_CKPT_DIR" \
  --epochs 15 \
  --lr 5e-5 \
  --lora-r 32 \
  --max-len 32000 \
  --batch-size 1 \
  --grad-accum 8 \
  2>&1 | tee "$SFT_BASE/train.log"

LORA_DIR="$SFT_CKPT_DIR/final_lora"
if [ ! -d "$LORA_DIR" ]; then
  echo "ERROR: LoRA not saved to $LORA_DIR"
  exit 1
fi

# 3. Bench
echo "[$(date +%T)] Benching SFT v2 LoRA on val_5"
bash scripts/sft/bench_sft_lora.sh "$LORA_DIR" "$SFT_BASE/bench"

echo "=========================================="
echo "[$(date +%T)] SFT v2 ALL DONE."
echo "Data:  $DATA_FILE"
echo "LoRA:  $LORA_DIR"
echo "Bench: $SFT_BASE/bench/summary.txt"
cat "$SFT_BASE/bench/summary.txt" 2>/dev/null || true
echo "=========================================="
