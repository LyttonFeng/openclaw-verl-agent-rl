#!/bin/bash
# SFT v9: full pipeline + hermes-bug patch already applied + path-normalize +
# anti-overfit training params + 64K bench context.
#
# Run AFTER OC patch (PATCH-B) is in place at
#   /usr/local/lib/.../@mariozechner/pi-ai/dist/providers/openai-completions.js
set -euo pipefail

SFT_BASE="${SFT_BASE:-/workspace/verl_port/sft_v9}"
QWEN3_DIR="/workspace/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c"

mkdir -p "$SFT_BASE"
exec > >(tee -a "$SFT_BASE/runner.log") 2>&1

cd /workspace/openclaw-verl-agent-rl

echo "=========================================="
echo "[$(date +%T)] SFT v9 starting"
echo "  out: $SFT_BASE"
echo "  data: TOP_K=3 MIN_SCORE=0.4 MIN_MSGS=6 MAX_TOOL_CHARS=8000 DROP_ABOVE=32000"
echo "  train: epochs=10 lr=3e-5 dropout=0.2 lora_r=32 max-len=32000"
echo "  bench: vLLM --max-model-len 65536 (YaRN factor=2 = 64K)"
echo "=========================================="

# 1. Pipeline (already has merge_reads + normalize_paths + min_msgs filter)
SFT_DATA_DIR="$SFT_BASE/data"
mkdir -p "$SFT_DATA_DIR"

TOP_K=3 MIN_SCORE=0.4 MIN_MSGS=6 MAX_TOOL_CHARS=8000 DROP_ABOVE=32000 \
  bash scripts/sft/run_pipeline.sh "$SFT_DATA_DIR" \
    /workspace/verl_port/bench/dsv4flash_train_20260521_053546 \
    /workspace/verl_port/bench/dsv4flash_20260521_052408

DATA_FILE="$SFT_DATA_DIR/chatml.jsonl"
N_TRAIN=$(wc -l < "$DATA_FILE")
echo "[$(date +%T)] Final SFT data: $DATA_FILE ($N_TRAIN records)"

# 2. Train
SFT_CKPT_DIR="$SFT_BASE/ckpt"
echo "[$(date +%T)] Starting LoRA v9"

pkill -9 -f vllm 2>/dev/null || true
pkill -9 -f VLLM 2>/dev/null || true
sleep 5

python3 scripts/sft/train_qwen3_lora.py \
  --data "$DATA_FILE" \
  --model "$QWEN3_DIR" \
  --out "$SFT_CKPT_DIR" \
  --epochs 10 \
  --lr 3e-5 \
  --lora-r 32 \
  --lora-dropout 0.2 \
  --max-len 32000 \
  --batch-size 1 \
  --grad-accum 8 \
  2>&1 | tee "$SFT_BASE/train.log"

LORA_DIR="$SFT_CKPT_DIR/final_lora"
if [ ! -d "$LORA_DIR" ]; then
  echo "ERROR: LoRA not saved"
  exit 1
fi

# 3. Bench (uses bench_sft_lora.sh which now uses --max-model-len 65536)
echo "[$(date +%T)] Benching SFT v9 on val_5 × 3"
bash scripts/sft/bench_sft_lora.sh "$LORA_DIR" "$SFT_BASE/bench"

echo "=========================================="
echo "[$(date +%T)] SFT v9 DONE"
cat "$SFT_BASE/bench/summary.txt" 2>/dev/null || true
echo "=========================================="
