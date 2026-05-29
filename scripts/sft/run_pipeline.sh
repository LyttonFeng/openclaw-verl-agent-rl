#!/bin/bash
# End-to-end SFT data pipeline:
#   1. Extract OC transcripts → raw JSONL
#   2. Filter top-K by score per task
#   3. Convert to ChatML
#   4. Validate
#
# Usage:
#   bash run_pipeline.sh <out_dir> <bench_dir1> [<bench_dir2> ...]
set -euo pipefail

OUT_DIR="${1:?Usage: run_pipeline.sh <out_dir> <bench_dir1> [<bench_dir2> ...]}"
shift
BENCH_DIRS=("$@")
if [ ${#BENCH_DIRS[@]} -eq 0 ]; then
  echo "ERROR: need at least one bench dir"
  exit 1
fi

mkdir -p "$OUT_DIR"
SCRIPTS=$(dirname "$(realpath "$0")")

RAW="$OUT_DIR/raw.jsonl"
RAW_MERGED="$OUT_DIR/raw_merged.jsonl"
FILTERED="$OUT_DIR/filtered.jsonl"
CHATML_RAW="$OUT_DIR/chatml_raw.jsonl"
CHATML="$OUT_DIR/chatml.jsonl"

echo "[1a/5] extract → $RAW"
python3 "$SCRIPTS/1_extract_oc_transcript.py" \
  --bench-dirs "${BENCH_DIRS[@]}" \
  --output "$RAW"

echo
echo "[1b/5] merge consecutive same-file reads → $RAW_MERGED"
python3 "$SCRIPTS/1b_merge_reads.py" \
  --input "$RAW" \
  --output "$RAW_MERGED"

NORM="$OUT_DIR/raw_normalized.jsonl"
echo
echo "[1c/5] normalize /tmp/pinchbench/<NNNN>/ paths → $NORM"
python3 "$SCRIPTS/1c_normalize_paths.py" \
  --input "$RAW_MERGED" \
  --output "$NORM"

echo
echo "[2/5] filter top-K per task → $FILTERED"
python3 "$SCRIPTS/2_filter_quality.py" \
  --input "$NORM" \
  --output "$FILTERED" \
  --top-k "${TOP_K:-2}" \
  --min-score "${MIN_SCORE:-0.5}" \
  --min-msgs "${MIN_MSGS:-0}"

echo
echo "[3a/5] convert to ChatML → $CHATML_RAW"
python3 "$SCRIPTS/3_to_chatml.py" \
  --input "$FILTERED" \
  --output "$CHATML_RAW"

echo
echo "[3b/5] truncate oversized tool results → $CHATML"
python3 "$SCRIPTS/3b_truncate_tools.py" \
  --input "$CHATML_RAW" \
  --output "$CHATML" \
  --max-tool-chars "${MAX_TOOL_CHARS:-8000}" \
  --drop-above-tokens "${DROP_ABOVE:-24000}"

echo
echo "[4/5] validate $CHATML"
TOKENIZER_FLAG=""
if [ -n "${TOKENIZER_DIR:-}" ]; then
  TOKENIZER_FLAG="--tokenizer $TOKENIZER_DIR"
fi
python3 "$SCRIPTS/4_validate.py" \
  --input "$CHATML" \
  --max-tokens "${MAX_TOKENS:-32000}" \
  $TOKENIZER_FLAG

echo
echo "DONE — final SFT data: $CHATML"
