#!/bin/bash
# Val3 agentic eval via the transformers shim, flash+ensemble judge, temp=0 (greedy).
# Arg1: mode tag (base|lora). Arg2 (lora only): adapter dir.
# Starts shim (self-redirect, SHIM_DEFAULT_TEMP=0) -> run_val3_bench_isolated.sh.
set -uo pipefail
source ~/.pinchbench_env 2>/dev/null || true
source /root/openclaw-venv/bin/activate
cd /workspace/openclaw-naive-meeting-analysis-github

MODE="${1:-base}"
ADAPTER="${2:-}"
pkill -9 -f "tf_shim|benchmark.py" 2>/dev/null || true
sleep 4
rm -f /tmp/shim_eval_${MODE}.log

LORA_ENV=""
[ "$MODE" = "lora" ] && LORA_ENV="LORA_ADAPTER=$ADAPTER"
env PORT=8021 SHIM_DEFAULT_TEMP=${EVAL_TEMP:-0} SHIM_LOG=/tmp/shim_eval_${MODE}.log $LORA_ENV \
  python -u /tmp/tf_shim.py &
SHIM_PID=$!
for i in $(seq 1 90); do
  grep -q "shim] ready" /tmp/shim_eval_${MODE}.log 2>/dev/null && break
  kill -0 "$SHIM_PID" 2>/dev/null || { echo "SHIM DIED"; tail -15 /tmp/shim_eval_${MODE}.log; exit 1; }
  sleep 3
done
grep -q "shim] ready" /tmp/shim_eval_${MODE}.log 2>/dev/null || { echo "SHIM NOT READY"; tail -15 /tmp/shim_eval_${MODE}.log; exit 1; }
echo "[eval $MODE] shim ready (lora=${ADAPTER:-none})"

MODEL=qwen35-4b \
BASE_URL=http://127.0.0.1:8021/v1 \
RUNS="${RUNS:-3}" \
JUDGE_MODEL=deepseek-v4-flash \
JUDGE_ENSEMBLE="${JUDGE_ENSEMBLE:-3}" \
TASKS_DIR=/tmp/meeting_analysis_tasks_skilltest \
OUTPUT_DIR=/tmp/eval_val3_${MODE} \
PINCHBENCH_MODEL_TEMPERATURE=${EVAL_TEMP:-0} \
bash scripts/run_val3_bench_isolated.sh
RC=$?
pkill -9 -f tf_shim 2>/dev/null || true
echo "EVAL_VAL3_DONE_${MODE} rc=$RC out=/tmp/eval_val3_${MODE}"
