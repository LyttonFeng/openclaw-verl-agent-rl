#!/bin/bash
# Usage: run_agentic_val3.sh <base|lora> [tasks_csv] [runs]
# Starts the transformers shim (optionally with LoRA), runs the isolated Val3
# agentic benchmark against it, then kills the shim.
set -uo pipefail
MODE="${1:-base}"
TASKS="${2:-task_meeting_advisory_stakeholders}"
RUNS="${3:-1}"

[ -f "$HOME/.pinchbench_env" ] && source "$HOME/.pinchbench_env"
: "${DEEPSEEK_API_KEY:?set DEEPSEEK_API_KEY or create ~/.pinchbench_env}"
export PINCHBENCH_GRADE_JUDGE_API_KEY="$DEEPSEEK_API_KEY"

source /root/openclaw-venv/bin/activate
LOG=/tmp/tf_shim_${MODE}.log

pkill -9 -f "python -u /tmp/tf_shim" 2>/dev/null
sleep 1
rm -f "$LOG"
if [ "$MODE" = "lora" ]; then
  export LORA_ADAPTER=/tmp/nma_q35_w2/q35_w2/checkpoint/lora_adapter
else
  unset LORA_ADAPTER 2>/dev/null || true
fi
cd /tmp
PORT=8021 python -u /tmp/tf_shim.py > "$LOG" 2>&1 &
SHIM=$!
echo "[$MODE] shim pid=$SHIM, waiting ready..."
for i in $(seq 1 90); do
  grep -q "shim] ready" "$LOG" 2>/dev/null && { echo "[$MODE] shim READY"; break; }
  grep -qE "Traceback|Error:|Exception" "$LOG" 2>/dev/null && { echo "[$MODE] SHIM CRASH"; tail -15 "$LOG"; kill -9 $SHIM; exit 1; }
  sleep 2
done
grep -q "shim] ready" "$LOG" 2>/dev/null || { echo "[$MODE] SHIM TIMEOUT"; tail -15 "$LOG"; kill -9 $SHIM; exit 1; }

cd /workspace/openclaw-naive-meeting-analysis-github
echo "[$MODE] running bench tasks=$TASKS runs=$RUNS"
TASKS_DIR=/tmp/meeting_analysis_tasks_skilltest \
MODEL=qwen35-4b \
BASE_URL=http://127.0.0.1:8021/v1 \
RUNS="$RUNS" \
VAL3_TASKS="$TASKS" \
OUTPUT_DIR=/tmp/agentic_val3_${MODE} \
KEEP_OPENCLAW_HOME=0 \
PINCHBENCH_MODEL_API_KEY=dummy \
bash scripts/run_val3_bench_isolated.sh
RC=$?
echo "[$MODE] bench rc=$RC"
kill -9 $SHIM 2>/dev/null
echo "AGENTIC_VAL3_DONE_${MODE}"
