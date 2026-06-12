#!/bin/bash
# Self-contained one-round driver: start the transformers shim (self-redirects its
# stdout to a file so an ssh drop can't BrokenPipe it), wait until ready, then run
# the online RL round against it. run_ledger14 pkills tf_shim before the train step
# to free the GPU. Run this ON the pod (survives ssh drops as one process tree).
set -uo pipefail
source ~/.pinchbench_env 2>/dev/null || true
source /root/openclaw-venv/bin/activate
cd /workspace/openclaw-naive-meeting-analysis-github

RUN_NAME="${RUN_NAME:-val3plus6_w1d}"
echo "[round] killing stale shim/round..."
pkill -9 -f "tf_shim|generate_ledger_online" 2>/dev/null || true
sleep 3
rm -f /tmp/shim_serial.log

echo "[round] starting shim..."
PORT=8021 SHIM_DEFAULT_TEMP=1.0 SHIM_LOG=/tmp/shim_serial.log python -u /tmp/tf_shim.py &
SHIM_PID=$!
for i in $(seq 1 80); do
  grep -q "shim] ready" /tmp/shim_serial.log 2>/dev/null && break
  if ! kill -0 "$SHIM_PID" 2>/dev/null; then echo "[round] SHIM DIED at startup"; tail -20 /tmp/shim_serial.log; exit 1; fi
  sleep 3
done
grep -q "shim] ready" /tmp/shim_serial.log 2>/dev/null || { echo "[round] SHIM not ready"; tail -20 /tmp/shim_serial.log; exit 1; }
echo "[round] shim ready (pid $SHIM_PID)"

echo "[round] launching online RL round ($RUN_NAME)..."
MODEL_PATH=/tmp/qwen3.5-4b \
SERVED_MODEL=qwen35-4b \
VLLM_BASE_URL=http://127.0.0.1:8021/v1 \
TASKS_FILE=$PWD/data/meeting_analysis_val3_slim_train/val3_plus6_train.json \
NUM_WORKERS=1 N_RESPONSES=4 LORA_RANK=32 LR=5e-5 \
BASE_DIR=/tmp/nma_round1 RUN_NAME=$RUN_NAME \
bash scripts/run_ledger14_online_rl.sh
RC=$?
pkill -9 -f tf_shim 2>/dev/null || true
echo "ROUND_COMPLETE rc=$RC run=/tmp/nma_round1/$RUN_NAME"
