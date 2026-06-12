#!/bin/bash
# Fast isolated smoke: start shim (self-redirects stdout to file) + run ROLLOUT
# only on gov+tech (2 tasks, N=1, workers=1). Confirms gov/tech produce real
# non-empty scores through the shim before committing to the full ~3h round.
set -uo pipefail
source ~/.pinchbench_env 2>/dev/null || true
source /root/openclaw-venv/bin/activate
cd /workspace/openclaw-naive-meeting-analysis-github

pkill -9 -f "tf_shim|generate_ledger" 2>/dev/null || true
sleep 3
rm -f /tmp/shim_serial.log
PORT=8021 SHIM_DEFAULT_TEMP=1.0 SHIM_LOG=/tmp/shim_serial.log python -u /tmp/tf_shim.py &
SHIM_PID=$!
for i in $(seq 1 80); do
  grep -q "shim] ready" /tmp/shim_serial.log 2>/dev/null && break
  sleep 3
done
grep -q "shim] ready" /tmp/shim_serial.log 2>/dev/null || { echo "SHIM NOT READY"; tail -10 /tmp/shim_serial.log; exit 1; }
echo "[smoke] shim ready pid=$SHIM_PID"

RUN=/tmp/govtech_smoke_$$
mkdir -p "$RUN/rollouts"
export OPENCLAW_HOME="$RUN/.oc" PINCHBENCH_OPENCLAW_HOME="$RUN/.oc/.openclaw" PINCHBENCH_RUN_ROOT="$RUN/pb"
export PINCHBENCH_FORCE_LOCAL_OPENCLAW=1 PINCHBENCH_SKIP_OPENCLAW_WEB_PREFLIGHT=1 PINCHBENCH_CLEAN_BENCHMARK_WORKSPACE=0
export PINCHBENCH_OPENCLAW_CONTEXT_WINDOW=65536 PINCHBENCH_OPENCLAW_MAX_TOKENS=16384 OPENCLAW_MODEL_REASONING=0 OPENCLAW_HOST=localhost
export MEETING_JUDGE_PROVIDER=deepseek MEETING_JUDGE_MODEL=deepseek-v4-flash MEETING_JUDGE_BASE_URL=https://api.deepseek.com/v1 PINCHBENCH_JUDGE_ENSEMBLE=2
mkdir -p "$PINCHBENCH_OPENCLAW_HOME" "$PINCHBENCH_RUN_ROOT"

echo "[smoke] running gov+tech rollout (N=1, workers=1)"
python -u train/generate_ledger_online_rollouts.py \
  --tasks-file "$PWD/data/meeting_analysis_val3_slim_train/govtech_smoke.json" \
  --vllm-base-url http://127.0.0.1:8021/v1 --model qwen35-4b \
  --output-dir "$RUN/rollouts" --n-responses 1 --num-workers 1 \
  --judge-model deepseek-v4-flash --judge-base-url https://api.deepseek.com/v1 || echo "[smoke] rollout rc=$?"
echo "=== graded ==="
python3 -c "
import json,os
f='$RUN/rollouts/graded_trajectories.jsonl'
rows=[json.loads(l) for l in open(f)] if os.path.exists(f) else []
print('rows:',len(rows))
for r in rows:
    print('  %-34s score=%.3f gate=%s empty=%s'%(r.get('task_id'),r.get('score',-1),r.get('gate_passed'),not (r.get('response') or '').strip()))
"
pkill -9 -f tf_shim 2>/dev/null || true
echo "GOVTECH_SMOKE_DONE"
