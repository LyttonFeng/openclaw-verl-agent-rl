#!/bin/bash
# Eval r1 adapter + captree gated mem on the 3 Val3 tasks (tf_shim temp 0.3, n=4) -> deliverables
# for committee pairwise vs base+captree-mem (captree_mem.jsonl).
set -uo pipefail
source ~/.pinchbench_env 2>/dev/null || true
source /root/openclaw-venv/bin/activate
unset OPENCLAW_HOST ECS_HOST OPENCLAW_REMOTE_ACTIVATE_CMD
export PINCHBENCH_FORCE_LOCAL_OPENCLAW=1 ROLLOUT_TIMEOUT_MULT=6.0
REPO=/workspace/openclaw-naive-meeting-analysis-github; cd "$REPO"
DD=$REPO/data/meeting_analysis_val3_slim_train
R=/tmp/nma_round1/eval_r1_mem; mkdir -p $R/mem; rm -f $R/shim.log $R/mem/graded_trajectories.jsonl
pkill -9 -f "tf_shim|benchmark.py" 2>/dev/null||true; sleep 4
export PORT=8021 SHIM_DEFAULT_TEMP=0.3 SHIM_MAX_BATCH=2 SHIM_LOG=$R/shim.log
export LORA_ADAPTER=/tmp/nma_round1/captree_gated_r1/checkpoint/lora_adapter
python -u $REPO/scripts/tf_agentic/tf_shim_batched.py &
for i in $(seq 1 120); do grep -q "shim] ready" $R/shim.log 2>/dev/null && break; sleep 3; done
grep -q "shim] ready" $R/shim.log || { echo SHIM_NOT_READY; tail -8 $R/shim.log; exit 1; }
grep -i "lora=" $R/shim.log | tail -1
PINCHBENCH_JUDGE_ENSEMBLE=1 python -u train/generate_ledger_online_rollouts.py --tasks-file $DD/val3_3_captree_gated.json --vllm-base-url http://127.0.0.1:8021/v1 --model qwen35-4b --output-dir $R/mem --n-responses 4 --num-workers 2 --judge-model deepseek-v4-flash --judge-base-url https://api.deepseek.com/v1 || echo "[warn] rc=$?"
echo "r1mem rows=$(wc -l < $R/mem/graded_trajectories.jsonl)"
pkill -9 -f tf_shim 2>/dev/null||true
echo EVAL_R1_MEM_DONE
