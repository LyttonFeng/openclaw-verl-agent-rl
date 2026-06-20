#!/bin/bash
# RESUME: r2 + r3 on-policy continue from the saved r1 ckpt. Fixes the LORA_ADAPTER env bug
# (now a proper export, not a literal-command expansion). r1 already trained.
set -uo pipefail
source ~/.pinchbench_env 2>/dev/null || true
source /root/openclaw-venv/bin/activate
unset OPENCLAW_HOST ECS_HOST OPENCLAW_REMOTE_ACTIVATE_CMD
export PINCHBENCH_FORCE_LOCAL_OPENCLAW=1 ROLLOUT_TIMEOUT_MULT=6.0
REPO=/workspace/openclaw-naive-meeting-analysis-github; cd "$REPO"
SHIM=$REPO/scripts/tf_agentic/tf_shim_batched.py
TASKS=$REPO/data/meeting_analysis_val3_slim_train/val3_plus6_captree_gated_6grp.json
BASE_REF=/workspace/saved_adapters/base_ref_temp03.jsonl
HEALTH=/tmp/nma_round1/captree_health.json
ROLLOUT_TEMP=0.7; export DELIBERATE=1

do_round(){  # $1=run_name $2=init_lora $3=lr
  local RN=$1 INIT=$2 LR=$3
  local RUN=/tmp/nma_round1/$RN
  local ROLL=$RUN/rollouts
  mkdir -p "$RUN/checkpoint" "$ROLL"; rm -f "$RUN/shim.log" "$ROLL/graded_trajectories.jsonl"
  echo "[$RN] FROM=$INIT LR=$LR"
  [ -e "$INIT/adapter_model.safetensors" ] || { echo "[$RN] INIT_ADAPTER_MISSING:$INIT"; return 1; }
  pkill -9 -f "tf_shim|benchmark.py" 2>/dev/null||true; sleep 4
  export PORT=8021 SHIM_DEFAULT_TEMP=$ROLLOUT_TEMP SHIM_MAX_BATCH=2 SHIM_LOG=$RUN/shim.log
  export LORA_ADAPTER="$INIT"          # <-- proper env export (the bug fix)
  python -u "$SHIM" &
  for i in $(seq 1 120); do grep -q "shim] ready" "$RUN/shim.log" 2>/dev/null && break; sleep 3; done
  grep -q "shim] ready" "$RUN/shim.log" || { echo "[$RN] SHIM_NOT_READY"; tail -8 "$RUN/shim.log" 2>/dev/null; return 1; }
  grep -i "lora=" "$RUN/shim.log" | tail -1
  local ACTIVE=$RUN/active.json SKIP=$RUN/skip.json
  python3 "$REPO/scripts/tf_agentic/select_active_tasks.py" --tasks-file "$TASKS" --out-file "$ACTIVE" --skip-out "$SKIP" --health "$HEALTH" --dead-threshold 2 --reprobe-every 3 || { cp "$TASKS" "$ACTIVE"; echo "[]">"$SKIP"; }
  PINCHBENCH_JUDGE_ENSEMBLE=1 python -u train/generate_ledger_online_rollouts.py --tasks-file "$ACTIVE" --vllm-base-url http://127.0.0.1:8021/v1 --model qwen35-4b --output-dir "$ROLL" --n-responses 4 --num-workers 2 --judge-model deepseek-v4-flash --judge-base-url https://api.deepseek.com/v1 || echo "[$RN] rollout rc nonzero"
  pkill -9 -f tf_shim 2>/dev/null||true; sleep 5
  [ -s "$ROLL/graded_trajectories.jsonl" ] || { echo "[$RN] NO_ROLLOUTS"; return 1; }
  echo "[$RN] rollouts=$(wc -l < $ROLL/graded_trajectories.jsonl) rows"
  python3 "$REPO/scripts/tf_agentic/update_task_health.py" --graded "$ROLL/graded_trajectories.jsonl" --skip-list "$SKIP" --health "$HEALTH" || true
  python3 "$REPO/scripts/tf_agentic/rollout_healthcheck.py" "$ROLL/graded_trajectories.jsonl" || { echo "[$RN] ROLLOUT_HEALTH_FAILED"; return 2; }
  JUDGE_LIB_DIR=/tmp/judge_lib COMMITTEE_DIR=$REPO/scripts/tf_agentic RULER_DIR=$REPO/scripts/tf_agentic DELIBERATE=1 GRADED_IN=$ROLL/graded_trajectories.jsonl GRADED_OUT=$RUN/graded_blend.jsonl TASKS_FILE=$TASKS BASE_REF_FILE=$BASE_REF AUTO_W=0.0 python3 -u "$REPO/scripts/tf_agentic/inject_committee_reward.py" || { echo "[$RN] INJECT_FAILED"; return 1; }
  RUN_DIR=$RUN GRADED_NAME=graded_blend.jsonl INIT_LORA=$INIT LR=$LR bash "$REPO/scripts/tf_agentic/retrain_committee.sh"
  unset LORA_ADAPTER
  [ -e "$RUN/checkpoint/lora_adapter/adapter_model.safetensors" ] && echo "[$RN] CKPT_OK" || { echo "[$RN] CKPT_MISSING"; return 1; }
}

do_round captree_gated_r2 /tmp/nma_round1/captree_gated_r1/checkpoint/lora_adapter 2.0e-5 || { echo CHAIN_FAILED_R2; exit 1; }
do_round captree_gated_r3 /tmp/nma_round1/captree_gated_r2/checkpoint/lora_adapter 2.0e-5 || { echo CHAIN_FAILED_R3; exit 1; }
echo "CAPTREE_R2R3_DONE"
