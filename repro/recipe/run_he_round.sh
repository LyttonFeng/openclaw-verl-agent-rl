#!/bin/bash
# run_he_round.sh — COLD-START RL from BASE on H-E instance-specific tasks, with GATED mem.
# Reward = AUTO_W * key_correctness (he_grade vs ground-truth key) + (1-AUTO_W) * committee.
set -uo pipefail
source ~/.pinchbench_env 2>/dev/null || true
source /root/openclaw-venv/bin/activate
unset OPENCLAW_HOST ECS_HOST OPENCLAW_REMOTE_ACTIVATE_CMD
export PINCHBENCH_FORCE_LOCAL_OPENCLAW=1
export ROLLOUT_TIMEOUT_MULT="${ROLLOUT_TIMEOUT_MULT:-4.0}"
ROLLOUT_TEMP="${ROLLOUT_TEMP:-0.7}"
SHIM=/workspace/openclaw-naive-meeting-analysis-github/scripts/tf_agentic/tf_shim_batched.py
NUM_WORKERS="${NUM_WORKERS:-2}"; SHIM_MAX_BATCH="${SHIM_MAX_BATCH:-2}"
REPO=/workspace/openclaw-naive-meeting-analysis-github; cd "$REPO"
RUN_NAME="${RUN_NAME:-he_r1}"
AUTO_W="${AUTO_W:-0.5}"; export DELIBERATE="${DELIBERATE:-1}"
LR="${LR:-2.5e-5}"; export LR
BASE_REF=/workspace/saved_adapters/base_ref_temp03.jsonl
TASKS="${TASKS:-$REPO/data/meeting_analysis_val3_slim_train/train_he24_gated.json}"
HEALTH=/workspace/saved_adapters/he_r1_health.json
RUN=/tmp/nma_round1/$RUN_NAME; ROLLOUT_DIR=$RUN/rollouts
mkdir -p "$RUN/checkpoint" "$ROLLOUT_DIR"; exec >"$RUN/run.log" 2>&1
echo "[he-round] $RUN_NAME FROM=BASE(cold) TASKS=$(basename $TASKS) AUTO_W(key)=$AUTO_W LR=$LR TEMP=$ROLLOUT_TEMP"
pkill -9 -f "tf_shim|benchmark.py" 2>/dev/null || true; sleep 3; rm -f $RUN/shim.log
PORT=8021 SHIM_DEFAULT_TEMP=$ROLLOUT_TEMP SHIM_MAX_BATCH=$SHIM_MAX_BATCH SHIM_LOG=$RUN/shim.log python -u "$SHIM" &
for i in $(seq 1 100); do grep -q "shim] ready" $RUN/shim.log 2>/dev/null && break; sleep 3; done
grep -q "shim] ready" $RUN/shim.log || { echo SHIM_NOT_READY; tail -15 $RUN/shim.log; exit 1; }
grep -i "lora=" $RUN/shim.log | tail -1
echo "[he-round] B0: task select"
ACTIVE=$RUN/active.json; SKIP=$RUN/skip.json
python3 "$REPO/scripts/tf_agentic/select_active_tasks.py" --tasks-file "$TASKS" --out-file "$ACTIVE" --skip-out "$SKIP" --health "$HEALTH" --dead-threshold 2 --reprobe-every 3 || { cp "$TASKS" "$ACTIVE"; echo "[]">"$SKIP"; }
echo "[he-round] B: rollouts K=4 (base+gated-mem)"
PINCHBENCH_JUDGE_ENSEMBLE=1 python -u train/generate_ledger_online_rollouts.py --tasks-file "$ACTIVE" --vllm-base-url http://127.0.0.1:8021/v1 --model qwen35-4b --output-dir "$ROLLOUT_DIR" --n-responses 4 --num-workers $NUM_WORKERS --judge-model deepseek-v4-flash --judge-base-url https://api.deepseek.com/v1 || echo "[warn] rollout rc nonzero"
pkill -9 -f tf_shim 2>/dev/null || true; sleep 5
[ -s "$ROLLOUT_DIR/graded_trajectories.jsonl" ] || { echo NO_ROLLOUTS; exit 1; }
echo "[he-round] B+: KEY-GRADE -> overwrite automated_score with key_score"
python3 /tmp/he_grade.py --graded "$ROLLOUT_DIR/graded_trajectories.jsonl" || { echo KEYGRADE_FAILED; exit 1; }
python3 "$REPO/scripts/tf_agentic/update_task_health.py" --graded "$ROLLOUT_DIR/graded_trajectories.jsonl" --skip-list "$SKIP" --health "$HEALTH" || echo "[warn] health upd"
echo "[he-round] B2: healthcheck"
python3 "$REPO/scripts/tf_agentic/rollout_healthcheck.py" "$ROLLOUT_DIR/graded_trajectories.jsonl" || { echo "ROLLOUT_HEALTH_FAILED"; exit 2; }
echo "[he-round] C: committee inject (AUTO_W=$AUTO_W key + committee blend)"
JUDGE_LIB_DIR=/tmp/judge_lib COMMITTEE_DIR=$REPO/scripts/tf_agentic RULER_DIR=$REPO/scripts/tf_agentic DELIBERATE=$DELIBERATE GRADED_IN=$ROLLOUT_DIR/graded_trajectories.jsonl GRADED_OUT=$RUN/graded_blend.jsonl TASKS_FILE=$TASKS BASE_REF_FILE=$BASE_REF AUTO_W=$AUTO_W python3 -u "$REPO/scripts/tf_agentic/inject_committee_reward.py" || { echo INJECT_FAILED; exit 1; }
echo "[he-round] D: COLD-START train from base (LR=$LR)"
RUN_DIR=$RUN GRADED_NAME=graded_blend.jsonl INIT_LORA= LR=$LR bash "$REPO/scripts/tf_agentic/retrain_committee.sh"
echo "HE_ROUND_DONE ckpt=$([ -e $RUN/checkpoint/lora_adapter/adapter_model.safetensors ] && echo yes || echo NO)"
