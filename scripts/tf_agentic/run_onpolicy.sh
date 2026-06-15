#!/bin/bash
# On-policy round (committee_w2), fully pod-resident (survives mac sleep / ssh drop).
#  A) shim serves committee_blend @ temp=1.0  -> diverse on-policy rollouts
#  B) generate_ledger_online_rollouts: K=4 per task (val3_plus6) + flash grade (for automated_score)
#  C) committee re-score: llm_rubric + base-model reference anchor (放法B) + blend 0.5/0.5
#  D) continue-train FROM committee_blend (on-policy; logprobs also use that adapter)
set -uo pipefail
source ~/.pinchbench_env 2>/dev/null || true
source /root/openclaw-venv/bin/activate
# Use the pod's LOCAL OpenClaw. ~/.pinchbench_env sets OPENCLAW_HOST/ECS_HOST which would route
# to a remote ECS box (needs ssh keys we don't have) and break workspace sync. Unset + force local.
unset OPENCLAW_HOST ECS_HOST OPENCLAW_REMOTE_ACTIVATE_CMD
export PINCHBENCH_FORCE_LOCAL_OPENCLAW=1
# advisory has a 71K-char transcript; default 180×2=360s timed out all 4 rollouts last round
# (no-write + written-but-auto=0). Give generous headroom: 180×4=720s.
export ROLLOUT_TIMEOUT_MULT=4.0
REPO=/workspace/openclaw-naive-meeting-analysis-github
cd "$REPO"
RUN=/tmp/nma_round1/onpolicy_w2
ROLLOUT_DIR=$RUN/rollouts
ADAPTER=/tmp/nma_round1/committee_blend_w1/checkpoint/lora_adapter
TASKS=$REPO/data/meeting_analysis_val3_slim_train/val3_plus6_train.json
BASE_REF=/tmp/nma_round1/val3plus6_w1e/rollouts/graded_trajectories.jsonl
mkdir -p "$RUN/checkpoint" "$ROLLOUT_DIR"

echo "[onpolicy] A: start shim serving committee_blend @ temp=1.0"
pkill -9 -f "tf_shim|benchmark.py" 2>/dev/null || true
sleep 3; rm -f "$RUN/shim.log"
PORT=8021 SHIM_DEFAULT_TEMP=1.0 SHIM_LOG=$RUN/shim.log LORA_ADAPTER=$ADAPTER python -u /tmp/tf_shim.py &
SHIM=$!
for i in $(seq 1 90); do
  grep -q "shim] ready" "$RUN/shim.log" 2>/dev/null && break
  kill -0 "$SHIM" 2>/dev/null || { echo "SHIM_DIED"; tail -15 "$RUN/shim.log"; exit 1; }
  sleep 3
done
grep -q "shim] ready" "$RUN/shim.log" 2>/dev/null || { echo "SHIM_NOT_READY"; tail -15 "$RUN/shim.log"; exit 1; }
grep "lora=" "$RUN/shim.log" | tail -1

echo "[onpolicy] B: rollouts K=4 (val3_plus6) + flash grade"
PINCHBENCH_JUDGE_ENSEMBLE=1 python -u train/generate_ledger_online_rollouts.py \
  --tasks-file "$TASKS" --vllm-base-url http://127.0.0.1:8021/v1 --model qwen35-4b \
  --output-dir "$ROLLOUT_DIR" --n-responses 4 --num-workers 1 \
  --judge-model deepseek-v4-flash --judge-base-url https://api.deepseek.com/v1 \
  || echo "[warn] rollout driver exited non-zero (validating file instead)"
pkill -9 -f tf_shim 2>/dev/null || true; sleep 5
[ -s "$ROLLOUT_DIR/graded_trajectories.jsonl" ] || { echo "NO_ROLLOUTS"; exit 1; }
echo "[onpolicy] rollouts: $(wc -l < $ROLLOUT_DIR/graded_trajectories.jsonl) rows"

echo "[onpolicy] B2: rollout health check (catch timeout/no-write/false-auto=0 BEFORE training)"
python3 "$REPO/scripts/tf_agentic/rollout_healthcheck.py" "$ROLLOUT_DIR/graded_trajectories.jsonl" \
  || { echo "ROLLOUT_HEALTH_FAILED — stopping before train (fix harness, do not train on garbage)"; exit 2; }

echo "[onpolicy] C: committee re-score (rubric + base-ref) + blend"
JUDGE_LIB_DIR=/tmp/judge_lib COMMITTEE_DIR=$REPO/scripts/tf_agentic RULER_DIR=$REPO/scripts/tf_agentic \
GRADED_IN=$ROLLOUT_DIR/graded_trajectories.jsonl GRADED_OUT=$RUN/graded_blend.jsonl \
TASKS_FILE=$TASKS BASE_REF_FILE=$BASE_REF AUTO_W=0.5 \
python3 -u "$REPO/scripts/tf_agentic/inject_committee_reward.py" || { echo "INJECT_FAILED"; exit 1; }

echo "[onpolicy] D: continue-train FROM committee_blend"
RUN_DIR=$RUN GRADED_NAME=graded_blend.jsonl INIT_LORA=$ADAPTER \
bash "$REPO/scripts/tf_agentic/retrain_committee.sh"
echo "ONPOLICY_W2_DONE ckpt=$RUN/checkpoint/lora_adapter"
