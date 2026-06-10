#!/usr/bin/env bash
# Unattended round-by-round RL loop for slim12 -> Val3.
#
# Each round:
#   1. ensure vLLM serves the current rollout model (base, or prev LoRA = flywheel)
#   2. run_naive_ppo_round.sh  (conservative hyperparams + process gate)  -> LoRA
#   3. eval_lora_val3.sh        (tech_action_items behavior smoke -> Val3) -> overall pct
#   4. append result to the loop summary; if Val3 improved AND tech had no 0s,
#      promote this LoRA as the next round's rollout model (flywheel); else keep base.
#
# All progress is appended to $SUMMARY so it can be tailed live.
# Run detached:  setsid bash train/autoloop_val3.sh > autoloop.log 2>&1 < /dev/null &
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"
PYTHON_BIN="${PYTHON_BIN:-/root/openclaw-venv/bin/python}"
BASE_MODEL_PATH="${BASE_MODEL_PATH:-/workspace/qwen_models/qwen3-4b}"
TRAIN_SPLIT="${TRAIN_SPLIT:-$REPO_ROOT/data/train/meeting_analysis_slim12_split.json}"
RUNS_ROOT="${RUNS_ROOT:-/workspace/naive_meeting_analysis_runs}"
N_ROUNDS="${N_ROUNDS:-4}"
START_ROUND="${START_ROUND:-1}"
BASELINE_PCT="${BASELINE_PCT:-50.9}"

# Conservative training defaults (anti base-drift)
export LR="${LR:-2e-6}" KL_BETA="${KL_BETA:-0.05}" REF_KL_BETA="${REF_KL_BETA:-0.05}"
export LORA_RANK="${LORA_RANK:-16}" N_RESPONSES="${N_RESPONSES:-4}"
export PYTHON_BIN MODEL_PATH="$BASE_MODEL_PATH" TRAIN_SPLIT

SUMMARY="${SUMMARY:-$RUNS_ROOT/autoloop_summary_$(date +%Y%m%d_%H%M%S).log}"
mkdir -p "$RUNS_ROOT"
echo "autoloop start  baseline=${BASELINE_PCT}%  rounds=${START_ROUND}..$((START_ROUND+N_ROUNDS-1))" | tee -a "$SUMMARY"

ensure_vllm_base() {
  curl -s -m5 http://127.0.0.1:8021/v1/models 2>/dev/null | grep -q Qwen3-4B-base && return 0
  echo "  [vllm] base not up, starting..." | tee -a "$SUMMARY"
  pkill -9 -f vllm.entrypoints 2>/dev/null || true; sleep 5
  setsid "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
    --model "$BASE_MODEL_PATH" --served-model-name Qwen3-4B-base \
    --host 127.0.0.1 --port 8021 --max-model-len 40960 --gpu-memory-utilization 0.85 \
    --dtype bfloat16 --trust-remote-code --enable-auto-tool-choice \
    --tool-call-parser hermes --reasoning-parser qwen3 \
    > "$RUNS_ROOT/vllm_autoloop.log" 2>&1 < /dev/null &
  for i in $(seq 1 40); do curl -s -m5 http://127.0.0.1:8021/v1/models 2>/dev/null | grep -q Qwen3-4B-base && return 0; sleep 8; done
  return 1
}

ROLLOUT_MODEL="Qwen3-4B-base"   # flywheel updates this to a promoted LoRA name
PREV_LORA_PATH=""

for r in $(seq "$START_ROUND" "$((START_ROUND+N_ROUNDS-1))"); do
  RUN_NAME="autoloop_r${r}_$(date +%Y%m%d_%H%M%S)"
  echo "==== ROUND $r  rollout_model=$ROLLOUT_MODEL  ($RUN_NAME) ====" | tee -a "$SUMMARY"

  ensure_vllm_base || { echo "  ABORT: vLLM base failed to start" | tee -a "$SUMMARY"; break; }

  # 1+2: rollout + select(process gate) + train. Runner kills vLLM before train.
  RUN_NAME="$RUN_NAME" ROLLOUT_MODEL="$ROLLOUT_MODEL" \
    bash train/run_naive_ppo_round.sh > "$RUNS_ROOT/${RUN_NAME}.log" 2>&1
  LORA="$RUNS_ROOT/../openclaw-naive-meeting-analysis-github/results/train/$(ls -t results/train 2>/dev/null | head -1)/checkpoint/lora_adapter"
  LORA="$(ls -dt results/train/*/checkpoint/lora_adapter 2>/dev/null | head -1)"
  if [ -z "$LORA" ] || [ ! -d "$LORA" ]; then
    echo "  ROUND $r: no LoRA produced (check ${RUN_NAME}.log) -> stop" | tee -a "$SUMMARY"
    break
  fi
  echo "  ROUND $r LoRA: $LORA" | tee -a "$SUMMARY"

  # 3: eval (cold vLLM+LoRA -> tech smoke -> Val3). eval_lora_val3 exits 2 if tech still has 0s.
  EVAL_LOG="$RUNS_ROOT/${RUN_NAME}_eval.log"
  SMOKE_RUNS="${SMOKE_RUNS:-5}" VAL3_RUNS="${VAL3_RUNS:-3}" \
    bash train/eval_lora_val3.sh "$LORA" "r${r}-lora" > "$EVAL_LOG" 2>&1
  EVAL_RC=$?
  OVERALL="$(grep -aE 'OVERALL MEETING' "$EVAL_LOG" | grep -oE 'pct=[0-9.]+' | head -1 | cut -d= -f2)"
  TECHLINE="$(grep -a 'tech_action_items' "$EVAL_LOG" | grep -a 'zeros=' | head -1)"

  if [ "$EVAL_RC" = "2" ]; then
    echo "  ROUND $r: tech smoke FAILED (0s present) -> not promoting. $TECHLINE" | tee -a "$SUMMARY"
    # keep rollout model unchanged (do not flywheel a regressing LoRA)
  elif [ -n "$OVERALL" ]; then
    echo "  ROUND $r: Val3 overall=${OVERALL}% (baseline ${BASELINE_PCT}%). $TECHLINE" | tee -a "$SUMMARY"
    impr="$($PYTHON_BIN -c "print(1 if float('$OVERALL') > float('$BASELINE_PCT') else 0)" 2>/dev/null || echo 0)"
    if [ "$impr" = "1" ]; then
      echo "  ROUND $r: improved -> flywheel: next round rolls out r${r}-lora" | tee -a "$SUMMARY"
      ROLLOUT_MODEL="r${r}-lora"; PREV_LORA_PATH="$LORA"
    else
      echo "  ROUND $r: not above baseline -> keep base for next round" | tee -a "$SUMMARY"
    fi
  else
    echo "  ROUND $r: could not parse Val3 overall (see $EVAL_LOG)" | tee -a "$SUMMARY"
  fi
done
echo "==== autoloop done.  summary: $SUMMARY ====" | tee -a "$SUMMARY"
