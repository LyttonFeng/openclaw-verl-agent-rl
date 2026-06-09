#!/usr/bin/env bash
# Minimal naive meeting-analysis PPO-style training round.
#
# Pipeline:
#   1. Generate OpenClaw rollouts against a vLLM OpenAI-compatible endpoint.
#   2. Filter out task groups with no score variance / no effective GRPO signal.
#   3. Recompute rollout-time trainable-token logprobs (P_old).
#   4. Run a PyTorch/PEFT LoRA update with PPO ratio + clip + optional KL.
#
# This script intentionally does not include PRM, swarm policy, veRL, or quality
# filtering beyond the simple dynamic signal filter in select_grpo_samples.py.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"

if [ -f "$HOME/.pinchbench_env" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$HOME/.pinchbench_env"
  set +a
fi

# Training runs use the OpenClaw CLI on this pod. Older env files may still
# carry ECS/remote-host settings; clear them here so a clean GitHub checkout
# does not silently send rollouts to an external OpenClaw server.
unset OPENCLAW_HOST ECS_HOST OPENCLAW_PORT OPENCLAW_USER OPENCLAW_SSH_KEY
export PINCHBENCH_FORCE_LOCAL_OPENCLAW="${PINCHBENCH_FORCE_LOCAL_OPENCLAW:-1}"
export OC_PROVIDER_JS="${OC_PROVIDER_JS:-/usr/lib/node_modules/openclaw/node_modules/@mariozechner/pi-ai/dist/providers/openai-completions.js}"
export OPENCLAW_AGENT_TIMEOUT_SECONDS="${OPENCLAW_AGENT_TIMEOUT_SECONDS:-600}"
export OPENCLAW_LLM_IDLE_TIMEOUT_SECONDS="${OPENCLAW_LLM_IDLE_TIMEOUT_SECONDS:-0}"
export PINCHBENCH_OPENCLAW_CONTEXT_WINDOW="${PINCHBENCH_OPENCLAW_CONTEXT_WINDOW:-65536}"
export PINCHBENCH_OPENCLAW_MAX_TOKENS="${PINCHBENCH_OPENCLAW_MAX_TOKENS:-8192}"
export PINCHBENCH_SKIP_OPENCLAW_WEB_PREFLIGHT="${PINCHBENCH_SKIP_OPENCLAW_WEB_PREFLIGHT:-1}"
export PINCHBENCH_MODEL_TEMPERATURE="${PINCHBENCH_MODEL_TEMPERATURE:-0}"
export PYTHONUNBUFFERED="${PYTHONUNBUFFERED:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

RUN_ID="${RUN_ID:-naive_ppo_$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-$REPO_ROOT/results/train/$RUN_ID}"

OPENCLAW_HOME_ROOT="${OPENCLAW_HOME_ROOT:-$RUN_DIR/runtime/openclaw_home}"
PINCHBENCH_ROOT="${PINCHBENCH_ROOT:-$RUN_DIR/runtime/pinchbench}"
export OPENCLAW_HOME="${OPENCLAW_HOME:-$OPENCLAW_HOME_ROOT/$RUN_ID}"
export PINCHBENCH_OPENCLAW_HOME="${PINCHBENCH_OPENCLAW_HOME:-$OPENCLAW_HOME/.openclaw}"
export PINCHBENCH_RUN_ROOT="${PINCHBENCH_RUN_ROOT:-$PINCHBENCH_ROOT/$RUN_ID}"

if [ -z "${PINCHBENCH_AGENT_SUFFIX:-}" ]; then
  RUN_ID_SHORT="$(printf '%s' "$RUN_ID" | tr -cd '[:alnum:]_-' | cut -c1-12)"
  RUN_ID_CKSUM="$(printf '%s' "$RUN_ID" | cksum | awk '{print $1}')"
  export PINCHBENCH_AGENT_SUFFIX="tr_${RUN_ID_SHORT}_${RUN_ID_CKSUM}"
else
  export PINCHBENCH_AGENT_SUFFIX
fi
if [ "${#PINCHBENCH_AGENT_SUFFIX}" -gt 32 ]; then
  SUFFIX_SHORT="$(printf '%s' "$PINCHBENCH_AGENT_SUFFIX" | tr -cd '[:alnum:]_-' | cut -c1-16)"
  SUFFIX_CKSUM="$(printf '%s' "$PINCHBENCH_AGENT_SUFFIX" | cksum | awk '{print $1}')"
  export PINCHBENCH_AGENT_SUFFIX="${SUFFIX_SHORT}_${SUFFIX_CKSUM}"
fi

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
ROLLOUT_MODEL="${ROLLOUT_MODEL:-Qwen3-4B}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8021/v1}"
PREV_LORA="${PREV_LORA:-}"

TRAIN_SPLIT="${TRAIN_SPLIT:-$REPO_ROOT/data/train/meeting_analysis_all_samples_split.json}"
TASKS_DIR="${TASKS_DIR:-$REPO_ROOT/data/train/tasks}"
ASSETS_DIR="${ASSETS_DIR:-$REPO_ROOT/data/eval/assets}"

N_RESPONSES="${N_RESPONSES:-4}"
NUM_WORKERS="${NUM_WORKERS:-1}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v4-pro}"

MAX_SEQ_LENGTH="${MAX_SEQ_LENGTH:-40960}"
ROPE_SCALING_FACTOR="${ROPE_SCALING_FACTOR:-}"
LR="${LR:-5e-6}"
LORA_RANK="${LORA_RANK:-16}"
GRAD_ACCUM_STEPS="${GRAD_ACCUM_STEPS:-2}"
CLIP_EPS="${CLIP_EPS:-0.2}"
KL_BETA="${KL_BETA:-0.02}"
REF_LOGPROBS_FILE="${REF_LOGPROBS_FILE:-}"
REF_KL_BETA="${REF_KL_BETA:-0.02}"
VARIANCE_THRESHOLD="${VARIANCE_THRESHOLD:-1e-8}"
STOP_VLLM_BEFORE_TRAIN="${STOP_VLLM_BEFORE_TRAIN:-1}"

stop_vllm_for_training() {
  if [ "$STOP_VLLM_BEFORE_TRAIN" != "1" ]; then
    return 0
  fi
  echo
  echo "[gpu] stopping vLLM before logprob/train to free single-GPU memory"
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  sleep 10
}

mkdir -p "$RUN_DIR/rollouts" "$RUN_DIR/selection" "$RUN_DIR/checkpoint" \
  "$PINCHBENCH_OPENCLAW_HOME" "$PINCHBENCH_RUN_ROOT"

echo "== naive meeting-analysis PPO round =="
echo "run_dir:       $RUN_DIR"
echo "model_path:    $MODEL_PATH"
echo "rollout_model: $ROLLOUT_MODEL"
echo "vllm_base_url: $VLLM_BASE_URL"
echo "prev_lora:     ${PREV_LORA:-none}"
echo "train_split:   $TRAIN_SPLIT"
echo "tasks_dir:     $TASKS_DIR"
echo "n_responses:   $N_RESPONSES"
echo "judge_model:   $JUDGE_MODEL"
echo "python_bin:    $PYTHON_BIN"
echo "local_claw:    $PINCHBENCH_FORCE_LOCAL_OPENCLAW"
echo "openclaw_home: $OPENCLAW_HOME"
echo "run_root:      $PINCHBENCH_RUN_ROOT"
echo "agent_suffix:  $PINCHBENCH_AGENT_SUFFIX"

echo
echo "[1/4] rollout sampling"
"$PYTHON_BIN" train/generate_meeting_rollouts.py \
  --split-file "$TRAIN_SPLIT" \
  --split train \
  --tasks-dir "$TASKS_DIR" \
  --assets-dir "$ASSETS_DIR" \
  --vllm-base-url "$VLLM_BASE_URL" \
  --model "$ROLLOUT_MODEL" \
  --n-responses "$N_RESPONSES" \
  --output-dir "$RUN_DIR/rollouts" \
  --judge-model "$JUDGE_MODEL" \
  --timeout "$TIMEOUT_SECONDS" \
  --num-workers "$NUM_WORKERS"

GRADED_FILE="$RUN_DIR/rollouts/graded_trajectories.jsonl"

echo
echo "[2/4] dynamic training-signal filter"
"$PYTHON_BIN" train/select_grpo_samples.py \
  --graded-file "$GRADED_FILE" \
  --output-dir "$RUN_DIR/selection" \
  --variance-threshold "$VARIANCE_THRESHOLD" \
  --drop-bad-trajectories

FILTERED_FILE="$RUN_DIR/selection/graded_trajectories_prm_valid.jsonl"

stop_vllm_for_training

echo
echo "[3/4] recompute P_old logprobs"
LOGPROBS_FILE="$RUN_DIR/rollout_logprobs.jsonl"
LOGPROB_ARGS=(
  --graded-file "$FILTERED_FILE"
  --model-path "$MODEL_PATH"
  --output "$LOGPROBS_FILE"
  --max-seq-length "$MAX_SEQ_LENGTH"
)
if [ -n "$ROPE_SCALING_FACTOR" ]; then
  LOGPROB_ARGS+=(--rope-scaling-factor "$ROPE_SCALING_FACTOR")
fi
if [ -n "$PREV_LORA" ]; then
  LOGPROB_ARGS+=(--lora-path "$PREV_LORA")
fi
"$PYTHON_BIN" train/compute_rollout_logprobs.py "${LOGPROB_ARGS[@]}"

echo
echo "[4/4] PPO-style LoRA update"
TRAIN_ARGS=(
  --graded-file "$FILTERED_FILE"
  --model-path "$MODEL_PATH"
  --output-dir "$RUN_DIR/checkpoint"
  --lr "$LR"
  --lora-rank "$LORA_RANK"
  --grad-accum-steps "$GRAD_ACCUM_STEPS"
  --max-seq-length "$MAX_SEQ_LENGTH"
  --logprobs-file "$LOGPROBS_FILE"
  --clip-eps "$CLIP_EPS"
  --kl-beta "$KL_BETA"
)
if [ -n "$ROPE_SCALING_FACTOR" ]; then
  TRAIN_ARGS+=(--rope-scaling-factor "$ROPE_SCALING_FACTOR")
fi
if [ -n "$REF_LOGPROBS_FILE" ]; then
  TRAIN_ARGS+=(--ref-logprobs-file "$REF_LOGPROBS_FILE" --ref-kl-beta "$REF_KL_BETA")
fi
if [ -n "$PREV_LORA" ]; then
  TRAIN_ARGS+=(--lora-path "$PREV_LORA")
fi
"$PYTHON_BIN" train/train_meeting_grpo_step.py "${TRAIN_ARGS[@]}"

echo
echo "DONE"
echo "LoRA adapter: $RUN_DIR/checkpoint/lora_adapter"
