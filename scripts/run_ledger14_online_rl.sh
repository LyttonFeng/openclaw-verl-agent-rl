#!/usr/bin/env bash
# Online GRPO smoke run for the 14 claw-data-agent evidence tasks.
#
# Pipeline:
#   1. Roll out each online RL task with OpenClaw/Qwen3-4B.
#   2. Grade with the embedded task grade_function and reward_contract.
#      The process gate is applied inside generate_ledger_online_rollouts.py:
#        - missing expected output file -> score = 0
#        - quote_verified below gate    -> score = 0
#        - table_format below gate       -> score = 0
#   3. Recompute rollout-time logprobs with the old policy.
#   4. Train one LoRA GRPO/PPO step.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

# Task workspace_files use source "meetings/<file>", resolved by the rollout
# driver against assets/meetings/. On the naive branch the transcripts live at
# data/eval/assets/meetings/, so link them in (idempotent) to avoid every
# rollout failing with "Workspace file not found".
if [ ! -e assets/meetings ] && [ -d data/eval/assets/meetings ]; then
  mkdir -p assets && ln -sfn "$REPO_ROOT/data/eval/assets/meetings" assets/meetings
fi

if [ -f "$HOME/.pinchbench_env" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$HOME/.pinchbench_env"
  set +a
  echo "[env] sourced $HOME/.pinchbench_env"
fi

VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8021/v1}"
SERVED_MODEL="${SERVED_MODEL:-Qwen3-4B-base}"
MODEL_PATH="${MODEL_PATH:-/workspace/qwen_models/qwen3-4b}"
# Default to the combined 26-task registry (14 ledger + 12 slim).
TASKS_FILE="${TASKS_FILE:-$REPO_ROOT/data/meeting_analysis_val3_slim_train/canonical_26_tasks.json}"
# Prefer project venv (has yaml/torch) if present; else python3.
if [ -z "${PYTHON_BIN:-}" ]; then
  if [ -x /root/openclaw-venv/bin/python ]; then PYTHON_BIN=/root/openclaw-venv/bin/python; else PYTHON_BIN=python3; fi
fi
if [ -d /workspace/qwen_models/qwen3-4b ] && [ "${MODEL_PATH:-}" = "" ]; then MODEL_PATH=/workspace/qwen_models/qwen3-4b; fi

N_RESPONSES="${N_RESPONSES:-4}"
NUM_WORKERS="${NUM_WORKERS:-4}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-chat}"
JUDGE_BASE_URL="${JUDGE_BASE_URL:-https://api.deepseek.com/v1}"
AUTO_ONLY="${AUTO_ONLY:-0}"

LR="${LR:-1e-6}"
LORA_RANK="${LORA_RANK:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-32768}"
CLIP_EPS="${CLIP_EPS:-0.2}"
KL_BETA="${KL_BETA:-0.05}"
PRM_ALPHA="${PRM_ALPHA:-1.0}"
PRM_BETA="${PRM_BETA:-0.0}"
PRM_MODE="${PRM_MODE:-additive}"

# Single-GPU pod: train on GPU0 after killing vLLM to free it (see STOP_VLLM_BEFORE_TRAIN).
TRAIN_CUDA_VISIBLE_DEVICES="${TRAIN_CUDA_VISIBLE_DEVICES:-0}"
STOP_VLLM_BEFORE_TRAIN="${STOP_VLLM_BEFORE_TRAIN:-1}"

RUN_NAME="${RUN_NAME:-ledger14_online_rl_$(date +%Y%m%d_%H%M%S)}"
BASE_DIR="${BASE_DIR:-/workspace/naive_meeting_analysis_runs}"
RUN_DIR="$BASE_DIR/$RUN_NAME"
ROLLOUT_DIR="$RUN_DIR/rollouts"
LOGPROBS_FILE="$RUN_DIR/rollout_logprobs.jsonl"
CKPT_DIR="$RUN_DIR/checkpoint"

OPENCLAW_HOME_ROOT="${OPENCLAW_HOME_ROOT:-/tmp/openclaw_home}"
PINCHBENCH_ROOT="${PINCHBENCH_ROOT:-/tmp/pinchbench}"
export OPENCLAW_HOME="${OPENCLAW_HOME:-$OPENCLAW_HOME_ROOT/$RUN_NAME}"
export PINCHBENCH_OPENCLAW_HOME="${PINCHBENCH_OPENCLAW_HOME:-$OPENCLAW_HOME/.openclaw}"
export PINCHBENCH_RUN_ROOT="${PINCHBENCH_RUN_ROOT:-$PINCHBENCH_ROOT/$RUN_NAME}"

mkdir -p "$ROLLOUT_DIR" "$CKPT_DIR"
LOG_FILE="$RUN_DIR/run.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "════════════════════════════════════════════════════════════════"
echo "  Ledger14 Online RL"
echo "════════════════════════════════════════════════════════════════"
echo "  Run dir:      $RUN_DIR"
echo "  vLLM URL:     $VLLM_BASE_URL"
echo "  Served model: $SERVED_MODEL"
echo "  Model path:   $MODEL_PATH"
echo "  Tasks file:   $TASKS_FILE"
echo "  Rollouts:     $N_RESPONSES per task, workers=$NUM_WORKERS"
echo "  Judge:        $JUDGE_MODEL @ $JUDGE_BASE_URL (AUTO_ONLY=$AUTO_ONLY)"
echo "  Train GPU(s): $TRAIN_CUDA_VISIBLE_DEVICES"
echo "  OpenClaw home:$OPENCLAW_HOME"
echo "  Run root:     $PINCHBENCH_RUN_ROOT"
echo "════════════════════════════════════════════════════════════════"

echo "[preflight] vLLM models"
curl -sSf "$VLLM_BASE_URL/models" | python3 -m json.tool

if [ "$AUTO_ONLY" != "1" ] && [ -z "${DEEPSEEK_API_KEY:-}" ]; then
  echo "ERROR: DEEPSEEK_API_KEY is required unless AUTO_ONLY=1"
  exit 1
fi

export PINCHBENCH_FORCE_LOCAL_OPENCLAW=1
export PINCHBENCH_ALLOW_LOCAL_OPENCLAW=1
export PINCHBENCH_SKIP_OPENCLAW_WEB_PREFLIGHT=1
# OpenClaw 2026.6 attests a workspace when the agent is created. The online
# rollout driver creates a fresh per-run workspace, so deleting it again inside
# prepare_task_workspace would trigger WorkspaceVanishedError.
export PINCHBENCH_CLEAN_BENCHMARK_WORKSPACE="${PINCHBENCH_CLEAN_BENCHMARK_WORKSPACE:-0}"
export PINCHBENCH_OPENCLAW_CONTEXT_WINDOW="${PINCHBENCH_OPENCLAW_CONTEXT_WINDOW:-65536}"
export PINCHBENCH_OPENCLAW_MAX_TOKENS="${PINCHBENCH_OPENCLAW_MAX_TOKENS:-8192}"
export OPENCLAW_MODEL_REASONING="${OPENCLAW_MODEL_REASONING:-0}"
export OPENCLAW_HOST=localhost
export MEETING_JUDGE_PROVIDER="${MEETING_JUDGE_PROVIDER:-deepseek}"
mkdir -p "$PINCHBENCH_OPENCLAW_HOME" "$PINCHBENCH_RUN_ROOT"

AUTO_ONLY_ARG=()
if [ "$AUTO_ONLY" = "1" ]; then
  AUTO_ONLY_ARG=(--auto-only)
fi

echo
echo "[step 1] online rollouts + embedded grading + process gate"
"$PYTHON_BIN" "$REPO_ROOT/train/generate_ledger_online_rollouts.py" \
  --tasks-file "$TASKS_FILE" \
  --vllm-base-url "$VLLM_BASE_URL" \
  --model "$SERVED_MODEL" \
  --output-dir "$ROLLOUT_DIR" \
  --n-responses "$N_RESPONSES" \
  --num-workers "$NUM_WORKERS" \
  --judge-model "$JUDGE_MODEL" \
  --judge-base-url "$JUDGE_BASE_URL" \
  "${AUTO_ONLY_ARG[@]}"

GRADED_FILE="$ROLLOUT_DIR/graded_trajectories.jsonl"
if [ ! -s "$GRADED_FILE" ]; then
  echo "ERROR: rollout file missing or empty: $GRADED_FILE"
  exit 1
fi

echo
echo "[step 1 summary]"
"$PYTHON_BIN" - <<PY
import json
from pathlib import Path
p = Path("$GRADED_FILE")
rows = [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
tasks = sorted({r["task_id"] for r in rows})
passed = sum(1 for r in rows if r.get("process_gate_passed"))
print(f"trajectories={len(rows)} tasks={len(tasks)} gate_passed={passed}")
for task in tasks:
    ss = [r.get("score", 0.0) for r in rows if r["task_id"] == task]
    gp = sum(1 for r in rows if r["task_id"] == task and r.get("process_gate_passed"))
    print(f"  {task}: n={len(ss)} min={min(ss):.3f} max={max(ss):.3f} gap={max(ss)-min(ss):.3f} gate={gp}/{len(ss)}")
PY

echo
if [ "$STOP_VLLM_BEFORE_TRAIN" = "1" ]; then
  echo "[gpu] stopping vLLM before logprob/train to free single-GPU memory"
  pkill -f "vllm.entrypoints.openai.api_server" 2>/dev/null || true
  sleep 8
fi

echo
echo "[step 2] compute old-policy logprobs"
CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
"$PYTHON_BIN" "$REPO_ROOT/train/compute_rollout_logprobs.py" \
  --graded-file "$GRADED_FILE" \
  --model-path "$MODEL_PATH" \
  --output "$LOGPROBS_FILE" \
  --max-seq-length "$MAX_SEQ_LEN" \
  --dtype bf16 \
  --max-memory-per-gpu 75GiB

echo
echo "[step 3] train GRPO/PPO LoRA"
CUDA_VISIBLE_DEVICES="$TRAIN_CUDA_VISIBLE_DEVICES" \
PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}" \
"$PYTHON_BIN" -u "$REPO_ROOT/train/train_meeting_grpo_step.py" \
  --graded-file "$GRADED_FILE" \
  --model-path "$MODEL_PATH" \
  --output-dir "$CKPT_DIR" \
  --lr "$LR" \
  --lora-rank "$LORA_RANK" \
  --grad-accum-steps "$GRAD_ACCUM" \
  --max-seq-length "$MAX_SEQ_LEN" \
  --prm-alpha "$PRM_ALPHA" \
  --prm-beta "$PRM_BETA" \
  --prm-mode "$PRM_MODE" \
  --logprobs-file "$LOGPROBS_FILE" \
  --clip-eps "$CLIP_EPS" \
  --kl-beta "$KL_BETA"

echo
echo "════════════════════════════════════════════════════════════════"
echo "  Ledger14 Online RL complete"
echo "════════════════════════════════════════════════════════════════"
echo "  Run dir:      $RUN_DIR"
echo "  Rollouts:     $GRADED_FILE"
echo "  Logprobs:     $LOGPROBS_FILE"
echo "  LoRA adapter: $CKPT_DIR/lora_adapter"
echo "  Log:          $LOG_FILE"
