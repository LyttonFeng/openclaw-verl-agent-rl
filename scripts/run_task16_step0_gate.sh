#!/usr/bin/env bash
# Training-side step0 gate for task16.
#
# This intentionally uses the veRL rollout entrypoint, not scripts/benchmark.py.
# It validates that the training-side ModelProxy/OpenClaw/grading path can
# create and grade triage_report.md before any RL updates are trusted.

set -euo pipefail

if [ -f "${HOME}/.pinchbench_env" ]; then
  set -a
  # shellcheck disable=SC1090
  source "${HOME}/.pinchbench_env"
  set +a
fi

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
TASK16_DATA_DIR="${TASK16_DATA_DIR:-${REPO_ROOT}/data/task16_prompts}"
CANONICAL_REPEATS="${GATE_CANONICAL_VAL_REPEATS:-3}"
RUN_VERSION="${GATE_RUN_VERSION:-task16_step0_gate_qwen4b_$(date +%Y%m%d_%H%M%S)}"
VAL_FILE="${TASK16_DATA_DIR}/val_canonical_step0_gate_${CANONICAL_REPEATS}.parquet"

mkdir -p "${TASK16_DATA_DIR}" "${REPO_ROOT}/logs"

if [ ! -f "${TASK16_DATA_DIR}/train.parquet" ]; then
  python3 "${REPO_ROOT}/scripts/build_task16_prompts.py" \
    --tasks-dir "${REPO_ROOT}/pinchbench_tasks" \
    --output-dir "${TASK16_DATA_DIR}"
fi

python3 - <<'PY' "${TASK16_DATA_DIR}/train.parquet" "${VAL_FILE}" "${CANONICAL_REPEATS}"
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

train_path = Path(sys.argv[1])
out_path = Path(sys.argv[2])
repeats = int(sys.argv[3])

df = pd.read_parquet(train_path)
canonical = None
for _, row in df.iterrows():
    extra = row.get("extra_info")
    if isinstance(extra, dict) and extra.get("prompt_group") == "canonical":
        canonical = row.to_dict()
        break
if canonical is None:
    canonical = df.iloc[0].to_dict()

rows = []
for i in range(repeats):
    row = dict(canonical)
    extra = dict(row.get("extra_info") or {})
    extra["repeat_idx"] = 9000 + i
    extra["prompt_group"] = "canonical_step0_gate"
    row["extra_info"] = extra
    rows.append(row)

pd.DataFrame(rows).to_parquet(out_path, index=False)
print(f"Wrote canonical step0 val: {out_path} rows={len(rows)}")
PY

export RUN_VERSION
export VERL_MODEL="${VERL_MODEL:-Qwen/Qwen3-4B}"
export LORA_RANK="${LORA_RANK:-0}"
export PINCHBENCH_VAL_FILE_OVERRIDE="${VAL_FILE}"
export VAL_MAX_SAMPLES="${GATE_VAL_MAX_SAMPLES:-${CANONICAL_REPEATS}}"
export VAL_BEFORE_TRAIN=True
export VAL_ONLY=True
export TOTAL_TRAINING_STEPS=1
export TEST_FREQ=1
export SAVE_FREQ=1000000
export BATCH_SIZE=1
export MICRO_BATCH=1
export MAX_TURNS="${MAX_TURNS:-8}"
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-20000}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-12000}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
export VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.28}"
export VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
export PINCHBENCH_AGENT_LOOP_RUNTIME=benchmark
export PINCHBENCH_OPENCLAW_CONTEXT_WINDOW="${PINCHBENCH_OPENCLAW_CONTEXT_WINDOW:-32768}"
export PINCHBENCH_OPENCLAW_MAX_TOKENS="${PINCHBENCH_OPENCLAW_MAX_TOKENS:-8192}"
export PINCHBENCH_MODEL_TEMPERATURE="${PINCHBENCH_MODEL_TEMPERATURE:-0.7}"
export PINCHBENCH_MODEL_TOP_P="${PINCHBENCH_MODEL_TOP_P:-0.8}"
export PINCHBENCH_MODEL_TOP_K="${PINCHBENCH_MODEL_TOP_K:-20}"
export PINCHBENCH_BENCHMARK_MODEL_ID="${PINCHBENCH_BENCHMARK_MODEL_ID:-${VERL_MODEL##*/}}"
export OPENCLAW_MODEL_REASONING="${OPENCLAW_MODEL_REASONING:-0}"
export PINCHBENCH_RL_INJECT_TOOL_FORMAT_SUFFIX="${PINCHBENCH_RL_INJECT_TOOL_FORMAT_SUFFIX:-1}"
export TRAIN_LOG_PATH="${TRAIN_LOG_PATH:-${REPO_ROOT}/logs/${RUN_VERSION}.train.log}"

echo "=============================="
echo "task16 training-side step0 gate"
echo "RUN_VERSION=${RUN_VERSION}"
echo "MODEL=${VERL_MODEL} served=${PINCHBENCH_BENCHMARK_MODEL_ID}"
echo "VAL_FILE=${PINCHBENCH_VAL_FILE_OVERRIDE}"
echo "OPENCLAW=${OPENCLAW_USER:-root}@${OPENCLAW_HOST:-<unset>}:${OPENCLAW_PORT:-22}"
echo "RL_TOOL_CALL_SUFFIX=${PINCHBENCH_RL_INJECT_TOOL_FORMAT_SUFFIX}"
echo "LOG=${TRAIN_LOG_PATH}"
echo "=============================="

bash "${REPO_ROOT}/scripts/run_task16_rl.sh"
