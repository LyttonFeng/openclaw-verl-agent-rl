#!/usr/bin/env bash
# Conservative task16 RL run: low LR, short horizon, terminal grader reward only.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "${REPO_ROOT}"

RUN_VERSION="${RUN_VERSION:-task16_terminal_only_lr1e6_16step_valcanon5}"
VAL_FILE="${PINCHBENCH_VAL_FILE_OVERRIDE:-${TASK16_DATA_DIR:-${REPO_ROOT}/data/task16_prompts}/val_canonical5.parquet}"
if [ ! -f "${VAL_FILE}" ] && [ -f "${REPO_ROOT}/data/task16_prompts/val.parquet" ]; then
  /usr/bin/python3 - <<'PY'
from pathlib import Path
import pandas as pd

base = Path("data/task16_prompts")
df = pd.read_parquet(base / "val.parquet")
df5 = df.head(5).copy()
df5.to_parquet(base / "val_canonical5.parquet", index=False)
df5.to_json(base / "val_canonical5.jsonl", orient="records", lines=True, force_ascii=False)
print("wrote data/task16_prompts/val_canonical5.parquet rows=5")
PY
fi

export RUN_VERSION
export TRAIN_LOG_PATH="${TRAIN_LOG_PATH:-${REPO_ROOT}/logs/${RUN_VERSION}.log}"
export PINCHBENCH_TRAIN_FILE_OVERRIDE="${PINCHBENCH_TRAIN_FILE_OVERRIDE:-${REPO_ROOT}/data/task16_prompts/train_stage2_balanced.parquet}"
export PINCHBENCH_VAL_FILE_OVERRIDE="${PINCHBENCH_VAL_FILE_OVERRIDE:-${VAL_FILE}}"

export REWARD_MODE=terminal-only
export PINCHBENCH_TERMINAL_REWARD_USE_GRADE_SCORE=1
export PINCHBENCH_TASK16_TERMINAL_REWARD_WEIGHT="${PINCHBENCH_TASK16_TERMINAL_REWARD_WEIGHT:-1.0}"
export PINCHBENCH_TASK16_NO_REPORT_TERMINAL_PENALTY="${PINCHBENCH_TASK16_NO_REPORT_TERMINAL_PENALTY:-0.0}"
export PINCHBENCH_TASK16_EVIDENCE_REWARD_WEIGHT=0.0

export LR="${LR:-1e-6}"
export KL_LOSS_COEF="${KL_LOSS_COEF:-0.05}"
export TOTAL_TRAINING_STEPS=16
export TASK16_TOTAL_TRAINING_STEPS_OVERRIDE=16
export VAL_BEFORE_TRAIN=True
export TEST_FREQ=4
export TASK16_TEST_FREQ_OVERRIDE=4
export SAVE_FREQ=4
export TASK16_SAVE_FREQ_OVERRIDE=4

if [ -n "${TASK16_MAX_PROMPT_LENGTH_OVERRIDE:-}" ]; then
  export MAX_PROMPT_LENGTH="${TASK16_MAX_PROMPT_LENGTH_OVERRIDE}"
elif [ -z "${MAX_PROMPT_LENGTH:-}" ] || [ "${MAX_PROMPT_LENGTH}" = "16000" ]; then
  export MAX_PROMPT_LENGTH=18000
fi
if [ -n "${TASK16_MAX_RESPONSE_LENGTH_OVERRIDE:-}" ]; then
  export MAX_RESPONSE_LENGTH="${TASK16_MAX_RESPONSE_LENGTH_OVERRIDE}"
elif [ -z "${MAX_RESPONSE_LENGTH:-}" ] || [ "${MAX_RESPONSE_LENGTH}" = "4096" ]; then
  export MAX_RESPONSE_LENGTH=12000
fi
export MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-18000}"
export TASK16_MAX_PROMPT_LENGTH_OVERRIDE="${TASK16_MAX_PROMPT_LENGTH_OVERRIDE:-${MAX_PROMPT_LENGTH}}"
export MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-12000}"
export PINCHBENCH_TASK16_MAX_TOKENS_PER_TURN="${PINCHBENCH_TASK16_MAX_TOKENS_PER_TURN:-2048}"
export VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"

exec bash scripts/run_task16_rl.sh "$@"
