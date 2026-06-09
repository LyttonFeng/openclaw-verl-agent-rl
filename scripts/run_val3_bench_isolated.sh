#!/usr/bin/env bash
# Run the meeting_analysis Val3 benchmark in an isolated OpenClaw home.
#
# Val3 is the fast diagnostic subset used for stable single-agent RL iteration:
#   - task_meeting_advisory_stakeholders
#   - task_meeting_gov_speaker_summary
#   - task_meeting_tech_action_items
#
# This wrapper defaults to the local Qwen3-4B base endpoint and forces
# temperature=0 for stable baseline runs.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL="${MODEL:-Qwen3-4B-base}"
BASE_URL="${BASE_URL:-http://127.0.0.1:8021/v1}"
RUNS="${RUNS:-3}"
TIMEOUT_MULTIPLIER="${TIMEOUT_MULTIPLIER:-3}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v4-pro}"
TASKS_DIR="${TASKS_DIR:-$REPO_ROOT/data/eval/val5}"
SKILL_DIR="${PINCHBENCH_SKILL_DIR:-$REPO_ROOT/data/eval}"
VAL3_TASKS="${VAL3_TASKS:-task_meeting_advisory_stakeholders,task_meeting_gov_speaker_summary,task_meeting_tech_action_items}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$RANDOM}"
OPENCLAW_HOME_ROOT="${OPENCLAW_HOME_ROOT:-/tmp/openclaw_home}"
PINCHBENCH_ROOT="${PINCHBENCH_ROOT:-/tmp/pinchbench}"
export OPENCLAW_HOME="${OPENCLAW_HOME:-$OPENCLAW_HOME_ROOT/$RUN_ID}"
export PINCHBENCH_OPENCLAW_HOME="${PINCHBENCH_OPENCLAW_HOME:-$OPENCLAW_HOME/.openclaw}"
export PINCHBENCH_RUN_ROOT="${PINCHBENCH_RUN_ROOT:-$PINCHBENCH_ROOT/$RUN_ID}"

# OpenClaw normalizes/truncates long agent IDs. Benchmark transcript lookup uses
# the requested agent ID, so a truncated ID means "sessions dir does not exist".
# Keep the default suffix short even when RUN_ID is verbose.
if [ -z "${PINCHBENCH_AGENT_SUFFIX:-}" ]; then
  RUN_ID_SHORT="$(printf '%s' "$RUN_ID" | tr -cd '[:alnum:]_-' | cut -c1-12)"
  RUN_ID_CKSUM="$(printf '%s' "$RUN_ID" | cksum | awk '{print $1}')"
  export PINCHBENCH_AGENT_SUFFIX="v3_${RUN_ID_SHORT}_${RUN_ID_CKSUM}"
else
  export PINCHBENCH_AGENT_SUFFIX
fi
if [ "${#PINCHBENCH_AGENT_SUFFIX}" -gt 32 ]; then
  SUFFIX_SHORT="$(printf '%s' "$PINCHBENCH_AGENT_SUFFIX" | tr -cd '[:alnum:]_-' | cut -c1-16)"
  SUFFIX_CKSUM="$(printf '%s' "$PINCHBENCH_AGENT_SUFFIX" | cksum | awk '{print $1}')"
  export PINCHBENCH_AGENT_SUFFIX="${SUFFIX_SHORT}_${SUFFIX_CKSUM}"
fi

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/val3_isolated/$RUN_ID}"
KEEP_OPENCLAW_HOME="${KEEP_OPENCLAW_HOME:-0}"
KEEP_PINCHBENCH_RUN_ROOT="${KEEP_PINCHBENCH_RUN_ROOT:-0}"
PRE_CLEAN_STALE_OPENCLAW="${PRE_CLEAN_STALE_OPENCLAW:-1}"

pre_clean_stale_openclaw() {
  if [ "$PRE_CLEAN_STALE_OPENCLAW" != "1" ]; then
    return 0
  fi

  echo "Pre-cleaning stale Val3 OpenClaw state"
  echo "  agent_suffix:         $PINCHBENCH_AGENT_SUFFIX"
  echo "  run_id:               $RUN_ID"

  # Keep cleanup scoped to this isolated run id / agent suffix. Do not match
  # the model server process; vLLM is intentionally shared across benchmark runs.
  pkill -f "scripts/benchmark.py.*${RUN_ID}" 2>/dev/null || true
  pkill -f "scripts/benchmark.py.*${PINCHBENCH_AGENT_SUFFIX}" 2>/dev/null || true
  pkill -f "bench-.*${PINCHBENCH_AGENT_SUFFIX}" 2>/dev/null || true
  pkill -f "openclaw.*${PINCHBENCH_AGENT_SUFFIX}" 2>/dev/null || true

  # Stale isolated homes can confuse OpenClaw agent discovery on reruns.
  if [ "$KEEP_OPENCLAW_HOME" != "1" ]; then
    rm -rf "$OPENCLAW_HOME"
  fi
  if [ "$KEEP_PINCHBENCH_RUN_ROOT" != "1" ]; then
    rm -rf "$PINCHBENCH_RUN_ROOT"
  fi
}

cleanup() {
  local status=$?
  trap - EXIT INT TERM
  jobs -pr | xargs -r kill 2>/dev/null || true
  pkill -P $$ 2>/dev/null || true
  if [ "$KEEP_OPENCLAW_HOME" != "1" ]; then
    rm -rf "$OPENCLAW_HOME"
  fi
  if [ "$KEEP_PINCHBENCH_RUN_ROOT" != "1" ]; then
    rm -rf "$PINCHBENCH_RUN_ROOT"
  fi
  exit "$status"
}
trap cleanup EXIT INT TERM

pre_clean_stale_openclaw
mkdir -p "$PINCHBENCH_OPENCLAW_HOME" "$PINCHBENCH_RUN_ROOT" "$OUTPUT_DIR"

# Stable OpenClaw runtime settings for long meeting_analysis tasks.
python3 - <<'PY'
import json
import os
from pathlib import Path

store = Path(os.environ["PINCHBENCH_OPENCLAW_HOME"])
store.mkdir(parents=True, exist_ok=True)
path = store / "openclaw.json"
data = {}
if path.exists():
    try:
        data = json.loads(path.read_text("utf-8"))
    except json.JSONDecodeError:
        data = {}
agents = data.setdefault("agents", {})
agents.setdefault("list", [])
defaults = agents.setdefault("defaults", {})
defaults["timeoutSeconds"] = int(os.environ.get("OPENCLAW_AGENT_TIMEOUT_SECONDS", "600"))
llm = defaults.setdefault("llm", {})
llm["idleTimeoutSeconds"] = int(os.environ.get("OPENCLAW_LLM_IDLE_TIMEOUT_SECONDS", "0"))
path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", "utf-8")
PY

export PINCHBENCH_TASKS_DIR="$TASKS_DIR"
export PINCHBENCH_SKILL_DIR="$SKILL_DIR"
export PINCHBENCH_GRADE_JUDGE_BASE_URL="${PINCHBENCH_GRADE_JUDGE_BASE_URL:-https://api.deepseek.com/v1}"
export PINCHBENCH_GRADE_JUDGE_MODEL="$JUDGE_MODEL"
# Val3 uses this pod's local OpenClaw CLI/runtime. Ignore stale remote/ECS
# variables that may exist in ~/.pinchbench_env from older experiments.
unset OPENCLAW_HOST ECS_HOST OPENCLAW_PORT OPENCLAW_USER OPENCLAW_SSH_KEY
export PINCHBENCH_FORCE_LOCAL_OPENCLAW="${PINCHBENCH_FORCE_LOCAL_OPENCLAW:-1}"
export PINCHBENCH_SKIP_OPENCLAW_WEB_PREFLIGHT="${PINCHBENCH_SKIP_OPENCLAW_WEB_PREFLIGHT:-1}"
export PINCHBENCH_OPENCLAW_CONTEXT_WINDOW="${PINCHBENCH_OPENCLAW_CONTEXT_WINDOW:-65536}"
export PINCHBENCH_OPENCLAW_MAX_TOKENS="${PINCHBENCH_OPENCLAW_MAX_TOKENS:-8192}"
export PINCHBENCH_MODEL_TEMPERATURE="${PINCHBENCH_MODEL_TEMPERATURE:-0}"
export OPENCLAW_MODEL_REASONING="${OPENCLAW_MODEL_REASONING:-0}"

MODEL_API_KEY="${PINCHBENCH_MODEL_API_KEY:-${DEEPSEEK_API_KEY:-dummy}}"

echo "Val3 isolated benchmark"
echo "  run_id:               $RUN_ID"
echo "  model:                $MODEL"
echo "  base_url:             $BASE_URL"
echo "  output_dir:           $OUTPUT_DIR"
echo "  openclaw_home:        $OPENCLAW_HOME"
echo "  pinchbench_run_root:  $PINCHBENCH_RUN_ROOT"
echo "  agent_suffix:         $PINCHBENCH_AGENT_SUFFIX"
echo "  runs:                 $RUNS"
echo "  temperature:          $PINCHBENCH_MODEL_TEMPERATURE"
echo "  timeout_multiplier:   $TIMEOUT_MULTIPLIER"
echo "  tasks_dir:            $TASKS_DIR"
echo "  skill_dir:            $PINCHBENCH_SKILL_DIR"
echo "  tasks:                $VAL3_TASKS"

"$PYTHON_BIN" "$REPO_ROOT/scripts/benchmark.py" \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --api-key "$MODEL_API_KEY" \
  --suite "$VAL3_TASKS" \
  --runs "$RUNS" \
  --timeout-multiplier "$TIMEOUT_MULTIPLIER" \
  --judge "$JUDGE_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --no-upload \
  --no-parallel-judge \
  --no-judge-cache \
  --no-fail-fast \
  "$@"
