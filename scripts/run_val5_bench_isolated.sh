#!/usr/bin/env bash
# Run the meeting_analysis Val5 benchmark in an isolated OpenClaw home.
#
# This wrapper prevents stale OpenClaw agents/sessions from contaminating
# benchmark scores by giving every run a private OPENCLAW_HOME, run root, and
# agent namespace. Temporary runtime state is removed on success, failure, or
# interruption. Result files under OUTPUT_DIR are kept.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MODEL="${MODEL:-deepseek-v4-flash}"
BASE_URL="${BASE_URL:-https://api.deepseek.com/v1}"
RUNS="${RUNS:-3}"
TIMEOUT_MULTIPLIER="${TIMEOUT_MULTIPLIER:-3}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-v4-pro}"
TASKS_DIR="${TASKS_DIR:-$REPO_ROOT/pinchbench_tasks/meeting_analysis}"
VAL5_TASKS="${VAL5_TASKS:-task_meeting_advisory_stakeholders,task_meeting_council_votes,task_meeting_gov_speaker_summary,task_meeting_tech_action_items,task_meeting_sentiment_analysis}"

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)_$RANDOM}"
OPENCLAW_HOME_ROOT="${OPENCLAW_HOME_ROOT:-/tmp/openclaw_home}"
PINCHBENCH_ROOT="${PINCHBENCH_ROOT:-/tmp/pinchbench}"
export OPENCLAW_HOME="${OPENCLAW_HOME:-$OPENCLAW_HOME_ROOT/$RUN_ID}"
export PINCHBENCH_OPENCLAW_HOME="${PINCHBENCH_OPENCLAW_HOME:-$OPENCLAW_HOME/.openclaw}"
export PINCHBENCH_RUN_ROOT="${PINCHBENCH_RUN_ROOT:-$PINCHBENCH_ROOT/$RUN_ID}"
export PINCHBENCH_AGENT_SUFFIX="${PINCHBENCH_AGENT_SUFFIX:-$RUN_ID}"

OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/results/val5_isolated/$RUN_ID}"
KEEP_OPENCLAW_HOME="${KEEP_OPENCLAW_HOME:-0}"
KEEP_PINCHBENCH_RUN_ROOT="${KEEP_PINCHBENCH_RUN_ROOT:-0}"

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

mkdir -p "$PINCHBENCH_OPENCLAW_HOME" "$PINCHBENCH_RUN_ROOT" "$OUTPUT_DIR"

# Align with the known stable OpenClaw runtime settings for long multi-turn
# meeting_analysis tasks: no stream idle timeout, bounded total timeout.
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
export PINCHBENCH_GRADE_JUDGE_BASE_URL="${PINCHBENCH_GRADE_JUDGE_BASE_URL:-https://api.deepseek.com/v1}"
export PINCHBENCH_GRADE_JUDGE_MODEL="$JUDGE_MODEL"
export PINCHBENCH_FORCE_LOCAL_OPENCLAW="${PINCHBENCH_FORCE_LOCAL_OPENCLAW:-1}"
export PINCHBENCH_SKIP_OPENCLAW_WEB_PREFLIGHT="${PINCHBENCH_SKIP_OPENCLAW_WEB_PREFLIGHT:-1}"
export PINCHBENCH_OPENCLAW_CONTEXT_WINDOW="${PINCHBENCH_OPENCLAW_CONTEXT_WINDOW:-65536}"
export PINCHBENCH_OPENCLAW_MAX_TOKENS="${PINCHBENCH_OPENCLAW_MAX_TOKENS:-8192}"
export OPENCLAW_MODEL_REASONING="${OPENCLAW_MODEL_REASONING:-0}"

MODEL_API_KEY="${PINCHBENCH_MODEL_API_KEY:-${DEEPSEEK_API_KEY:-dummy}}"

echo "Val5 isolated benchmark"
echo "  run_id:               $RUN_ID"
echo "  model:                $MODEL"
echo "  base_url:             $BASE_URL"
echo "  output_dir:           $OUTPUT_DIR"
echo "  openclaw_home:        $OPENCLAW_HOME"
echo "  pinchbench_run_root:  $PINCHBENCH_RUN_ROOT"
echo "  agent_suffix:         $PINCHBENCH_AGENT_SUFFIX"
echo "  runs:                 $RUNS"
echo "  timeout_multiplier:   $TIMEOUT_MULTIPLIER"

"$PYTHON_BIN" "$REPO_ROOT/scripts/benchmark.py" \
  --model "$MODEL" \
  --base-url "$BASE_URL" \
  --api-key "$MODEL_API_KEY" \
  --suite "$VAL5_TASKS" \
  --runs "$RUNS" \
  --timeout-multiplier "$TIMEOUT_MULTIPLIER" \
  --judge "$JUDGE_MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --no-upload \
  --no-parallel-judge \
  --no-judge-cache \
  --no-fail-fast \
  "$@"
