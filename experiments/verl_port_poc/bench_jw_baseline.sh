#!/usr/bin/env bash
# Reproducible benchmark launcher for jiuwenclaw + Qwen3-4B base on the 5
# meeting_analysis val tasks. Wraps the official jiuwenclaw bench harness
# (run_pinchbench_jiuwenclaw.py) with the path / env setup that took
# multiple iterations to get right.
#
# Prereqs on the pod:
#   - /root/jiuwen_work/jiuwenclaw/.venv (uv-managed) with openjiuwen + jiuwen
#   - /root/jiuwen_work/pinchbench/scripts/run_pinchbench_jiuwenclaw.py
#   - /workspace/openclaw-verl-agent-rl/pinchbench_tasks/meeting_analysis/*
#   - A standalone vLLM at $API_BASE serving the model under $MODEL_NAME
#   - DEEPSEEK_API_KEY exported (used by the LLM judge)
#
# Critical correctness gotchas (see memory/bench_jiuwenclaw_path_mismatch.md):
#   - bench MUST pass --jiuwen-data-dir matching the stack's JIUWENCLAW_DATA_DIR.
#     If they diverge, workspace_files are copied to one dir but jiuwen reads
#     the other → File not found → all tasks score 0.
#
# Usage:
#   API_BASE=http://127.0.0.1:8123/v1 MODEL_NAME=Qwen3-4B \
#     bash bench_jw_baseline.sh ./out_dir [num_runs]

set -euo pipefail

OUT_ROOT="${1:-/workspace/verl_port/bench/jiuwen_baseline}"
N_RUNS="${2:-3}"

: "${API_BASE:?Set API_BASE (vLLM endpoint, e.g. http://127.0.0.1:8123/v1)}"
: "${MODEL_NAME:?Set MODEL_NAME (served_model_name on the vLLM)}"
: "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY for the LLM judge}"

API_KEY="${API_KEY:-dummy}"
MODEL_PROVIDER="${MODEL_PROVIDER:-OpenAI}"
WS_PORT="${WS_PORT:-611}"
AGENT_SERVER_PORT="${AGENT_SERVER_PORT:-18092}"
GATEWAY_PORT="${GATEWAY_PORT:-19001}"
JIUWENCLAW_DATA_DIR="${JIUWENCLAW_DATA_DIR:-/root/.jiuwenclaw}"
JIUWENCLAW_REPO="${JIUWENCLAW_REPO:-/root/jiuwen_work/jiuwenclaw}"
SKILL_ROOT="${SKILL_ROOT:-/workspace/openclaw-verl-agent-rl}"
SUITE="${SUITE:-task_meeting_advisory_stakeholders,task_meeting_council_votes,task_meeting_gov_speaker_summary,task_meeting_tech_action_items,task_meeting_sentiment_analysis}"
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-chat}"

LOG_DIR="${LOG_DIR:-/tmp/jw_bench}"
mkdir -p "$LOG_DIR" "$OUT_ROOT"

# 1. Make sure the standalone jiuwenclaw stack is up on $WS_PORT.
if ! ss -tln 2>/dev/null | awk 'NR>1 {print $4}' | grep -qE ":${WS_PORT}\$"; then
  echo "[bench] starting headless jiuwenclaw stack on ws://127.0.0.1:${WS_PORT}/ws"
  API_BASE="$API_BASE" API_KEY="$API_KEY" MODEL_NAME="$MODEL_NAME" MODEL_PROVIDER="$MODEL_PROVIDER" \
    WS_PORT="$WS_PORT" AGENT_SERVER_PORT="$AGENT_SERVER_PORT" GATEWAY_PORT="$GATEWAY_PORT" \
    JIUWENCLAW_DATA_DIR="$JIUWENCLAW_DATA_DIR" LOG_DIR="${LOG_DIR}/stack" \
    USE_RL_ONLINE_RAIL=0 MEMORY_ENABLED=false MAX_ITERATIONS=12 \
    JIUWENCLAW_REPO="$JIUWENCLAW_REPO" \
    setsid nohup bash "$(dirname "${BASH_SOURCE[0]}")/start_jw_headless.sh" \
    < /dev/null > "${LOG_DIR}/stack.out" 2>&1 &
  for _ in $(seq 1 30); do
    if ss -tln 2>/dev/null | awk 'NR>1 {print $4}' | grep -qE ":${WS_PORT}\$"; then break; fi
    sleep 2
  done
  if ! ss -tln 2>/dev/null | awk 'NR>1 {print $4}' | grep -qE ":${WS_PORT}\$"; then
    echo "[bench] FATAL: stack did not bind ${WS_PORT} in 60s"
    tail -30 "${LOG_DIR}/stack.out"
    exit 1
  fi
  echo "[bench] stack ready"
fi

# 2. Run the bench N times.
export DEEPSEEK_API_KEY OPENAI_API_KEY="${OPENAI_API_KEY:-$DEEPSEEK_API_KEY}" \
       PINCHBENCH_GRADE_JUDGE_API_KEY="${PINCHBENCH_GRADE_JUDGE_API_KEY:-$DEEPSEEK_API_KEY}"

for i in $(seq 1 "$N_RUNS"); do
  OUT="${OUT_ROOT}/run${i}"
  mkdir -p "$OUT"
  echo "[bench] === run ${i}/${N_RUNS} starting $(date) ==="
  cd "$JIUWENCLAW_REPO"
  "${JIUWENCLAW_REPO}/.venv/bin/python" \
    /root/jiuwen_work/pinchbench/scripts/run_pinchbench_jiuwenclaw.py \
    --skill-root "$SKILL_ROOT" \
    --ws-url "ws://127.0.0.1:${WS_PORT}/ws" \
    --jiuwen-data-dir "$JIUWENCLAW_DATA_DIR" \
    --suite "$SUITE" \
    --judge-model "$JUDGE_MODEL" \
    --judge-backend api \
    --output-dir "$OUT"
  echo "[bench] === run ${i}/${N_RUNS} done $(date) ==="
done

# 3. Summarize all runs.
python3 "$(dirname "${BASH_SOURCE[0]}")/bench_summarize.py" "$OUT_ROOT"
