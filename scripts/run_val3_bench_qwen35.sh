#!/usr/bin/env bash
# Isolated Val3 benchmark for Qwen3.5-4B (non-think). Thin wrapper over
# run_val3_bench_isolated.sh that pins the Qwen3.5 serving contract:
#   * served model qwen35-4b on the local vLLM (start_qwen35_vllm.sh, port 8023)
#   * OPENCLAW_MODEL_REASONING=0  -> non-think (template default is also non-think)
#   * native tool calls via the server's qwen3_coder parser
#
# Reference result (2026-06-11, fresh local weights, no training):
#   advisory_stakeholders 0.952 | gov_speaker_summary 0.586 | tech_action_items 0.717
#   OVERALL MEETING 75.2%   (vs Qwen3-4B base ~47-50%, trained C26 ~50.5%)
#
# Prereqs: vLLM already serving qwen35-4b (scripts/start_qwen35_vllm.sh) and
# ~/.pinchbench_env sourced (DEEPSEEK_API_KEY for the grading judge).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"

export MODEL="${MODEL:-qwen35-4b}"
export BASE_URL="${BASE_URL:-http://127.0.0.1:8023/v1}"
export PINCHBENCH_MODEL_API_KEY="${PINCHBENCH_MODEL_API_KEY:-dummy}"
export OPENCLAW_MODEL_REASONING="${OPENCLAW_MODEL_REASONING:-0}"          # non-think
export PINCHBENCH_OPENCLAW_MAX_TOKENS="${PINCHBENCH_OPENCLAW_MAX_TOKENS:-16384}"
export PINCHBENCH_MODEL_TEMPERATURE="${PINCHBENCH_MODEL_TEMPERATURE:-0}"
export VAL3_TASKS="${VAL3_TASKS:-task_meeting_advisory_stakeholders,task_meeting_gov_speaker_summary,task_meeting_tech_action_items}"
export RUNS="${RUNS:-3}"
export OUTPUT_DIR="${OUTPUT_DIR:-/tmp/qwen35_val3}"
export PYTHON_BIN="${PYTHON_BIN:-/root/openclaw-venv/bin/python}"

echo "Val3 bench (Qwen3.5-4B non-think): model=$MODEL base_url=$BASE_URL runs=$RUNS reasoning=$OPENCLAW_MODEL_REASONING"
exec bash "$REPO_ROOT/scripts/run_val3_bench_isolated.sh"
