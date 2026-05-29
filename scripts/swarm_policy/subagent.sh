#!/bin/bash
# Lead invokes this via exec: subagent.sh "<task description>" <source_file> [<offset>] [<length>]
# Returns: stdout = sub-agent's final summary text.
# Sub-agent runs frozen Qwen3-4B base on a dedicated vLLM (port 8767 by default).
set -e

export SUBAGENT_MODEL="${SUBAGENT_MODEL:-qwen3-base}"
export SUBAGENT_VLLM_URL="${SUBAGENT_VLLM_URL:-http://localhost:8767/v1}"
export SUBAGENT_TIMEOUT="${SUBAGENT_TIMEOUT:-420}"
export SUBAGENT_LOG_DIR="${SUBAGENT_LOG_DIR:-/tmp/subagent_logs}"
export PINCHBENCH_FORCE_LOCAL_OPENCLAW=1
export OPENCLAW_HOST=localhost
export PINCHBENCH_SKIP_OPENCLAW_WEB_PREFLIGHT=1
export PINCHBENCH_ALLOW_LOCAL_OPENCLAW=1

exec /root/openclaw-venv/bin/python "$(dirname "$0")/run_subagent.py" "$@"
