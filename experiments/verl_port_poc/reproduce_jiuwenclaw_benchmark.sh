#!/usr/bin/env bash
# One-command reproduction for JiuwenClaw + Qwen3-4B meeting_analysis benchmark.
#
# Usage:
#   source ~/.pinchbench_env
#   bash experiments/verl_port_poc/reproduce_jiuwenclaw_benchmark.sh \
#     /workspace/verl_port/bench/jiuwen_base_4b_repro 3

set -euo pipefail

OUT_ROOT="${1:-/workspace/verl_port/bench/jiuwen_base_4b_repro}"
N_RUNS="${2:-3}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

: "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY, e.g. source ~/.pinchbench_env}"

export MODEL_DIR="${MODEL_DIR:-/root/hf_cache/Qwen3-4B}"
export PORT="${PORT:-8123}"
export SERVED_NAME="${SERVED_NAME:-Qwen3-4B}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export API_BASE="${API_BASE:-http://127.0.0.1:${PORT}/v1}"
export MODEL_NAME="${MODEL_NAME:-${SERVED_NAME}}"
export JIUWENCLAW_REPO="${JIUWENCLAW_REPO:-/root/jiuwen_work/jiuwenclaw}"
export SKILL_ROOT="${SKILL_ROOT:-${REPO_ROOT}}"
export JIUWENCLAW_DATA_DIR="${JIUWENCLAW_DATA_DIR:-/root/.jiuwenclaw}"

if [ ! -d "$JIUWENCLAW_REPO" ]; then
  echo "[repro] FATAL: JIUWENCLAW_REPO not found: $JIUWENCLAW_REPO" >&2
  exit 1
fi

if [ ! -f "$MODEL_DIR/config.json" ]; then
  echo "[repro] FATAL: MODEL_DIR does not contain config.json: $MODEL_DIR" >&2
  echo "[repro] Set MODEL_DIR=/path/to/Qwen3-4B or download Qwen/Qwen3-4B first." >&2
  exit 1
fi

echo "[repro] repo=$REPO_ROOT"
echo "[repro] model=$MODEL_DIR served=$SERVED_NAME api=$API_BASE"
echo "[repro] jiuwenclaw=$JIUWENCLAW_REPO data=$JIUWENCLAW_DATA_DIR"
echo "[repro] out=$OUT_ROOT runs=$N_RUNS"

bash "${SCRIPT_DIR}/launch_vllm_qwen3_base.sh"

API_BASE="$API_BASE" MODEL_NAME="$MODEL_NAME" \
  bash "${SCRIPT_DIR}/bench_jw_baseline.sh" "$OUT_ROOT" "$N_RUNS"
