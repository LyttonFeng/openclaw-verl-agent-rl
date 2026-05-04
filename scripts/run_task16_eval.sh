#!/usr/bin/env bash
# Minimal eval hook: verify data and environment, then point users at OpenClaw/PinchBench eval.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DATA_DIR="${DATA_DIR:-${REPO_ROOT}/data/task16_prompts}"

python3 "${REPO_ROOT}/scripts/check_data.py" "${DATA_DIR}"
python3 "${REPO_ROOT}/scripts/check_env.py"

cat <<'EOF'
Data and environment checks passed.

For checkpoint eval, serve the LoRA with scripts/start_vllm_lora.sh and run the
same OpenClaw task16 grader path used by training. This repo keeps only the RL
reproduction pieces; it does not vendor the full benchmark harness.
EOF
