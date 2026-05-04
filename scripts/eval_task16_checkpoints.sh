#!/usr/bin/env bash
# Evaluate every saved LoRA checkpoint on real task16 using the full benchmark repo.

set -euo pipefail

CKPT_ROOT="${CKPT_ROOT:?set CKPT_ROOT to the run checkpoint directory}"
BENCH_REPO="${BENCH_REPO:-/Users/lytton/work/reinforement_learning/pinchbench-skill}"
BASE_MODEL="${BASE_MODEL:-Qwen/Qwen3-1.7B}"
REMOTE_HOST="${REMOTE_HOST:-154.54.102.41}"
REMOTE_PORT="${REMOTE_PORT:-18230}"
REMOTE_USER="${REMOTE_USER:-root}"
SSH_KEY="${SSH_KEY:-${HOME}/.ssh/id_ed25519}"
REMOTE_VLLM_PORT="${REMOTE_VLLM_PORT:-8010}"
LOCAL_VLLM_PORT="${LOCAL_VLLM_PORT:-18010}"
MODEL_NAME="${MODEL_NAME:-task16-lora}"

for adapter in "${CKPT_ROOT}"/global_step_*/actor/lora_adapter; do
  [ -d "${adapter}" ] || continue
  step="$(basename "$(dirname "$(dirname "${adapter}")")")"
  echo "== eval ${step} =="

  ssh -o StrictHostKeyChecking=no -i "${SSH_KEY}" "${REMOTE_USER}@${REMOTE_HOST}" -p "${REMOTE_PORT}" \
    "pkill -f 'vllm.entrypoints.openai.api_server' || true"

  ssh -fN -o StrictHostKeyChecking=no -o ServerAliveInterval=30 -o ExitOnForwardFailure=yes \
    -L "127.0.0.1:${LOCAL_VLLM_PORT}:127.0.0.1:${REMOTE_VLLM_PORT}" \
    -i "${SSH_KEY}" "${REMOTE_USER}@${REMOTE_HOST}" -p "${REMOTE_PORT}" || true

  ssh -f -o StrictHostKeyChecking=no -i "${SSH_KEY}" "${REMOTE_USER}@${REMOTE_HOST}" -p "${REMOTE_PORT}" \
    "cd /workspace/openclaw-verl-agent-rl && BASE_MODEL='${BASE_MODEL}' LORA_PATH='${adapter}' SERVED_MODEL_NAME='${MODEL_NAME}' PORT='${REMOTE_VLLM_PORT}' nohup bash scripts/start_vllm_lora.sh > logs/eval_${step}.vllm.out 2>&1"

  for _ in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:${LOCAL_VLLM_PORT}/v1/models" >/dev/null 2>&1; then
      break
    fi
    sleep 2
  done

  (
    cd "${BENCH_REPO}"
    MODEL="${MODEL_NAME}" \
    BASE_URL="http://127.0.0.1:${LOCAL_VLLM_PORT}/v1" \
    SAVE_RL8_COMPARE=1 \
    RL8_COMPARE_PREFIX="task16_terminal16_${step}" \
      bash scripts/run_bench_rl8.sh --suite task_16_email_triage
  )
done
