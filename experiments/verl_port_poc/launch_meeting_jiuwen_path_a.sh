#!/usr/bin/env bash
# Path A orchestrator: veRL hybrid engine + headless jiuwenclaw (no separate vLLM).
#
# Architecture:
#   veRL hybrid engine
#     ├─ FSDP actor (on GPUs)
#     ├─ vLLM (hybrid, on same GPUs) ← always-latest weights, exposed as HTTP
#     │    serves /v1/chat/completions at $VERL_VLLM_URL
#     └─ AgentLoopWorker → JiuwenClawAgentLoop.run() → WS to port 611
#   Headless jiuwenclaw (no own vLLM)
#     ├─ agent_server (WS port 611) ← veRL's entry point for rollouts
#     ├─ gateway (port 613)
#     └─ OpenAIModelClient reads $API_BASE → POSTs back to $VERL_VLLM_URL
#
# Bootstrap order:
#   1. Pre-flight (kill stale stacks, check GPUs).
#   2. Start veRL in background — its vLLM HTTP server comes up during init
#      and prints "LLMServerManager: ['IP:PORT', ...]" to stdout.
#   3. Tail veRL log, extract URL; bail if not seen within timeout.
#   4. Start headless jiuwenclaw with API_BASE = that URL.
#   5. veRL's first rollout fires shortly after; if jiuwenclaw WS is up by
#      then, training proceeds normally.
#
# DRY RUN: pass `--dry-run` to skip launching veRL — just verifies the
# headless stack + external vLLM wiring works. Useful for smoke testing
# this script against an already-running vLLM on the pod.

set -euo pipefail

DRY_RUN=0
for arg in "$@"; do
  [ "$arg" = "--dry-run" ] && DRY_RUN=1
done

# ─── Paths ────────────────────────────────────────────────
REPO_INTEGRATION=/workspace/verl_port/openclaw_integration
REPO_DATA=/workspace/openclaw-verl-agent-rl
DATA_DIR=/workspace/verl_port/data_meeting
OUTPUT_DIR=/workspace/verl_port/ckpt_jw
AGENT_LOOP_CONFIG="${REPO_DATA}/agent_loop/config.yaml"
LOG_DIR=/tmp/jw_path_a
mkdir -p "$LOG_DIR"
TS=$(date +%Y%m%d_%H%M%S)
VERL_LOG="$LOG_DIR/verl_$TS.log"
JW_LOG_DIR="$LOG_DIR/jw_$TS"

# ─── Env ──────────────────────────────────────────────────
if [ -f /root/.pinchbench_env ]; then set -a; source /root/.pinchbench_env; set +a; fi
: "${DEEPSEEK_API_KEY:?DEEPSEEK_API_KEY required (terminal judge)}"
export PINCHBENCH_GRADE_JUDGE_MODEL="${PINCHBENCH_GRADE_JUDGE_MODEL:-deepseek-chat}"
export PINCHBENCH_GRADE_JUDGE_BASE_URL="${PINCHBENCH_GRADE_JUDGE_BASE_URL:-https://api.deepseek.com/v1}"
export PINCHBENCH_GRADE_JUDGE_API_KEY="${PINCHBENCH_GRADE_JUDGE_API_KEY:-$DEEPSEEK_API_KEY}"
export REWARD_MODE="baseline"
export PINCHBENCH_REWARD_RETURN_MODE="scalar"

MODEL_NAME="${MODEL_NAME:-Qwen3-4B}"

# ─── Pre-flight ───────────────────────────────────────────
echo "[path-a] pre-flight: clean stale jiuwenclaw/veRL processes..."
pkill -9 -f 'run_online_rl' 2>/dev/null || true
pkill -9 -f 'jiuwenclaw.gateway\|jiuwenclaw.server\|jiuwenclaw.app' 2>/dev/null || true
pkill -9 -f 'vllm.entrypoints' 2>/dev/null || true
pkill -9 -f 'launch_main_ppo' 2>/dev/null || true
sleep 4
GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
echo "[path-a] post-cleanup GPU max used: ${GPU_USED} MiB"
if [ "$GPU_USED" -gt 5000 ]; then
  echo "[path-a] WARN: GPU still has ${GPU_USED}MiB used — stale process? Continuing anyway."
fi

# ─── Hyperparams (mirror launch_meeting_openclaw_lora.sh) ─
MODEL=/root/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
N_GPUS="${N_GPUS:-2}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MICRO_BATCH="${MICRO_BATCH:-1}"
ROLLOUT_N="${ROLLOUT_N:-2}"
MAX_TURNS="${MAX_TURNS:-20}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-20000}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-12000}"
LR="${LR:-2e-6}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_TARGET_MODULES='"[q_proj,k_proj,v_proj,o_proj]"'
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.5}"
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-32000}"
SAVE_FREQ="${SAVE_FREQ:-8}"
TEST_FREQ="${TEST_FREQ:--1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-24}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-300}"
EXPERIMENT_NAME="${EXPERIMENT_NAME:-verl_jw_path_a_$TS}"

REWARD_MANAGER_PATH="${REPO_DATA}/rewards/meeting_reward_single_turn.py"

mkdir -p "$OUTPUT_DIR"

# ─── Step 2: launch veRL in background ────────────────────
if [ "$DRY_RUN" = "0" ]; then
  echo "[path-a] launching veRL → $VERL_LOG"
  env PYTHONPATH="${REPO_INTEGRATION}:${REPO_DATA}:${PYTHONPATH:-}" \
  nohup python3 -m rl.train.launch_main_ppo \
      algorithm.adv_estimator=grpo \
      algorithm.use_kl_in_reward=False \
      algorithm.norm_adv_by_std_in_grpo=False \
      data.train_files="${DATA_DIR}/train.parquet" \
      data.val_files="${DATA_DIR}/val.parquet" \
      data.train_batch_size="${BATCH_SIZE}" \
      data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
      data.max_response_length="${MAX_RESPONSE_LENGTH}" \
      data.filter_overlong_prompts=True data.truncation=left data.return_raw_chat=True \
      actor_rollout_ref.model.path="${MODEL}" \
      actor_rollout_ref.model.use_remove_padding=True \
      actor_rollout_ref.model.enable_gradient_checkpointing=True \
      +actor_rollout_ref.model.override_config.attn_implementation=sdpa \
      actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
      actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}" \
      actor_rollout_ref.model.target_modules="${LORA_TARGET_MODULES}" \
      actor_rollout_ref.actor.optim.lr="${LR}" \
      actor_rollout_ref.actor.ppo_mini_batch_size="${BATCH_SIZE}" \
      actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH}" \
      actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}" \
      actor_rollout_ref.actor.ppo_epochs=1 actor_rollout_ref.actor.use_remove_padding=True \
      actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef=0.01 \
      actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0.0 \
      actor_rollout_ref.actor.fsdp_config.param_offload=True \
      actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
      actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
      actor_rollout_ref.rollout.temperature=0.7 actor_rollout_ref.rollout.top_p=0.9 \
      actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_MEM_UTIL}" \
      actor_rollout_ref.rollout.max_model_len="${VLLM_MAX_MODEL_LEN}" \
      actor_rollout_ref.rollout.max_num_seqs="${VLLM_MAX_NUM_SEQS}" \
      actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
      actor_rollout_ref.rollout.load_format=safetensors \
      actor_rollout_ref.rollout.layered_summon=True \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
      actor_rollout_ref.rollout.multi_turn.enable=True \
      actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_TURNS}" \
      actor_rollout_ref.rollout.agent.default_agent_loop=jiuwenclaw_agent \
      actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG}" \
      actor_rollout_ref.rollout.agent.num_workers=1 \
      actor_rollout_ref.ref.fsdp_config.param_offload=True \
      actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
      +reward.custom_reward_function.path="${REWARD_MANAGER_PATH}" \
      reward.custom_reward_function.name=compute_score \
      trainer.critic_warmup=0 trainer.val_before_train=False \
      trainer.logger='["console"]' \
      +ray_kwargs.ray_init.include_dashboard=False \
      +ray_kwargs.ray_init.runtime_env.env_vars.HF_HOME=/root/hf_cache \
      +ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_CACHE=/root/hf_cache/hub \
      +ray_kwargs.ray_init.runtime_env.env_vars.JIUWENCLAW_WS_URL="'ws://127.0.0.1:611/ws'" \
      +ray_kwargs.ray_init.runtime_env.env_vars.JIUWENCLAW_TIMEOUT="'${AGENT_TIMEOUT}'" \
      +ray_kwargs.ray_init.runtime_env.env_vars.PINCHBENCH_DIR="'${REPO_DATA}'" \
      +ray_kwargs.ray_init.runtime_env.env_vars.DEEPSEEK_API_KEY="'${DEEPSEEK_API_KEY}'" \
      +ray_kwargs.ray_init.runtime_env.env_vars.PINCHBENCH_GRADE_JUDGE_MODEL="'${PINCHBENCH_GRADE_JUDGE_MODEL}'" \
      +ray_kwargs.ray_init.runtime_env.env_vars.PINCHBENCH_GRADE_JUDGE_BASE_URL="'${PINCHBENCH_GRADE_JUDGE_BASE_URL}'" \
      +ray_kwargs.ray_init.runtime_env.env_vars.PINCHBENCH_GRADE_JUDGE_API_KEY="'${PINCHBENCH_GRADE_JUDGE_API_KEY}'" \
      +ray_kwargs.ray_init.runtime_env.env_vars.REWARD_MODE=baseline \
      +ray_kwargs.ray_init.runtime_env.env_vars.PINCHBENCH_REWARD_RETURN_MODE=scalar \
      +ray_kwargs.ray_init.runtime_env.env_vars.MAX_TURNS="'${MAX_TURNS}'" \
      +ray_kwargs.ray_init.runtime_env.env_vars.AGENT_TIMEOUT="'${AGENT_TIMEOUT}'" \
      trainer.project_name=verl_port_meeting \
      trainer.experiment_name="${EXPERIMENT_NAME}" \
      trainer.n_gpus_per_node="${N_GPUS}" trainer.nnodes=1 \
      trainer.save_freq="${SAVE_FREQ}" trainer.test_freq="${TEST_FREQ}" \
      trainer.total_epochs="${TOTAL_EPOCHS}" \
      trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
      trainer.resume_mode=disable \
      trainer.default_local_dir="${OUTPUT_DIR}" \
      >"$VERL_LOG" 2>&1 &
  VERL_PID=$!
  echo "[path-a] veRL PID=$VERL_PID log=$VERL_LOG"

  # ─── Step 3: discover veRL HTTP vLLM URL from log ─────
  echo "[path-a] waiting for 'LLMServerManager:' line (up to 600s)..."
  VERL_VLLM_URL=""
  for i in $(seq 1 300); do
    if ! kill -0 "$VERL_PID" 2>/dev/null; then
      echo "[path-a] FATAL: veRL exited before LLMServerManager appeared"
      tail -50 "$VERL_LOG" >&2
      exit 3
    fi
    LINE=$(grep -m1 "LLMServerManager:" "$VERL_LOG" || true)
    if [ -n "$LINE" ]; then
      # Extract first IP:PORT from list like LLMServerManager: ['10.0.0.1:12345', ...]
      ADDR=$(echo "$LINE" | grep -oE "'[0-9.]+:[0-9]+'" | head -1 | tr -d "'")
      if [ -n "$ADDR" ]; then
        VERL_VLLM_URL="http://${ADDR}/v1"
        echo "[path-a] discovered veRL vLLM HTTP: $VERL_VLLM_URL"
        break
      fi
    fi
    sleep 2
  done
  if [ -z "$VERL_VLLM_URL" ]; then
    echo "[path-a] FATAL: timeout waiting for LLMServerManager. Last log lines:"
    tail -30 "$VERL_LOG" >&2
    kill -9 "$VERL_PID" 2>/dev/null || true
    exit 4
  fi
else
  VERL_VLLM_URL="${VERL_VLLM_URL:-http://127.0.0.1:614/v1}"
  echo "[path-a] DRY RUN — using preset VERL_VLLM_URL=$VERL_VLLM_URL"
fi

# ─── Step 4: start headless jiuwenclaw ────────────────────
echo "[path-a] launching headless jiuwenclaw with API_BASE=$VERL_VLLM_URL MODEL_NAME=$MODEL_NAME"
API_BASE="$VERL_VLLM_URL" MODEL_NAME="$MODEL_NAME" LOG_DIR="$JW_LOG_DIR" \
  bash "$(dirname "$0")/start_jw_headless.sh"

# ─── Step 5: ws-side smoke check ──────────────────────────
echo "[path-a] verifying ws://127.0.0.1:611 reachable..."
if ! (echo > /dev/tcp/127.0.0.1/611) 2>/dev/null; then
  echo "[path-a] FATAL: jiuwenclaw WS not up. Check $JW_LOG_DIR/agent_server_*.log"
  exit 5
fi
echo "[path-a] WS up. Path A wiring complete."

if [ "$DRY_RUN" = "1" ]; then
  echo "[path-a] DRY RUN done — verify with the smoke script:"
  echo "    python3 experiments/verl_port_poc/smoke_jiuwenclaw_agent_loop.py --prompt 'say hi'"
  exit 0
fi

# Foreground: tail veRL log so this script blocks while training runs
echo "[path-a] tailing veRL log (Ctrl-C exits tail; veRL continues in background as PID=$VERL_PID)"
tail -f "$VERL_LOG"
