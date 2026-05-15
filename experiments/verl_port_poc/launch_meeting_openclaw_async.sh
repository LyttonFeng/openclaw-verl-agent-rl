#!/usr/bin/env bash
# veRL FullyAsyncTrainer + OpenClaw multi-turn rollout + GRPO.
#
# Combines:
#   - launch_meeting_jiuwen_async.sh's verl async setup (separated trainer/rollout,
#     verl_patches/ pre-applied on pod, FSDP2 strategy, NCCL ckpt engine)
#   - launch_meeting_openclaw_lora.sh's openclaw runtime config (SSH-loopback to
#     local openclaw binary, agent_loop=openclaw_agent, reward_manager.py)
#
# What's different from jiuwen_async:
#   - default_agent_loop=openclaw_agent (NOT jiuwenclaw_agent)
#   - NO jiuwenclaw stack startup, NO RLOnlineRail, NO mock_trajectory_gateway
#   - NO MEMORY_ENABLED / NO sitecustomize injection
#   - OPENCLAW_HOST=localhost SSH-loopback (OpenClaw runs as subprocess via SSH self-pair)
#
# Goal: verify whether verl-async + GRPO can beat the sync 47.8% baseline on
# the same OpenClaw runtime. If async also stalls at ~47.8%, the bottleneck
# is the algorithm / task, not the framework.

set -euo pipefail

DRY_RUN=0
for arg in "$@"; do [ "$arg" = "--dry-run" ] && DRY_RUN=1; done

# ─── OpenClaw local-loopback mode ─────────────────────────
export OPENCLAW_HOST="${OPENCLAW_HOST:-localhost}"
export OPENCLAW_USER="${OPENCLAW_USER:-root}"
export OPENCLAW_PORT="${OPENCLAW_PORT:-22}"
export OPENCLAW_SSH_KEY="${OPENCLAW_SSH_KEY:-/root/.ssh/id_ed25519}"
export PINCHBENCH_ALLOW_LOCAL_OPENCLAW=1
export PINCHBENCH_SKIP_OPENCLAW_WEB_PREFLIGHT=1
export PINCHBENCH_SKIP_TRAIN_INFER_PARITY=1

# ─── Paths ────────────────────────────────────────────────
REPO_INTEGRATION=/workspace/verl_port/openclaw_integration
REPO_DATA=/workspace/openclaw-verl-agent-rl
DATA_DIR=/workspace/verl_port/data_meeting
OUTPUT_DIR=/workspace/verl_port/ckpt_openclaw_async
AGENT_LOOP_CONFIG="${REPO_INTEGRATION}/rl/agent_loop/config.yaml"
REWARD_MANAGER_PATH="${REPO_INTEGRATION}/rl/train/reward_manager.py"
LOG_DIR=/tmp/openclaw_async
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
TS=$(date +%Y%m%d_%H%M%S)
VERL_LOG="$LOG_DIR/verl_$TS.log"

export PINCHBENCH_DIR="${REPO_DATA}"
export PYTHONPATH="${REPO_INTEGRATION}:${REPO_DATA}:${PYTHONPATH:-}"
export HF_HOME="${HF_HOME:-/root/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"
export ATTN_IMPLEMENTATION=sdpa
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RAY_DISABLE_DASHBOARD=1 VLLM_NO_USAGE_STATS=1
export PINCHBENCH_LORA_ONLY_CKPT=1
export PINCHBENCH_BEST_CKPT=1
export PINCHBENCH_KEEP_LATEST_CKPT=1
export VLLM_USE_V1=1

# ─── Env: DeepSeek judge + .pinchbench_env (overrides OPENCLAW_HOST then we
# force it back to localhost) ──────────────────────────────
if [ -f /root/.pinchbench_env ]; then
  set -a
  # shellcheck disable=SC1091
  source /root/.pinchbench_env
  set +a
fi
export OPENCLAW_HOST=localhost   # force local loopback even if pinchbench_env set remote IP
: "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY (terminal judge)}"
export PINCHBENCH_GRADE_JUDGE_MODEL="${PINCHBENCH_GRADE_JUDGE_MODEL:-deepseek-chat}"
export PINCHBENCH_GRADE_JUDGE_BACKEND="${PINCHBENCH_GRADE_JUDGE_BACKEND:-api}"
export PINCHBENCH_GRADE_JUDGE_BASE_URL="${PINCHBENCH_GRADE_JUDGE_BASE_URL:-https://api.deepseek.com/v1}"
export PINCHBENCH_GRADE_JUDGE_API_KEY="${PINCHBENCH_GRADE_JUDGE_API_KEY:-${DEEPSEEK_API_KEY}}"
export REWARD_MODE="baseline"
export PINCHBENCH_REWARD_RETURN_MODE="scalar"
unset PRM_VLLM_BASE_URL PRM_API_KEY PRM_MODEL PRM_USE_CHAT_COMPLETIONS PRM_RESOLVE_MODEL PINCHBENCH_TERMINAL_REWARD_WEIGHT

# ─── Pre-flight: kill stale procs + verify ports + GPU clean ──
# Carry over hardening from jiuwen_async lessons (skip self-PID, verify ports,
# fail-fast on residual state). OpenClaw is subprocess so no port to verify,
# but we still want clean veRL + vLLM state.
echo "[async] pre-flight: kill stale procs..."
_PKILL_PATTERNS=(
  'launch_main_ppo'
  'fully_async_main'
  'vllm\.entrypoints'
  'vllm serve'
)
_MY_PID=$$
for _pid in $(pgrep -f 'launch_meeting_openclaw_async' 2>/dev/null); do
  if [ "$_pid" != "$_MY_PID" ] && [ "$_pid" != "$PPID" ]; then
    kill -9 "$_pid" 2>/dev/null || true
  fi
done
for _pat in "${_PKILL_PATTERNS[@]}"; do
  pkill -9 -f "$_pat" 2>/dev/null || true
done
sleep 4
for _pid in $(pgrep -f 'VLLM::EngineCore\|VLLM::Worker' 2>/dev/null); do
  kill -9 "$_pid" 2>/dev/null || true
done
sleep 2

GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
echo "[async] post-cleanup GPU max used: ${GPU_USED} MiB"
if [ "$GPU_USED" -gt 5000 ]; then
  echo "[async] FATAL: GPU still has ${GPU_USED}MiB after cleanup — manual kill needed"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
  exit 8
fi

# ─── Hyperparams ──────────────────────────────────────────
MODEL=/root/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
N_GPUS_TRAINER="${N_GPUS_TRAINER:-1}"   # FSDP actor + ref on GPU 0
N_GPUS_ROLLOUT="${N_GPUS_ROLLOUT:-1}"   # vLLM on GPU 1
BATCH_SIZE="${BATCH_SIZE:-2}"            # match openclaw sync baseline group=2
MICRO_BATCH="${MICRO_BATCH:-1}"
ROLLOUT_N="${ROLLOUT_N:-2}"
MAX_TURNS="${MAX_TURNS:-20}"             # OpenClaw config in sync baseline

LORA_RANK="${LORA_RANK:-16}"             # match main R4' baseline (was 32 in jiuwen)
LORA_ALPHA="${LORA_ALPHA:-32}"
LORA_TARGET_MODULES='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'
LR="${LR:-2e-6}"

# OpenClaw multi-turn growth — prompts grow much smaller than jiuwenclaw
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-48000}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-12000}"

# vLLM rollout (60% mem on GPU 1; GPU 0 has FSDP actor + ref)
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-65536}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.6}"

ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-65536}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.01}"

# Async-specific
STALENESS_THRESHOLD="${STALENESS_THRESHOLD:-0.3}"
TRIGGER_PARAM_SYNC_STEP="${TRIGGER_PARAM_SYNC_STEP:-2}"
REQUIRE_BATCHES="${REQUIRE_BATCHES:-4}"

SAVE_FREQ="${SAVE_FREQ:-1}"
TEST_FREQ="${TEST_FREQ:--1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-2}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-24}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-300}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-openclaw_async_$TS}"

# ─── Preflights ──────────────────────────────────────────
# OpenClaw preflight (SSH loopback to local openclaw binary)
python3 -c "
import subprocess
r = subprocess.run(['ssh','-o','StrictHostKeyChecking=no','-o','ConnectTimeout=10','-i','${OPENCLAW_SSH_KEY}','-p','${OPENCLAW_PORT}','${OPENCLAW_USER}@${OPENCLAW_HOST}','command -v openclaw && openclaw --version'], capture_output=True, text=True)
assert r.returncode == 0, f'OpenClaw preflight FAILED: {r.stderr}'
print('[preflight] OpenClaw OK:', r.stdout.strip())
"

# DeepSeek judge preflight
python3 -c "
import sys
sys.path.insert(0, '${REPO_DATA}/scripts')
from lib_grading import preflight_judge_connection
preflight_judge_connection(judge_model='deepseek-chat', judge_backend='api', judge_base_url='https://api.deepseek.com/v1', judge_api_key='${DEEPSEEK_API_KEY}')
print('[preflight] DeepSeek judge OK')
"

echo "═══════════════════════════════════════════════════════════════════"
echo "  veRL FullyAsync + OpenClaw multi-turn — GRPO"
echo "  Model:        ${MODEL}    LoRA r=${LORA_RANK} α=${LORA_ALPHA}"
echo "  GRPO:         adv=grpo  rollout.n=${ROLLOUT_N}  REQUIRE_BATCHES=${REQUIRE_BATCHES}"
echo "  Reward:       terminal-only DeepSeek-chat"
echo "  GPUs:         trainer=${N_GPUS_TRAINER}  rollout=${N_GPUS_ROLLOUT}"
echo "  Batch:        ${BATCH_SIZE}    LR: ${LR}    KL coef: ${KL_LOSS_COEF}"
echo "  Steps cap:    ${TOTAL_TRAINING_STEPS}    save@every-param-sync (${SAVE_FREQ})"
echo "  Param sync:   every ${TRIGGER_PARAM_SYNC_STEP} trainer step"
echo "  OpenClaw:     ${OPENCLAW_HOST} (subprocess via SSH)    MAX_TURNS=${MAX_TURNS}"
echo "  Output:       ${OUTPUT_DIR}    log=${VERL_LOG}"
echo "═══════════════════════════════════════════════════════════════════"

if [ "$DRY_RUN" = "1" ]; then exit 0; fi

echo "[async] launching veRL FullyAsyncTrainer → $VERL_LOG"
cd "$REPO_INTEGRATION"
env PYTHONPATH="${REPO_INTEGRATION}:${REPO_DATA}:${PYTHONPATH:-}" VLLM_USE_V1=1 \
nohup python3 -m verl.experimental.fully_async_policy.fully_async_main \
    --config-name fully_async_ppo_trainer \
    hydra.run.dir="$LOG_DIR/hydra_$TS" \
    hydra.output_subdir="." \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.norm_adv_by_std_in_grpo=False \
    algorithm.rollout_correction.bypass_mode=False \
    data.train_files="${DATA_DIR}/train_full23.parquet" \
    data.val_files="${DATA_DIR}/val_5test.parquet" \
    data.train_batch_size=0 \
    data.gen_batch_size=1 \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True data.truncation=left data.return_raw_chat=True \
    async_training.staleness_threshold="${STALENESS_THRESHOLD}" \
    async_training.trigger_parameter_sync_step="${TRIGGER_PARAM_SYNC_STEP}" \
    async_training.require_batches="${REQUIRE_BATCHES}" \
    async_training.partial_rollout=True \
    actor_rollout_ref.hybrid_engine=False \
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
    actor_rollout_ref.actor.use_kl_loss=True actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF}" \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.use_rollout_log_probs=True \
    actor_rollout_ref.actor.strategy=fsdp2 \
    actor_rollout_ref.ref.strategy=fsdp2 \
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.name=vllm actor_rollout_ref.rollout.mode=async \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.temperature=0.7 actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.max_model_len="${VLLM_MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_seqs="${VLLM_MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.enable_auto_tool_choice=true \
    +actor_rollout_ref.rollout.engine_kwargs.vllm.tool_call_parser=hermes \
    actor_rollout_ref.rollout.calculate_log_probs=True \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_TURNS}" \
    actor_rollout_ref.rollout.agent.default_agent_loop=openclaw_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG}" \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
    rollout.nnodes=1 \
    rollout.n_gpus_per_node="${N_GPUS_ROLLOUT}" \
    rollout.total_rollout_steps=1536 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    +reward.custom_reward_function.path="${REWARD_MANAGER_PATH}" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 trainer.val_before_train=False \
    trainer.logger='["console"]' \
    +ray_kwargs.ray_init.include_dashboard=False \
    +ray_kwargs.ray_init.num_gpus=2 \
    +ray_kwargs.ray_init.runtime_env.env_vars.HF_HOME=/root/hf_cache \
    +ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_CACHE=/root/hf_cache/hub \
    +ray_kwargs.ray_init.runtime_env.env_vars.TRANSFORMERS_CACHE=/root/hf_cache/hub \
    +ray_kwargs.ray_init.runtime_env.env_vars.OPENCLAW_HOST="'${OPENCLAW_HOST}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.OPENCLAW_USER="'${OPENCLAW_USER}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.OPENCLAW_PORT="'${OPENCLAW_PORT}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.OPENCLAW_SSH_KEY="'${OPENCLAW_SSH_KEY}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PINCHBENCH_ALLOW_LOCAL_OPENCLAW="'1'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PINCHBENCH_SKIP_OPENCLAW_WEB_PREFLIGHT="'1'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PINCHBENCH_SKIP_TRAIN_INFER_PARITY="'1'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PINCHBENCH_DIR="'${REPO_DATA}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.DEEPSEEK_API_KEY="'${DEEPSEEK_API_KEY}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PINCHBENCH_GRADE_JUDGE_MODEL="'${PINCHBENCH_GRADE_JUDGE_MODEL}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PINCHBENCH_GRADE_JUDGE_BASE_URL="'${PINCHBENCH_GRADE_JUDGE_BASE_URL}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PINCHBENCH_GRADE_JUDGE_API_KEY="'${PINCHBENCH_GRADE_JUDGE_API_KEY}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.REWARD_MODE=baseline \
    +ray_kwargs.ray_init.runtime_env.env_vars.PINCHBENCH_REWARD_RETURN_MODE=scalar \
    +ray_kwargs.ray_init.runtime_env.env_vars.MAX_TURNS="'${MAX_TURNS}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.AGENT_TIMEOUT="'${AGENT_TIMEOUT}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1="'1'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PINCHBENCH_LORA_ONLY_CKPT="'1'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PINCHBENCH_BEST_CKPT="'1'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PINCHBENCH_KEEP_LATEST_CKPT="'1'" \
    trainer.project_name=verl_port_meeting \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS_TRAINER}" trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ}" trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.resume_mode=auto \
    trainer.default_local_dir="${OUTPUT_DIR}" \
    >"$VERL_LOG" 2>&1 &
VERL_PID=$!
echo "[async] veRL PID=$VERL_PID log=$VERL_LOG"

echo "[async] waiting for vLLM HTTP server (up to 900s)..."
for i in $(seq 1 450); do
  if ! kill -0 "$VERL_PID" 2>/dev/null; then
    echo "[async] FATAL: veRL exited before vLLM HTTP came up"
    tail -50 "$VERL_LOG" >&2
    exit 3
  fi
  LINE=$(grep -m1 "LLMServerManager:" "$VERL_LOG" 2>/dev/null || true)
  if [ -n "$LINE" ]; then
    echo "[async] vLLM HTTP up: $LINE"
    break
  fi
  sleep 2
done

echo "[async] Async wiring complete."
echo "[async] tailing veRL log (Ctrl-C exits tail; veRL continues as PID=$VERL_PID)"
tail -f "$VERL_LOG"
