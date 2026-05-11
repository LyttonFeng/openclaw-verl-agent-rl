#!/usr/bin/env bash
# v3: veRL Online RL (REINFORCE++) + LoRA + OpenClaw multi-turn agent loop
# on meeting_analysis tasks. Adapted from pinchbench-skill rl/train/run_reinforce_lora.sh
# for our pod setup (local OpenClaw, no remote ECS).
#
# WHY THIS EXISTS:
#   Single-turn inline-transcript training (v2) optimizes the wrong distribution.
#   Real bench uses OpenClaw multi-turn (read file → think → write file). This
#   script wires veRL rollout to an OpenClawAgentLoop subclass that drives the
#   pod's local OpenClaw runtime.
#
# Pod-side prerequisites (all already set up):
#   /workspace/verl_port/openclaw_integration/  — pinchbench-skill rl/* + agent_loop/ + train/
#   /workspace/verl_port/data_meeting/train.parquet val.parquet
#   /workspace/openclaw-verl-agent-rl/pinchbench_tasks/meeting_analysis/*.md
#   /workspace/openclaw-verl-agent-rl/scripts/lib_tasks.py lib_grading.py
#   /usr/local/bin/openclaw (2026.4.5)
#   /root/.ssh/id_ed25519 self-pair (for OpenClaw preflight SSH)
#   /root/.pinchbench_env (DASHSCOPE_API_KEY for qwen-plus judge)
#
# DON'T LAUNCH AUTOMATICALLY. This is the adapted recipe; review before running.
#
# Run with:
#   bash /workspace/verl_port/launch_meeting_openclaw_lora.sh

set -euo pipefail

# ─── Local OpenClaw mode ──────────────────────────────────
export OPENCLAW_HOST="${OPENCLAW_HOST:-localhost}"
export OPENCLAW_USER="${OPENCLAW_USER:-root}"
export OPENCLAW_PORT="${OPENCLAW_PORT:-22}"
export OPENCLAW_SSH_KEY="${OPENCLAW_SSH_KEY:-/root/.ssh/id_ed25519}"
export PINCHBENCH_ALLOW_LOCAL_OPENCLAW=1
# Skip web-skill preflight — meeting tasks don't need web_search/web_fetch
export PINCHBENCH_SKIP_OPENCLAW_WEB_PREFLIGHT=1
# Skip train/infer parity — that check is for pinchbench-skill's 8 RL tasks, not meeting
export PINCHBENCH_SKIP_TRAIN_INFER_PARITY=1

# ─── DashScope qwen-plus grading judge ────────────────────
if [ -f /root/.pinchbench_env ]; then
  set -a
  # shellcheck disable=SC1091
  source /root/.pinchbench_env
  set +a
fi
: "${DASHSCOPE_API_KEY:?Set DASHSCOPE_API_KEY (in /root/.pinchbench_env) for qwen-plus judge}"
export PINCHBENCH_GRADE_JUDGE_API_KEY="${DASHSCOPE_API_KEY}"

# ─── Paths ────────────────────────────────────────────────
REPO_INTEGRATION=/workspace/verl_port/openclaw_integration
REPO_DATA=/workspace/openclaw-verl-agent-rl
DATA_DIR=/workspace/verl_port/data_meeting
OUTPUT_DIR=/workspace/verl_port/ckpt_openclaw
AGENT_LOOP_CONFIG="${REPO_INTEGRATION}/rl/agent_loop/config.yaml"
REWARD_MANAGER_PATH="${REPO_INTEGRATION}/rl/train/reward_manager.py"

export PINCHBENCH_DIR="${REPO_DATA}"
# PYTHONPATH order matters: integration code first (overrides anything), then repo (scripts, tasks)
export PYTHONPATH="${REPO_INTEGRATION}:${REPO_DATA}:${PYTHONPATH:-}"

# HF cache on /workspace
export HF_HOME="${HF_HOME:-/workspace/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"

# Use sdpa attention (no flash_attn build hassle for now)
export ATTN_IMPLEMENTATION=sdpa

# Threading sanity
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RAY_DISABLE_DASHBOARD=1 VLLM_NO_USAGE_STATS=1

# ─── LoRA-only checkpoint save (avoid MooseFS 8 GB write truncation) ─
export PINCHBENCH_LORA_ONLY_CKPT=1
export PINCHBENCH_BEST_CKPT=1
export PINCHBENCH_KEEP_LATEST_CKPT=1

# ─── Hyperparams ──────────────────────────────────────────
MODEL="${VERL_MODEL:-Qwen/Qwen3-4B}"
N_GPUS=2
BATCH_SIZE=2
MICRO_BATCH=1
LORA_RANK=32
LORA_ALPHA=64
LORA_TARGET_MODULES='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'
LR=2e-5

# Reward: oracle-judge (qwen-plus via DashScope) + terminal-weighted
export REWARD_MODE=oracle-judge
export PINCHBENCH_REWARD_RETURN_MODE=turn
export PINCHBENCH_TERMINAL_REWARD_WEIGHT=0.7   # audit hypothesis: weight up from 0.3
export PINCHBENCH_TASK_EMA_INIT=0.1
export MAX_TURNS=8
export OPENCLAW_MODEL_REASONING=0
export AGENT_TIMEOUT=240
export PINCHBENCH_AGENT_MAX_PROMPT_TOKENS=20000

# PRM (process reward): oracle-judge mode uses external qwen-plus
export PRM_VLLM_BASE_URL="https://dashscope.aliyuncs.com/compatible-mode/v1"
export PRM_API_KEY="${DASHSCOPE_API_KEY}"
export PRM_MODEL="qwen-plus"
export PRM_USE_CHAT_COMPLETIONS=1
export PRM_RESOLVE_MODEL=0

# vLLM rollout memory (A100-80GB so we can be generous)
export VLLM_GPU_MEM_UTIL=0.40
export VLLM_MAX_MODEL_LEN=32768
export VLLM_MAX_NUM_SEQS=32

# FSDP offload — meeting prompts up to 24K tokens, keep actor on GPU
ACTOR_PARAM_OFFLOAD=False
ACTOR_OPTIMIZER_OFFLOAD=False
REF_PARAM_OFFLOAD=True
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=32768

# Sequence caps
MAX_PROMPT_LENGTH=20000
MAX_RESPONSE_LENGTH=12000

# Training cadence
TOTAL_EPOCHS=2
TOTAL_TRAINING_STEPS=24    # 28 tasks / batch 2 = 14 steps/epoch, ~2 epochs → cap at 24
SAVE_FREQ=8
TEST_FREQ=8

RUN_STAMP="$(date +%H%M)"
EXPERIMENT_NAME="meeting_openclaw_lora_${RUN_STAMP}"

mkdir -p "${OUTPUT_DIR}"

echo "=============================="
echo "  veRL + OpenClaw multi-turn RL"
echo "  Model: ${MODEL}  LoRA r=${LORA_RANK}"
echo "  GPUs: ${N_GPUS}  Batch: ${BATCH_SIZE}"
echo "  Tasks: 28 meeting_analysis (round-robin)"
echo "  Total steps: ${TOTAL_TRAINING_STEPS} (capped)  save@${SAVE_FREQ} test@${TEST_FREQ}"
echo "  OpenClaw: ${OPENCLAW_HOST} (local subprocess)  max_turns=${MAX_TURNS}"
echo "  Reward: ${REWARD_MODE} (qwen-plus judge)  terminal_weight=${PINCHBENCH_TERMINAL_REWARD_WEIGHT}"
echo "  Output: ${OUTPUT_DIR}"
echo "=============================="

# OpenClaw preflight (will SSH to localhost)
python3 -c "
import subprocess
r = subprocess.run(['ssh','-o','StrictHostKeyChecking=no','-o','ConnectTimeout=10','-i','${OPENCLAW_SSH_KEY}','-p','${OPENCLAW_PORT}','${OPENCLAW_USER}@${OPENCLAW_HOST}','command -v openclaw && openclaw --version'], capture_output=True, text=True)
assert r.returncode == 0, f'OpenClaw preflight FAILED: {r.stderr}'
print('OpenClaw preflight OK:', r.stdout.strip())
"

# DashScope qwen-plus preflight
python3 -c "
import sys
sys.path.insert(0, '${REPO_DATA}/scripts')
from lib_grading import preflight_judge_connection
preflight_judge_connection(judge_model='qwen-plus', judge_backend='api', judge_base_url='https://dashscope.aliyuncs.com/compatible-mode/v1', judge_api_key='${DASHSCOPE_API_KEY}')
"

cd /root/verl

python3 -m rl.train.launch_main_ppo \
    algorithm.adv_estimator=reinforce_plus_plus \
    algorithm.gamma=0.0 \
    algorithm.use_kl_in_reward=True \
    algorithm.norm_adv_by_std_in_grpo=False \
    data.train_files="${DATA_DIR}/train.parquet" \
    data.val_files="${DATA_DIR}/val.parquet" \
    data.train_batch_size="${BATCH_SIZE}" \
    data.max_prompt_length="${MAX_PROMPT_LENGTH}" \
    data.max_response_length="${MAX_RESPONSE_LENGTH}" \
    data.filter_overlong_prompts=True \
    data.truncation=left \
    data.return_raw_chat=True \
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
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.shuffle=False \
    actor_rollout_ref.actor.use_remove_padding=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.01 \
    actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n=1 \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.max_model_len="${VLLM_MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_seqs="${VLLM_MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns=30 \
    actor_rollout_ref.rollout.agent.default_agent_loop=openclaw_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG}" \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD}" \
    reward.reward_manager.source=importlib \
    reward.reward_manager.module.path="${REWARD_MANAGER_PATH}" \
    reward.reward_manager.name=PinchBenchRewardManager \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console"]' \
    +ray_kwargs.ray_init.include_dashboard=False \
    +ray_kwargs.ray_init.runtime_env.env_vars.OPENCLAW_HOST="'${OPENCLAW_HOST}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.OPENCLAW_USER="'${OPENCLAW_USER}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.OPENCLAW_PORT="'${OPENCLAW_PORT}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.OPENCLAW_SSH_KEY="'${OPENCLAW_SSH_KEY}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.OPENCLAW_MODEL_REASONING="'${OPENCLAW_MODEL_REASONING}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PINCHBENCH_DIR="'${PINCHBENCH_DIR}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.DASHSCOPE_API_KEY="'${DASHSCOPE_API_KEY}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PINCHBENCH_TERMINAL_REWARD_WEIGHT="'${PINCHBENCH_TERMINAL_REWARD_WEIGHT}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PINCHBENCH_REWARD_RETURN_MODE="'${PINCHBENCH_REWARD_RETURN_MODE}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PINCHBENCH_GRADE_JUDGE_API_KEY="'${PINCHBENCH_GRADE_JUDGE_API_KEY}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PINCHBENCH_GRADE_JUDGE_MODEL=qwen-plus \
    +ray_kwargs.ray_init.runtime_env.env_vars.PINCHBENCH_GRADE_JUDGE_BASE_URL="'https://dashscope.aliyuncs.com/compatible-mode/v1'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PRM_VLLM_BASE_URL="'${PRM_VLLM_BASE_URL}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PRM_API_KEY="'${PRM_API_KEY}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PRM_MODEL="'${PRM_MODEL}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PRM_USE_CHAT_COMPLETIONS="'1'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.REWARD_MODE=oracle-judge \
    +ray_kwargs.ray_init.runtime_env.env_vars.MAX_TURNS="'${MAX_TURNS}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PINCHBENCH_ALLOW_LOCAL_OPENCLAW="'1'" \
    trainer.project_name=verl_port_meeting \
    trainer.experiment_name="${EXPERIMENT_NAME}" \
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.resume_mode=disable \
    trainer.default_local_dir="${OUTPUT_DIR}"
