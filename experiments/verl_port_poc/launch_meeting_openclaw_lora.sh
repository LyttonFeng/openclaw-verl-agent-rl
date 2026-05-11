#!/usr/bin/env bash
# veRL + OpenClaw multi-turn rollout + GRPO + terminal-only DeepSeek judge.
#
# Goal: reproduce openclaw-verl-agent-rl main-branch vanilla GRPO baseline
#       (no PRM, no quality filter, no terminal-weight) on the veRL framework
#       with OpenClaw as the rollout agent.
#
# Aligned with the main repo's baseline path:
#   - algorithm.adv_estimator=grpo
#   - REWARD_MODE=baseline (only terminal reward in pinchbench naming)
#   - PINCHBENCH_REWARD_RETURN_MODE=scalar → custom_reward_function.compute_score
#   - Terminal judge = DeepSeek-chat (same as scripts/lib_grading.py default)
#
# Pod-side prerequisites:
#   /workspace/verl_port/openclaw_integration/  pinchbench-skill rl/* + agent_loop/
#   /workspace/verl_port/data_meeting/{train,val}.parquet
#   /workspace/openclaw-verl-agent-rl/pinchbench_tasks/meeting_analysis/
#   /usr/local/bin/openclaw  (2026.4.5)
#   /root/.ssh/id_ed25519 self-pair
#   /root/.pinchbench_env exports DEEPSEEK_API_KEY

set -euo pipefail

# ─── Local OpenClaw mode ──────────────────────────────────
export OPENCLAW_HOST="${OPENCLAW_HOST:-localhost}"
export OPENCLAW_USER="${OPENCLAW_USER:-root}"
export OPENCLAW_PORT="${OPENCLAW_PORT:-22}"
export OPENCLAW_SSH_KEY="${OPENCLAW_SSH_KEY:-/root/.ssh/id_ed25519}"
export PINCHBENCH_ALLOW_LOCAL_OPENCLAW=1
export PINCHBENCH_SKIP_OPENCLAW_WEB_PREFLIGHT=1
export PINCHBENCH_SKIP_TRAIN_INFER_PARITY=1

# ─── DeepSeek judge keys ──────────────────────────────────
if [ -f /root/.pinchbench_env ]; then
  set -a
  # shellcheck disable=SC1091
  source /root/.pinchbench_env
  set +a
fi
# pinchbench_env on this pod hardcodes OPENCLAW_HOST to an ECS IP — force
# localhost since we want pod-local OpenClaw.
export OPENCLAW_HOST=localhost
: "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY (terminal judge)}"
export PINCHBENCH_GRADE_JUDGE_MODEL="${PINCHBENCH_GRADE_JUDGE_MODEL:-deepseek-chat}"
export PINCHBENCH_GRADE_JUDGE_BACKEND="${PINCHBENCH_GRADE_JUDGE_BACKEND:-api}"
export PINCHBENCH_GRADE_JUDGE_BASE_URL="${PINCHBENCH_GRADE_JUDGE_BASE_URL:-https://api.deepseek.com/v1}"
export PINCHBENCH_GRADE_JUDGE_API_KEY="${PINCHBENCH_GRADE_JUDGE_API_KEY:-${DEEPSEEK_API_KEY}}"

# ─── Reward: vanilla terminal-only ────────────────────────
export REWARD_MODE="baseline"                   # only terminal, no process reward
export PINCHBENCH_REWARD_RETURN_MODE="scalar"   # scalar reward → custom_reward_function.compute_score
# Remove all PRM / oracle-judge knobs (we want pure terminal).
unset PRM_VLLM_BASE_URL PRM_API_KEY PRM_MODEL PRM_USE_CHAT_COMPLETIONS PRM_RESOLVE_MODEL PINCHBENCH_TERMINAL_REWARD_WEIGHT

# ─── Paths ────────────────────────────────────────────────
REPO_INTEGRATION=/workspace/verl_port/openclaw_integration
REPO_DATA=/workspace/openclaw-verl-agent-rl
DATA_DIR=/workspace/verl_port/data_meeting
OUTPUT_DIR=/workspace/verl_port/ckpt_openclaw
AGENT_LOOP_CONFIG="${REPO_INTEGRATION}/rl/agent_loop/config.yaml"
REWARD_MANAGER_PATH="${REPO_INTEGRATION}/rl/train/reward_manager.py"

export PINCHBENCH_DIR="${REPO_DATA}"
export PYTHONPATH="${REPO_INTEGRATION}:${REPO_DATA}:${PYTHONPATH:-}"

# HF cache
export HF_HOME="${HF_HOME:-/root/hf_cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${HF_HOME}/hub}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-${HF_HUB_CACHE}}"

export ATTN_IMPLEMENTATION=sdpa
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export RAY_DISABLE_DASHBOARD=1 VLLM_NO_USAGE_STATS=1

# LoRA-only ckpt save (avoid MooseFS 8 GB write truncation)
export PINCHBENCH_LORA_ONLY_CKPT=1
export PINCHBENCH_BEST_CKPT=1
export PINCHBENCH_KEEP_LATEST_CKPT=1

# ─── Hyperparams ──────────────────────────────────────────
MODEL="${VERL_MODEL:-Qwen/Qwen3-4B}"
N_GPUS=2
BATCH_SIZE=2
MICRO_BATCH=1
LORA_RANK=16            # match main-repo run_meeting_grpo_prm_round.sh default
LORA_ALPHA=32
LORA_TARGET_MODULES='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'
LR=2e-6                 # match main-repo default

# GRPO: match main-branch convention — group=2, std normalization OFF
# (with group size 2, sample std is degenerate; main branch trains with std fixed
# at 1, i.e. advantage = reward - group_mean. Set norm_adv_by_std_in_grpo=False
# in the algorithm config below for this behavior.)
ROLLOUT_N=2

# OpenClaw multi-turn limits — generous since A100 80GB
export MAX_TURNS=20
export OPENCLAW_MODEL_REASONING=0
export AGENT_TIMEOUT=300
export PINCHBENCH_AGENT_MAX_PROMPT_TOKENS=24000

# vLLM rollout (A100-80GB) — 64K context via rope=2 YARN, matches main repo
# run_meeting_grpo_prm_round.sh which uses MAX_SEQ_LEN=65536 + ROPE_FACTOR=2.0
export VLLM_GPU_MEM_UTIL=0.45
export VLLM_MAX_MODEL_LEN=65536
export VLLM_MAX_NUM_SEQS=16   # halve from 32 to keep KV cache from blowing up

ACTOR_PARAM_OFFLOAD=False
ACTOR_OPTIMIZER_OFFLOAD=False
REF_PARAM_OFFLOAD=True
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=65536

MAX_PROMPT_LENGTH=48000   # accommodate multi-turn OpenClaw growth (transcript read + reasoning)
MAX_RESPONSE_LENGTH=12000

# Cadence: 28 tasks / batch 2 = 14 steps per epoch
TOTAL_EPOCHS=2
TOTAL_TRAINING_STEPS=24
SAVE_FREQ=8
TEST_FREQ=8

RUN_STAMP="$(date +%H%M)"
EXPERIMENT_NAME="meeting_openclaw_grpo_${RUN_STAMP}"

mkdir -p "${OUTPUT_DIR}"

echo "═══════════════════════════════════════════════════════════════════"
echo "  veRL + OpenClaw multi-turn — vanilla GRPO baseline"
echo "  Model:        ${MODEL}    LoRA r=${LORA_RANK} α=${LORA_ALPHA}"
echo "  GRPO:         adv_estimator=grpo  rollout.n=${ROLLOUT_N}  norm_adv_by_std=False (std≡1)"
echo "  Reward:       terminal-only (DeepSeek-chat)  no PRM / no quality filter"
echo "  GPUs:         ${N_GPUS}    Batch: ${BATCH_SIZE}    LR: ${LR}"
echo "  Tasks:        28 meeting_analysis (round-robin parquet)"
echo "  Steps cap:    ${TOTAL_TRAINING_STEPS}    save@${SAVE_FREQ}    test@${TEST_FREQ}"
echo "  OpenClaw:     ${OPENCLAW_HOST} (subprocess)    MAX_TURNS=${MAX_TURNS}"
echo "  Output:       ${OUTPUT_DIR}"
echo "═══════════════════════════════════════════════════════════════════"

# OpenClaw preflight (SSH to localhost)
python3 -c "
import subprocess
r = subprocess.run(['ssh','-o','StrictHostKeyChecking=no','-o','ConnectTimeout=10','-i','${OPENCLAW_SSH_KEY}','-p','${OPENCLAW_PORT}','${OPENCLAW_USER}@${OPENCLAW_HOST}','command -v openclaw && openclaw --version'], capture_output=True, text=True)
assert r.returncode == 0, f'OpenClaw preflight FAILED: {r.stderr}'
print('OpenClaw preflight OK:', r.stdout.strip())
"

# DeepSeek judge preflight
python3 -c "
import sys
sys.path.insert(0, '${REPO_DATA}/scripts')
from lib_grading import preflight_judge_connection
preflight_judge_connection(judge_model='deepseek-chat', judge_backend='api', judge_base_url='https://api.deepseek.com/v1', judge_api_key='${DEEPSEEK_API_KEY}')
"

cd "${REPO_INTEGRATION}"
echo "[debug] cwd=$(pwd)"
echo "[debug] PYTHONPATH=${PYTHONPATH}"
echo "[debug] which python3: $(which python3)"

env PYTHONPATH="${REPO_INTEGRATION}:${REPO_DATA}:${PYTHONPATH:-}" python3 -m rl.train.launch_main_ppo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
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
    +actor_rollout_ref.model.override_config.max_position_embeddings=65536 \
    '+actor_rollout_ref.model.override_config.rope_scaling={type:yarn,factor:2.0,original_max_position_embeddings:32768}' \
    actor_rollout_ref.model.lora_rank="${LORA_RANK}" \
    actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}" \
    actor_rollout_ref.model.target_modules="${LORA_TARGET_MODULES}" \
    actor_rollout_ref.actor.optim.lr="${LR}" \
    actor_rollout_ref.actor.ppo_mini_batch_size="${BATCH_SIZE}" \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH}" \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU}" \
    actor_rollout_ref.actor.ppo_epochs=1 \
    actor_rollout_ref.actor.use_remove_padding=True \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.01 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0.0 \
    actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}" \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}" \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.n="${ROLLOUT_N}" \
    actor_rollout_ref.rollout.temperature=0.7 \
    actor_rollout_ref.rollout.top_p=0.9 \
    actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_MEM_UTIL}" \
    actor_rollout_ref.rollout.max_model_len="${VLLM_MAX_MODEL_LEN}" \
    actor_rollout_ref.rollout.max_num_seqs="${VLLM_MAX_NUM_SEQS}" \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.multi_turn.enable=True \
    actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_TURNS}" \
    actor_rollout_ref.rollout.agent.default_agent_loop=openclaw_agent \
    actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG}" \
    actor_rollout_ref.rollout.agent.num_workers=1 \
    actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD}" \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
    +reward.custom_reward_function.path="${REWARD_MANAGER_PATH}" \
    reward.custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    trainer.val_before_train=False \
    trainer.logger='["console"]' \
    +ray_kwargs.ray_init.include_dashboard=False \
    +ray_kwargs.ray_init.runtime_env.env_vars.HF_HOME=/root/hf_cache \
    +ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_CACHE=/root/hf_cache/hub \
    +ray_kwargs.ray_init.runtime_env.env_vars.TRANSFORMERS_CACHE=/root/hf_cache/hub \
    +ray_kwargs.ray_init.runtime_env.env_vars.OPENCLAW_HOST="'${OPENCLAW_HOST}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.OPENCLAW_USER="'${OPENCLAW_USER}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.OPENCLAW_PORT="'${OPENCLAW_PORT}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.OPENCLAW_SSH_KEY="'${OPENCLAW_SSH_KEY}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.OPENCLAW_MODEL_REASONING="'${OPENCLAW_MODEL_REASONING}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PINCHBENCH_DIR="'${PINCHBENCH_DIR}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.DEEPSEEK_API_KEY="'${DEEPSEEK_API_KEY}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PINCHBENCH_GRADE_JUDGE_MODEL="'${PINCHBENCH_GRADE_JUDGE_MODEL}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PINCHBENCH_GRADE_JUDGE_BASE_URL="'${PINCHBENCH_GRADE_JUDGE_BASE_URL}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PINCHBENCH_GRADE_JUDGE_API_KEY="'${PINCHBENCH_GRADE_JUDGE_API_KEY}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.REWARD_MODE=baseline \
    +ray_kwargs.ray_init.runtime_env.env_vars.PINCHBENCH_REWARD_RETURN_MODE=scalar \
    +ray_kwargs.ray_init.runtime_env.env_vars.MAX_TURNS="'${MAX_TURNS}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.PINCHBENCH_ALLOW_LOCAL_OPENCLAW="'1'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.AGENT_TIMEOUT="'${AGENT_TIMEOUT}'" \
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
