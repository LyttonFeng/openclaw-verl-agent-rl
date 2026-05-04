#!/usr/bin/env bash
# veRL REINFORCE++ + LoRA + OpenClaw agent loop for task16.

set -euo pipefail

DRY_RUN=0
if [ "${1:-}" = "--dry-run" ]; then
  DRY_RUN=1
fi

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"

DATA_DIR="${PINCHBENCH_DATA_DIR_OVERRIDE:-${DATA_DIR:-${REPO_ROOT}/data/task16_prompts}}"
TRAIN_FILE="${PINCHBENCH_TRAIN_FILE_OVERRIDE:-${DATA_DIR}/train.parquet}"
VAL_FILE="${PINCHBENCH_VAL_FILE_OVERRIDE:-${DATA_DIR}/val.parquet}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/checkpoints/reinforce_lora${RUN_VERSION:+_${RUN_VERSION}}}"
AGENT_LOOP_CONFIG="${REPO_ROOT}/agent_loop/config.yaml"
REWARD_MANAGER_PATH="${REPO_ROOT}/rl/train/reward_manager.py"

export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-0}"
export HUGGINGFACE_HUB_ENABLE_HF_TRANSFER="${HUGGINGFACE_HUB_ENABLE_HF_TRANSFER:-0}"
export ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-sdpa}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export NUMEXPR_NUM_THREADS="${NUMEXPR_NUM_THREADS:-1}"
export RAY_DISABLE_DASHBOARD="${RAY_DISABLE_DASHBOARD:-1}"
export RAY_raylet_start_wait_time_s="${RAY_raylet_start_wait_time_s:-60}"
export VLLM_NO_USAGE_STATS="${VLLM_NO_USAGE_STATS:-1}"
export PINCHBENCH_DIR="${REPO_ROOT}"
export OPENCLAW_REMOTE_ACTIVATE_CMD="${OPENCLAW_REMOTE_ACTIVATE_CMD:-}"
export AGENT_TIMEOUT="${AGENT_TIMEOUT:-240}"
export PINCHBENCH_GRADE_JUDGE_MODEL="${PINCHBENCH_GRADE_JUDGE_MODEL:-qwen-plus}"
export PINCHBENCH_GRADE_JUDGE_BACKEND="${PINCHBENCH_GRADE_JUDGE_BACKEND:-api}"
export PINCHBENCH_GRADE_JUDGE_BASE_URL="${PINCHBENCH_GRADE_JUDGE_BASE_URL:-https://dashscope.aliyuncs.com/compatible-mode/v1}"
export PINCHBENCH_GRADE_JUDGE_API_KEY="${PINCHBENCH_GRADE_JUDGE_API_KEY:-${DASHSCOPE_API_KEY:-${JUDGE_API_KEY:-}}}"

MODEL="${VERL_MODEL:-Qwen/Qwen3-1.7B}"
N_GPUS="${VERL_N_GPUS:-1}"
BATCH_SIZE="${BATCH_SIZE:-1}"
VAL_BATCH_SIZE="${VAL_BATCH_SIZE:-1}"
MICRO_BATCH="${MICRO_BATCH:-1}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_TARGET_MODULES="${LORA_TARGET_MODULES:-[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj,in_proj_qkv,in_proj_qkvz,in_proj_ba,in_proj_a,in_proj_b,in_proj_z,out_proj]}"
LR="${LR:-2e-5}"
REWARD_MODE="${REWARD_MODE:-task16-event-only-v2}"
PINCHBENCH_REWARD_RETURN_MODE="${PINCHBENCH_REWARD_RETURN_MODE:-turn}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-32}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-4}"
TEST_FREQ="${TEST_FREQ:-4}"
SAVE_FREQ="${SAVE_FREQ:-4}"
RAY_NUM_CPUS="${RAY_NUM_CPUS:-8}"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-18000}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-12000}"
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.22}"
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-32768}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-8}"
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-True}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-True}"
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-True}"
ROLLOUT_LAYERED_SUMMON="${ROLLOUT_LAYERED_SUMMON:-False}"
PINCHBENCH_BEST_CKPT="${PINCHBENCH_BEST_CKPT:-1}"
PINCHBENCH_LORA_ONLY_CKPT="${PINCHBENCH_LORA_ONLY_CKPT:-1}"
PINCHBENCH_KEEP_LATEST_CKPT="${PINCHBENCH_KEEP_LATEST_CKPT:-1}"
TRAINER_RESUME_MODE="${TRAINER_RESUME_MODE:-disable}"
RUN_STAMP="$(date +%Y%m%d_%H%M)"
EXPERIMENT_NAME="task16_reinforce_lora_${RUN_STAMP}"
TENSORBOARD_DIR="${TENSORBOARD_DIR:-${OUTPUT_DIR}/tensorboard/${EXPERIMENT_NAME}}"

export MODEL REWARD_MODE PINCHBENCH_REWARD_RETURN_MODE
export PINCHBENCH_BEST_CKPT PINCHBENCH_LORA_ONLY_CKPT PINCHBENCH_KEEP_LATEST_CKPT
export MAX_TURNS="${MAX_TURNS:-8}"
export PINCHBENCH_AGENT_MAX_PROMPT_TOKENS="${PINCHBENCH_AGENT_MAX_PROMPT_TOKENS:-${MAX_PROMPT_LENGTH}}"
export PINCHBENCH_TASK_EMA_INIT="${PINCHBENCH_TASK_EMA_INIT:-0.0}"
export PINCHBENCH_TASK_EMA_ALPHA="${PINCHBENCH_TASK_EMA_ALPHA:-0.05}"
export PINCHBENCH_TASK_EMA_VAR_INIT="${PINCHBENCH_TASK_EMA_VAR_INIT:-0.0}"
export PINCHBENCH_TERMINAL_REWARD_WEIGHT="${PINCHBENCH_TASK16_TERMINAL_REWARD_WEIGHT:-0.8}"
export PRM_VLLM_BASE_URL="${PRM_VLLM_BASE_URL:-${PINCHBENCH_GRADE_JUDGE_BASE_URL}}"
export PRM_API_KEY="${PRM_API_KEY:-${PINCHBENCH_GRADE_JUDGE_API_KEY}}"
export PRM_MODEL="${PRM_MODEL:-qwen-plus}"
export PRM_USE_CHAT_COMPLETIONS="${PRM_USE_CHAT_COMPLETIONS:-1}"
export PRM_RESOLVE_MODEL="${PRM_RESOLVE_MODEL:-0}"
export TENSORBOARD_DIR

if [ ! -f "${TRAIN_FILE}" ] || [ ! -f "${VAL_FILE}" ]; then
  echo "Missing data files under ${DATA_DIR}. Run scripts/build_task16_prompts.py first."
  exit 1
fi

if [ "${DRY_RUN}" != "1" ]; then
  python3 "${REPO_ROOT}/scripts/check_env.py"
fi

if [ "${DRY_RUN}" != "1" ]; then
  if [ "${PINCHBENCH_NO_MASKED_WHITEN:-1}" = "1" ] || [ "${PINCHBENCH_NO_MASKED_WHITEN:-1}" = "true" ]; then
    python3 "${REPO_ROOT}/patches/patch_verl_core_algos_no_whiten.py"
  fi

  if [ "${PINCHBENCH_RAY_MINIMAL_DASHBOARD_AGENT:-1}" = "1" ] || [ "${PINCHBENCH_RAY_MINIMAL_DASHBOARD_AGENT:-1}" = "true" ]; then
    python3 "${REPO_ROOT}/patches/patch_ray_minimal_dashboard_agent.py"
  fi

  if [ "${PINCHBENCH_RAY_NODE_START_WAIT_PATCH:-1}" = "1" ] || [ "${PINCHBENCH_RAY_NODE_START_WAIT_PATCH:-1}" = "true" ]; then
    python3 "${REPO_ROOT}/patches/patch_ray_node_start_wait.py"
  fi
fi

mkdir -p "${OUTPUT_DIR}" "${TENSORBOARD_DIR}"

REWARD_CONFIG_ARGS=(
  reward.reward_manager.source=importlib
  reward.reward_manager.module.path="${REWARD_MANAGER_PATH}"
  reward.reward_manager.name=PinchBenchRewardManager
)

CMD=(
  python3 -m rl.train.launch_main_ppo
  algorithm.adv_estimator=reinforce_plus_plus
  algorithm.gamma=0.0
  data.train_files="${TRAIN_FILE}"
  data.val_files="${VAL_FILE}"
  data.train_batch_size="${BATCH_SIZE}"
  data.val_batch_size="${VAL_BATCH_SIZE}"
  data.max_prompt_length="${MAX_PROMPT_LENGTH}"
  data.max_response_length="${MAX_RESPONSE_LENGTH}"
  data.filter_overlong_prompts=True
  data.truncation=left
  data.return_raw_chat=True
  +data.apply_chat_template_kwargs.enable_thinking="${ENABLE_THINKING:-False}"
  actor_rollout_ref.model.path="${MODEL}"
  actor_rollout_ref.model.enable_gradient_checkpointing=True
  actor_rollout_ref.model.use_remove_padding="${MODEL_USE_REMOVE_PADDING:-False}"
  +actor_rollout_ref.model.override_config.attn_implementation=sdpa
  actor_rollout_ref.model.lora_rank="${LORA_RANK}"
  actor_rollout_ref.model.lora_alpha="${LORA_ALPHA}"
  actor_rollout_ref.model.target_modules="${LORA_TARGET_MODULES}"
  actor_rollout_ref.actor.optim.lr="${LR}"
  actor_rollout_ref.actor.ppo_mini_batch_size="${BATCH_SIZE}"
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${MICRO_BATCH}"
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-32768}"
  actor_rollout_ref.actor.ppo_epochs=1
  actor_rollout_ref.actor.shuffle=False
  actor_rollout_ref.actor.use_remove_padding="${ACTOR_USE_REMOVE_PADDING:-False}"
  actor_rollout_ref.actor.use_kl_loss=True
  actor_rollout_ref.actor.kl_loss_coef="${KL_LOSS_COEF:-0.01}"
  actor_rollout_ref.actor.kl_loss_type=low_var_kl
  actor_rollout_ref.actor.entropy_coeff=0.01
  actor_rollout_ref.actor.fsdp_config.param_offload="${ACTOR_PARAM_OFFLOAD}"
  actor_rollout_ref.actor.fsdp_config.optimizer_offload="${ACTOR_OPTIMIZER_OFFLOAD}"
  actor_rollout_ref.actor.fsdp_config.wrap_policy.min_num_params="${FSDP_WRAP_MIN_NUM_PARAMS:-1000000}"
  actor_rollout_ref.rollout.name=vllm
  actor_rollout_ref.rollout.n=1
  actor_rollout_ref.rollout.temperature="${ROLLOUT_TEMPERATURE:-0.8}"
  actor_rollout_ref.rollout.top_p="${ROLLOUT_TOP_P:-0.9}"
  actor_rollout_ref.rollout.top_k="${ROLLOUT_TOP_K:--1}"
  actor_rollout_ref.rollout.val_kwargs.do_sample="${VAL_DO_SAMPLE:-False}"
  actor_rollout_ref.rollout.val_kwargs.temperature="${VAL_TEMPERATURE:-1.0}"
  actor_rollout_ref.rollout.val_kwargs.top_p="${VAL_TOP_P:-1.0}"
  actor_rollout_ref.rollout.val_kwargs.top_k="${VAL_TOP_K:--1}"
  actor_rollout_ref.rollout.val_kwargs.n="${VAL_N:-1}"
  actor_rollout_ref.rollout.gpu_memory_utilization="${VLLM_GPU_MEM_UTIL}"
  actor_rollout_ref.rollout.max_model_len="${VLLM_MAX_MODEL_LEN}"
  actor_rollout_ref.rollout.max_num_seqs="${VLLM_MAX_NUM_SEQS}"
  actor_rollout_ref.rollout.tensor_model_parallel_size="${N_GPUS}"
  actor_rollout_ref.rollout.load_format=safetensors
  actor_rollout_ref.rollout.layered_summon="${ROLLOUT_LAYERED_SUMMON}"
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2
  actor_rollout_ref.rollout.multi_turn.enable=True
  actor_rollout_ref.rollout.multi_turn.max_assistant_turns=30
  actor_rollout_ref.rollout.agent.default_agent_loop=openclaw_agent
  actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG}"
  actor_rollout_ref.rollout.agent.num_workers=1
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${REF_LOG_PROB_MAX_TOKEN_LEN_PER_GPU:-32768}"
  actor_rollout_ref.ref.fsdp_config.param_offload="${REF_PARAM_OFFLOAD}"
  actor_rollout_ref.ref.fsdp_config.wrap_policy.min_num_params="${FSDP_WRAP_MIN_NUM_PARAMS:-1000000}"
  algorithm.use_kl_in_reward=True
  algorithm.norm_adv_by_std_in_grpo=False
  "${REWARD_CONFIG_ARGS[@]}"
  trainer.critic_warmup=0
  trainer.val_before_train="${VAL_BEFORE_TRAIN:-False}"
  trainer.logger='["console","tensorboard"]'
  ray_kwargs.ray_init.num_cpus="${RAY_NUM_CPUS}"
  +ray_kwargs.ray_init.include_dashboard=False
  trainer.default_local_dir="${OUTPUT_DIR}"
  trainer.experiment_name="${EXPERIMENT_NAME}"
  trainer.total_epochs="${TOTAL_EPOCHS}"
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS}"
  trainer.test_freq="${TEST_FREQ}"
  trainer.save_freq="${SAVE_FREQ}"
  trainer.resume_mode="${TRAINER_RESUME_MODE}"
  trainer.n_gpus_per_node="${N_GPUS}"
  trainer.nnodes=1
)

if [ "${DRY_RUN}" = "1" ]; then
  printf '%q ' "${CMD[@]}"
  printf '\n'
  exit 0
fi

"${CMD[@]}"
