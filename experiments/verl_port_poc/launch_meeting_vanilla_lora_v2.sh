#!/usr/bin/env bash
# v2: same as v1 (vanilla GRPO + ppo_epochs=2 + LoRA r=16) BUT save only the
# LoRA adapter (130 MB safetensors), not full FSDP shards (16 GB).
#
# v1 hit MooseFS write truncation on 8 GB shards → all 3 ckpts corrupted.
# v2 uses pinchbench's verl_lora_only_ckpt_patch via sitecustomize hook.
#
# Pod prerequisites:
#   /workspace/verl_patches/sitecustomize.py
#   /workspace/verl_patches/rl/verl_lora_only_ckpt_patch.py
#   /workspace/openclaw-verl-agent-rl
#   /workspace/verl_port/data/meeting_inline_{train,val}.parquet
#   /root/.pinchbench_env (DEEPSEEK_API_KEY)

set -xeuo pipefail
cd /root/verl

REPO=/workspace/openclaw-verl-agent-rl
DATA=/workspace/verl_port/data
PATCHES=/workspace/verl_patches

: "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY before launching}"

export MEETING_TASKS_DIR="$REPO/pinchbench_tasks/meeting_analysis"
export PYTHONPATH="$PATCHES:$REPO:${PYTHONPATH:-}"
export PINCHBENCH_LORA_ONLY_CKPT=1

NGPUS_PER_NODE=2 \
ROLLOUT_TP=1 \
ROLLOUT_GPU_MEM_UTIL=0.35 \
ROLLOUT_N=4 \
TRAIN_BATCH_SIZE=4 \
PPO_MINI_BATCH_SIZE=4 \
MAX_PROMPT_LENGTH=24576 \
MAX_RESPONSE_LENGTH=2048 \
PPO_MAX_TOKEN_LEN_PER_GPU=32768 \
ACTOR_LR=1e-5 \
KL_LOSS_COEF=0.001 \
TOTAL_EPOCHS=15 \
SAVE_FREQ=15 \
TEST_FREQ=99999 \
MODEL_PATH=Qwen/Qwen3-4B \
PROJECT_NAME=verl_port_meeting \
EXPERIMENT_NAME=qwen3_4b_meeting_vanilla_lora_v2_$(date +%H%M) \
bash /root/verl/examples/grpo_trainer/run_qwen3_8b_fsdp.sh \
  data.train_files=$DATA/meeting_inline_train.parquet \
  data.val_files=$DATA/meeting_inline_val.parquet \
  data.truncation=right \
  trainer.logger='["console"]' \
  trainer.default_local_dir=/workspace/verl_port/ckpt_meeting_v2 \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.actor.ppo_epochs=2 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  custom_reward_function.path=$REPO/rewards/meeting_reward_single_turn.py \
  custom_reward_function.name=compute_score \
  reward_model.enable=False \
  reward_model.reward_manager=naive
