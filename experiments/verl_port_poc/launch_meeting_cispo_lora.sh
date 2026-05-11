#!/usr/bin/env bash
# veRL CISPO (off-policy GRPO) + LoRA rank 16 on inlined meeting tasks.
# Reward = DeepSeek judge + automated checks (single-turn variant).
#
# Pod-side prerequisites (must already exist before running):
#   /workspace/openclaw-verl-agent-rl                — repo checkout
#   /workspace/verl_port/data/meeting_inline_train.parquet
#   /workspace/verl_port/data/meeting_inline_val.parquet
#   $DEEPSEEK_API_KEY exported in the environment
#
# Run with `nohup bash launch_meeting_cispo_lora.sh > /workspace/verl_port/run_meeting.log 2>&1 &`.

set -xeuo pipefail
cd /root/verl

REPO=/workspace/openclaw-verl-agent-rl
DATA=/workspace/verl_port/data

: "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY before launching}"

export MEETING_TASKS_DIR="$REPO/pinchbench_tasks/meeting_analysis"
export PYTHONPATH="$REPO:${PYTHONPATH:-}"

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
CLIP_RATIO_LOW=10 \
CLIP_RATIO_HIGH=0.2 \
TOTAL_EPOCHS=15 \
SAVE_FREQ=15 \
TEST_FREQ=99999 \
MODEL_PATH=Qwen/Qwen3-4B \
PROJECT_NAME=verl_port_meeting \
EXPERIMENT_NAME=qwen3_4b_meeting_cispo_lora_$(date +%H%M) \
bash /root/verl/examples/cispo_trainer/run_qwen3_8b_fsdp.sh \
  data.train_files=$DATA/meeting_inline_train.parquet \
  data.val_files=$DATA/meeting_inline_val.parquet \
  data.truncation=right \
  trainer.logger='["console"]' \
  trainer.default_local_dir=/workspace/verl_port/ckpt_meeting \
  actor_rollout_ref.model.lora_rank=16 \
  actor_rollout_ref.model.lora_alpha=32 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.rollout.load_format=safetensors \
  custom_reward_function.path=$REPO/rewards/meeting_reward_single_turn.py \
  custom_reward_function.name=compute_score \
  reward_model.enable=False \
  reward_model.reward_manager=naive
