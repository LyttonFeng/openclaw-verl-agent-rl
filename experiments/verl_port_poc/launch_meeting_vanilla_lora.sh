#!/usr/bin/env bash
# veRL vanilla GRPO + ppo_epochs=2 (light off-policy) + LoRA rank 16
# on inlined meeting tasks. Reward = DeepSeek judge + automated checks + quality gate.
#
# vs. CISPO variant: uses standard PPO clip min-trick and k3 KL (low_var_kl),
# matches our 250-line custom trainer's PPO三件套 logic. ppo_epochs=2 gives the
# second grad step a non-trivial importance ratio without needing a separate
# rollout-LoRA flow.
#
# Pod-side prerequisites:
#   /workspace/openclaw-verl-agent-rl
#   /workspace/verl_port/data/meeting_inline_train.parquet
#   /workspace/verl_port/data/meeting_inline_val.parquet
#   $DEEPSEEK_API_KEY (in /root/.pinchbench_env)
#
# Run with `nohup bash launch_meeting_vanilla_lora.sh > /workspace/verl_port/run_meeting.log 2>&1 &`.

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
TOTAL_EPOCHS=15 \
SAVE_FREQ=15 \
TEST_FREQ=99999 \
MODEL_PATH=Qwen/Qwen3-4B \
PROJECT_NAME=verl_port_meeting \
EXPERIMENT_NAME=qwen3_4b_meeting_vanilla_lora_$(date +%H%M) \
bash /root/verl/examples/grpo_trainer/run_qwen3_8b_fsdp.sh \
  data.train_files=$DATA/meeting_inline_train.parquet \
  data.val_files=$DATA/meeting_inline_val.parquet \
  data.truncation=right \
  trainer.logger='["console"]' \
  trainer.default_local_dir=/workspace/verl_port/ckpt_meeting \
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
