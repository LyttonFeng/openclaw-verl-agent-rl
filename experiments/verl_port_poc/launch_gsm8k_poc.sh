#!/usr/bin/env bash
# veRL GRPO POC: Qwen3-4B + GSM8K, 2x A100, console-only logger
set -xeuo pipefail

cd /root/verl

NGPUS_PER_NODE=2 \
ROLLOUT_TP=1 \
ROLLOUT_GPU_MEM_UTIL=0.4 \
TRAIN_BATCH_SIZE=32 \
PPO_MINI_BATCH_SIZE=16 \
MAX_PROMPT_LENGTH=512 \
MAX_RESPONSE_LENGTH=1024 \
PPO_MAX_TOKEN_LEN_PER_GPU=8192 \
ROLLOUT_N=2 \
ACTOR_LR=1e-6 \
KL_LOSS_COEF=0.001 \
TOTAL_EPOCHS=1 \
SAVE_FREQ=10 \
TEST_FREQ=10 \
MODEL_PATH=Qwen/Qwen3-4B \
PROJECT_NAME=verl_port_poc \
EXPERIMENT_NAME=qwen3_4b_gsm8k_grpo_poc_$(date +%H%M) \
bash /root/verl/examples/grpo_trainer/run_qwen3_8b_fsdp.sh \
  data.train_files=/workspace/verl_port/gsm8k/train.parquet \
  data.val_files=/workspace/verl_port/gsm8k/test.parquet \
  trainer.logger='[console]' \
  trainer.default_local_dir=/workspace/verl_port/checkpoints
