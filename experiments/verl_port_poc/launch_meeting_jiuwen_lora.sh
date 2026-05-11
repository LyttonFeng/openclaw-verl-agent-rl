#!/usr/bin/env bash
# Tier 4: veRL GRPO + JiuwenClaw rollout + LoRA.
#
# Identical hyperparams to launch_meeting_openclaw_lora.sh except:
#   - actor_rollout_ref.rollout.agent.default_agent_loop=jiuwenclaw_agent
#   - JIUWENCLAW_* env exposed to ray workers
#
# READ FIRST — GPU resource constraint:
#   The pod has 2× A100-80GB. veRL hybrid engine (FSDP + own vLLM) and
#   the jiuwenclaw stack's own vLLM both want all GPUs and don't fit.
#   The previous OpenClaw v3 run (PID 393686, 2026-05-11) was OOM-killed
#   exactly this way when a leftover jiuwenclaw stack was running.
#
#   Two viable modes:
#
#   (A) PATH A — jiuwenclaw stack runs WITHOUT its own vLLM; the gateway
#       proxies /v1/chat/completions to a vLLM HTTP endpoint exposed by
#       veRL's hybrid engine. Cleanest (always-latest weights, no sync
#       hook needed). Requires implementing a ModelProxy aiohttp bridge
#       inside veRL — NOT YET DONE. Set MODE=path-a once that lands.
#
#   (B) PATH B — keep jiuwenclaw's own vLLM running; after each grad step
#       dump the LoRA + POST /v1/load_lora_adapter (see jiuwen_lora_sync.py).
#       Requires GPU partitioning so jiuwenclaw vLLM and veRL FSDP+vLLM
#       don't fight. With 2 GPUs that means 1 GPU each + smaller TP — not
#       realistic for Qwen3-4B + meeting prompts (16k context). Needs ≥4
#       GPUs.
#
# DO NOT run this script blindly until one of A or B is set up.
#   - For (A): start jiuwenclaw with NO_LOCAL_VLLM=1 (pending impl) and
#     set JIUWEN_VLLM_PROXY_URL to the veRL-exposed vLLM HTTP endpoint
#     (also pending impl).
#   - For (B): start jiuwenclaw stack on CUDA_VISIBLE_DEVICES=0 only,
#     veRL on CUDA_VISIBLE_DEVICES=1 only, halve batch sizes — and accept
#     that 1 GPU won't fit Qwen3-4B+16k+ref+optimizer.
#
# This launcher pre-flights GPU memory before invoking veRL and aborts
# with a clear message if contention is likely.

set -euo pipefail

# ─── Sanity: jiuwenclaw stack reachable, env loaded ───────
: "${JIUWENCLAW_WS_URL:=ws://127.0.0.1:611/ws}"
: "${JIUWEN_VLLM_BASE:=http://127.0.0.1:614}"

if [ -f /root/.pinchbench_env ]; then
  set -a; source /root/.pinchbench_env; set +a
fi
: "${DEEPSEEK_API_KEY:?Set DEEPSEEK_API_KEY (terminal judge)}"

# Pre-flight: refuse to launch if GPUs are already heavily used (likely
# leftover jiuwenclaw stack or stale veRL).
echo "[preflight] GPU memory check..."
GPU_USED_MAX=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
GPU_USED_THRESHOLD_MB="${GPU_USED_THRESHOLD_MB:-20000}"  # >20GB on any card = contention
if [ "$GPU_USED_MAX" -gt "$GPU_USED_THRESHOLD_MB" ]; then
  echo "[preflight] ABORT: a GPU has ${GPU_USED_MAX} MiB used (>${GPU_USED_THRESHOLD_MB} MiB)."
  echo "[preflight] Likely a leftover jiuwenclaw stack or veRL process. Free GPU first:"
  echo "  pkill -9 -f 'vllm.entrypoints' ; pkill -9 -f run_online_rl ; pkill -9 -f launch_main_ppo"
  echo "[preflight] Then RESTART jiuwenclaw stack WITHOUT its own vLLM (Path A) or with"
  echo "[preflight] CUDA_VISIBLE_DEVICES=0 (Path B), and re-run this launcher."
  exit 2
fi

# WS reachability
if ! timeout 5 bash -c "(echo > /dev/tcp/127.0.0.1/611) 2>/dev/null"; then
  echo "[preflight] ABORT: jiuwenclaw WS port 611 not reachable."
  echo "[preflight] Start the stack: bash /root/jiuwen_work/start_jw_pod.sh"
  exit 2
fi

# ─── Paths (mirrors launch_meeting_openclaw_lora.sh) ──────
REPO_INTEGRATION=/workspace/verl_port/openclaw_integration
REPO_DATA=/workspace/openclaw-verl-agent-rl
DATA_DIR=/workspace/verl_port/data_meeting
OUTPUT_DIR=/workspace/verl_port/ckpt_jw
AGENT_LOOP_CONFIG="${REPO_DATA}/agent_loop/config.yaml"

# Reward (same vanilla terminal-only path as OpenClaw baseline)
export REWARD_MODE="baseline"
export PINCHBENCH_REWARD_RETURN_MODE="scalar"
export PINCHBENCH_GRADE_JUDGE_MODEL="${PINCHBENCH_GRADE_JUDGE_MODEL:-deepseek-chat}"
export PINCHBENCH_GRADE_JUDGE_BACKEND="${PINCHBENCH_GRADE_JUDGE_BACKEND:-api}"
export PINCHBENCH_GRADE_JUDGE_BASE_URL="${PINCHBENCH_GRADE_JUDGE_BASE_URL:-https://api.deepseek.com/v1}"
export PINCHBENCH_GRADE_JUDGE_API_KEY="${PINCHBENCH_GRADE_JUDGE_API_KEY:-${DEEPSEEK_API_KEY}}"

REWARD_MANAGER_PATH="${REPO_DATA}/rewards/meeting_reward_single_turn.py"

# ─── Hyperparams ──────────────────────────────────────────
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
ACTOR_PARAM_OFFLOAD="${ACTOR_PARAM_OFFLOAD:-True}"
ACTOR_OPTIMIZER_OFFLOAD="${ACTOR_OPTIMIZER_OFFLOAD:-True}"
REF_PARAM_OFFLOAD="${REF_PARAM_OFFLOAD:-True}"
SAVE_FREQ="${SAVE_FREQ:-8}"
TEST_FREQ="${TEST_FREQ:--1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-24}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-300}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-verl_jw_grpo_$(date +%Y%m%d_%H%M%S)}"

mkdir -p "$OUTPUT_DIR"

env PYTHONPATH="${REPO_INTEGRATION}:${REPO_DATA}:${PYTHONPATH:-}" \
  python3 -m rl.train.launch_main_ppo \
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
    actor_rollout_ref.rollout.agent.default_agent_loop=jiuwenclaw_agent \
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
    +ray_kwargs.ray_init.runtime_env.env_vars.JIUWENCLAW_WS_URL="'${JIUWENCLAW_WS_URL}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.JIUWENCLAW_TIMEOUT="'${AGENT_TIMEOUT}'" \
    +ray_kwargs.ray_init.runtime_env.env_vars.JIUWEN_VLLM_BASE="'${JIUWEN_VLLM_BASE}'" \
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
    trainer.n_gpus_per_node="${N_GPUS}" \
    trainer.nnodes=1 \
    trainer.save_freq="${SAVE_FREQ}" \
    trainer.test_freq="${TEST_FREQ}" \
    trainer.total_epochs="${TOTAL_EPOCHS}" \
    trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
    trainer.resume_mode=disable \
    trainer.default_local_dir="${OUTPUT_DIR}"
