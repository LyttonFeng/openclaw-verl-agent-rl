#!/usr/bin/env bash
# Tier 4 ALT: veRL FullyAsyncTrainer + JiuwenClaw rollout.
#
# Vs launch_meeting_jiuwen_path_a.sh (sync): this uses the async trainer in
# verl/experimental/fully_async_policy/. Key wins:
#   - No asyncio.gather() blocking on slow rollouts (one chat.error 不再卡死整步)
#   - Trainer and Rollout split onto separate GPUs (no hybrid engine contention)
#   - Importance-sampling correction for stale rollouts
#
# GPU layout (2× A100 80GB):
#   GPU 0 → Trainer (FSDP actor + ref + optim)
#   GPU 1 → Rollout vLLM (49k context, max_num_seqs=4)
#
# Note: still uses headless jiuwenclaw + Path A (vLLM HTTP served by veRL's
# rollout side, jiuwenclaw points API_BASE there).

set -euo pipefail

DRY_RUN=0
for arg in "$@"; do [ "$arg" = "--dry-run" ] && DRY_RUN=1; done

# ─── Paths ────────────────────────────────────────────────
REPO_INTEGRATION=/workspace/verl_port/openclaw_integration
REPO_DATA=/workspace/openclaw-verl-agent-rl
DATA_DIR=/workspace/verl_port/data_meeting
OUTPUT_DIR="${OUTPUT_DIR:-/workspace/verl_port/ckpt_jw_async}"
AGENT_LOOP_CONFIG="${REPO_DATA}/agent_loop/config.yaml"
LOG_DIR=/tmp/jw_async
mkdir -p "$LOG_DIR" "$OUTPUT_DIR"
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
export VLLM_USE_V1=1

MODEL_NAME="${MODEL_NAME:-Qwen3-4B}"

# ─── Rail-v1 knobs (used by pre-flight gateway + later ray env) ───
# When USE_RL_ONLINE_RAIL=1, jiuwenclaw stack injects RLOnlineRail
# (via sitecustomize.py + inject_rl_online_rail.py), captures vLLM-original
# prompt_ids / response_tokens / logprobs per turn, uploads PerTurnSample
# batches to TRAJECTORY_GATEWAY_URL. agent_loop reads matching JSONL from
# RAIL_V1_DIR by session_id (replaces history.json reverse-engineering).
USE_RL_ONLINE_RAIL="${USE_RL_ONLINE_RAIL:-1}"
TRAJECTORY_GATEWAY_URL="${TRAJECTORY_GATEWAY_URL:-http://127.0.0.1:9000}"
RAIL_V1_DIR="${RAIL_V1_DIR:-/tmp/jw_rail_v1}"
RAIL_V1_WAIT_S="${RAIL_V1_WAIT_S:-15}"
# Kill cross-trajectory drift (memory.db + daily_memory + MEMORY.md).
MEMORY_ENABLED="${MEMORY_ENABLED:-false}"
JW_N_STACKS="${JW_N_STACKS:-2}"

# ─── Pre-flight: kill stale procs + verify ports + GPU are free ───
# Catastrophic past failure: v52 jiuwenclaw subprocesses survived our pkill
# (broken regex), v53 stack startup "succeeded" because port 611/612 were
# already bound by old stacks — so chat.send hit DEAD endpoints (their
# vLLM URL was from v52, killed). Result: 100% chat.error, 0 rail data.
# Bullet-proof now: enumerate by process name (no fragile regex), retry,
# and abort if anything still bound to the ports we need.
echo "[async] pre-flight: kill stale procs..."
_PKILL_PATTERNS=(
  'run_online_rl'
  'run_jw_app_with_rl_rail'
  'jiuwenclaw\.app$'
  'jiuwenclaw\.server\.app_agentserver'
  'jiuwenclaw\.gateway\.app_gateway'
  'mock_trajectory_gateway'
  'vllm\.entrypoints'
  'vllm serve'
  'fully_async_main'
  'launch_main_ppo'
)
# Note: 'launch_meeting_jiuwen_async' deliberately EXCLUDED — pkill -f would
# match this script's own cmdline (we are it) and suicide. Stale launcher
# wrappers from prior runs are killed below by enumerating self's PID first.
_MY_PID=$$
for _pid in $(pgrep -f 'launch_meeting_jiuwen_async' 2>/dev/null); do
  if [ "$_pid" != "$_MY_PID" ] && [ "$_pid" != "$PPID" ]; then
    kill -9 "$_pid" 2>/dev/null || true
  fi
done
for _pat in "${_PKILL_PATTERNS[@]}"; do
  pkill -9 -f "$_pat" 2>/dev/null || true
done
sleep 4
# VLLM engine cores are subprocesses kept alive by Ray supervisors; do a 2nd pass
for _pid in $(pgrep -f 'VLLM::EngineCore\|VLLM::Worker' 2>/dev/null); do
  kill -9 "$_pid" 2>/dev/null || true
done
sleep 2

# Verify the ports we plan to bind are actually free
_PORTS_TO_CHECK=(9000)
for _i in $(seq 0 $((JW_N_STACKS - 1))); do
  _PORTS_TO_CHECK+=($((611 + _i)) $((18092 + _i)) $((19001 + _i)))
done
_PORTS_STILL_BUSY=""
for _p in "${_PORTS_TO_CHECK[@]}"; do
  if ss -lnt 2>/dev/null | awk '{print $4}' | grep -q ":${_p}\$"; then
    _PORTS_STILL_BUSY="${_PORTS_STILL_BUSY} ${_p}"
  fi
done
if [ -n "$_PORTS_STILL_BUSY" ]; then
  echo "[async] FATAL: ports still bound after pre-flight cleanup:$_PORTS_STILL_BUSY"
  echo "[async]  ss -lntp output:"
  ss -lntp 2>&1 | grep -E ":(611|612|18092|18093|19001|19002|9000) " || true
  exit 7
fi

GPU_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
echo "[async] post-cleanup GPU max used: ${GPU_USED} MiB ; ports clear"
if [ "$GPU_USED" -gt 5000 ]; then
  echo "[async] FATAL: GPU still has ${GPU_USED}MiB after cleanup — manual kill needed"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
  exit 8
fi

# ─── Mock trajectory gateway (receives RLOnlineRail PerTurnSample batches) ──
# Required when USE_RL_ONLINE_RAIL=1. Each rollout's trajectory lands as a
# JSONL file under RAIL_V1_DIR keyed by jiuwenclaw trajectory_id. Our
# agent_loop scans for session_id matches after the WS chat.send completes.
if [ "$USE_RL_ONLINE_RAIL" = "1" ]; then
  pkill -9 -f 'mock_trajectory_gateway' 2>/dev/null || true
  sleep 1
  rm -rf "$RAIL_V1_DIR"; mkdir -p "$RAIL_V1_DIR"
  GW_LOG="$LOG_DIR/mock_gateway_$TS.log"
  echo "[async] starting mock trajectory gateway at $TRAJECTORY_GATEWAY_URL → $RAIL_V1_DIR (log=$GW_LOG)"
  nohup python3 "$(dirname "$0")/mock_trajectory_gateway.py" \
    --port "${TRAJECTORY_GATEWAY_URL##*:}" \
    --out "$RAIL_V1_DIR" \
    > "$GW_LOG" 2>&1 &
  GW_PID=$!
  echo "$GW_PID" > "$LOG_DIR/mock_gateway.pid"
  # Wait for /health to respond
  GW_HOST_PORT="${TRAJECTORY_GATEWAY_URL#http://}"
  for i in $(seq 1 30); do
    if curl -sf --max-time 1 "http://${GW_HOST_PORT}/health" > /dev/null 2>&1; then
      echo "[async] mock gateway ready (${i}s)"; break
    fi
    sleep 1
    if [ "$i" = "30" ]; then
      echo "[async] FATAL: mock gateway didn't come up"
      tail -20 "$GW_LOG" >&2
      kill -9 "$GW_PID" 2>/dev/null || true
      exit 6
    fi
  done
fi

# ─── Hyperparams (healthier defaults) ──────────────────────
MODEL=/root/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c
N_GPUS_TRAINER="${N_GPUS_TRAINER:-1}"   # FSDP actor + ref on GPU 0
N_GPUS_ROLLOUT="${N_GPUS_ROLLOUT:-1}"   # vLLM on GPU 1
RAY_NUM_GPUS="${RAY_NUM_GPUS:-$((N_GPUS_TRAINER + N_GPUS_ROLLOUT))}"
BATCH_SIZE="${BATCH_SIZE:-4}"
MICRO_BATCH="${MICRO_BATCH:-1}"
ROLLOUT_N="${ROLLOUT_N:-2}"
MAX_TURNS="${MAX_TURNS:-15}"                   # healthy middle: not 20 (累积爆) not 10 (too short)

# Multi-stack jiuwenclaw for true rollout parallelism (each stack = independent
# WS / workspace / agent_server / gateway). agent_loop round-robins. Ports
# allocated starting at 611,612,... matched by JW_DATA_DIRS index.
JIUWENCLAW_WS_URLS=""
JIUWENCLAW_DATA_DIRS=""
for _i in $(seq 0 $((JW_N_STACKS - 1))); do
  _port=$((611 + _i))
  _dd="$HOME/.jiuwenclaw_${_i}"
  [ -z "$JIUWENCLAW_WS_URLS" ] && JIUWENCLAW_WS_URLS="ws://127.0.0.1:${_port}/ws" || JIUWENCLAW_WS_URLS="${JIUWENCLAW_WS_URLS},ws://127.0.0.1:${_port}/ws"
  [ -z "$JIUWENCLAW_DATA_DIRS" ] && JIUWENCLAW_DATA_DIRS="$_dd" || JIUWENCLAW_DATA_DIRS="${JIUWENCLAW_DATA_DIRS},$_dd"
done
echo "[async] planned JW_N_STACKS=$JW_N_STACKS WS_URLS=$JIUWENCLAW_WS_URLS"
MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-20000}"
MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-8000}"  # 6k 太紧 12k 太宽，8k 中庸
LR="${LR:-2e-6}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.01}"
LORA_RANK="${LORA_RANK:-32}"
LORA_ALPHA="${LORA_ALPHA:-64}"
LORA_TARGET_MODULES='[q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj]'
VLLM_MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-49152}"
VLLM_MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-4}"          # 可恢复 4：rollout 独占 GPU 没 OOM 压力
VLLM_GPU_MEM_UTIL="${VLLM_GPU_MEM_UTIL:-0.5}"        # rollout 独占 GPU，可拉回 0.5
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU="${ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU:-32000}"
SAVE_FREQ="${SAVE_FREQ:-1}"
TEST_FREQ="${TEST_FREQ:--1}"
TOTAL_EPOCHS="${TOTAL_EPOCHS:-1}"
TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-24}"
AGENT_TIMEOUT="${AGENT_TIMEOUT:-300}"

# Async-specific
STALENESS_THRESHOLD="${STALENESS_THRESHOLD:-0.3}"     # 30% stale 上限
TRIGGER_PARAM_SYNC_STEP="${TRIGGER_PARAM_SYNC_STEP:-2}"
REQUIRE_BATCHES="${REQUIRE_BATCHES:-4}"

EXPERIMENT_NAME="${EXPERIMENT_NAME:-verl_jw_async_$TS}"
REWARD_MANAGER_PATH="${REPO_DATA}/rewards/meeting_reward.py"
TRAINER_RESUME_MODE="${TRAINER_RESUME_MODE:-disable}"

# ─── Patch model config.json for yarn rope scaling (81920 ctx) ─────
# Qwen3-4B native max_position_embeddings=40960 is too small for jiuwen's
# heavy prompt + tool defs + transcript. Patch in-place; vLLM picks it up
# at load time. (See memory/jiuwenclaw_qwen3_4b_baseline_chain.md.)
ROPE_FACTOR="${ROPE_FACTOR:-2.0}"
if [ -f "${MODEL}/config.json" ]; then
  python3 - <<PY
import json
p = "${MODEL}/config.json"
d = json.load(open(p))
target = {"rope_type": "yarn", "factor": float(${ROPE_FACTOR}), "original_max_position_embeddings": 40960}
if d.get("rope_scaling") != target:
    d["rope_scaling"] = target
    json.dump(d, open(p, "w"), indent=2)
    print(f"[async] patched ${MODEL}/config.json rope_scaling={target}")
else:
    print(f"[async] rope_scaling already {target}")
PY
fi

# ─── Launch veRL FullyAsyncTrainer ─────────────────────────
if [ "$DRY_RUN" = "0" ]; then
  echo "[async] launching veRL FullyAsyncTrainer → $VERL_LOG"
  cd "$REPO_INTEGRATION"
  env PYTHONPATH="${REPO_INTEGRATION}:${REPO_DATA}:${PYTHONPATH:-}" VLLM_USE_V1=1 \
  nohup python3 -m verl.experimental.fully_async_policy.fully_async_main \
      --config-name fully_async_ppo_trainer \
      algorithm.adv_estimator=grpo \
      algorithm.use_kl_in_reward=False \
      algorithm.norm_adv_by_std_in_grpo=False \
      algorithm.rollout_correction.bypass_mode=False \
      data.train_files="${DATA_DIR}/train.parquet" \
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
      +actor_rollout_ref.rollout.engine_kwargs.vllm.reasoning_parser=deepseek_r1 \
      actor_rollout_ref.rollout.calculate_log_probs=True \
      actor_rollout_ref.rollout.layered_summon=True \
      actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
      actor_rollout_ref.rollout.multi_turn.enable=True \
      actor_rollout_ref.rollout.multi_turn.max_assistant_turns="${MAX_TURNS}" \
      actor_rollout_ref.rollout.agent.default_agent_loop=jiuwenclaw_agent \
      actor_rollout_ref.rollout.agent.agent_loop_config_path="${AGENT_LOOP_CONFIG}" \
      actor_rollout_ref.rollout.agent.num_workers="${AGENT_NUM_WORKERS:-1}" \
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
      +ray_kwargs.ray_init.num_gpus="${RAY_NUM_GPUS}" \
      +ray_kwargs.ray_init.runtime_env.env_vars.HF_HOME=/root/hf_cache \
      +ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_CACHE=/root/hf_cache/hub \
      +ray_kwargs.ray_init.runtime_env.env_vars.JIUWENCLAW_WS_URL="'ws://127.0.0.1:611/ws'" \
      +ray_kwargs.ray_init.runtime_env.env_vars.JIUWENCLAW_WS_URLS="'${JIUWENCLAW_WS_URLS}'" \
      +ray_kwargs.ray_init.runtime_env.env_vars.JIUWENCLAW_DATA_DIRS="'${JIUWENCLAW_DATA_DIRS}'" \
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
      +ray_kwargs.ray_init.runtime_env.env_vars.VLLM_USE_V1="'1'" \
      +ray_kwargs.ray_init.runtime_env.env_vars.USE_RL_ONLINE_RAIL="'${USE_RL_ONLINE_RAIL}'" \
      +ray_kwargs.ray_init.runtime_env.env_vars.RAIL_V1_DIR="'${RAIL_V1_DIR}'" \
      +ray_kwargs.ray_init.runtime_env.env_vars.RAIL_V1_WAIT_S="'${RAIL_V1_WAIT_S}'" \
      trainer.project_name=verl_port_meeting \
      trainer.experiment_name="${EXPERIMENT_NAME}" \
      trainer.n_gpus_per_node="${N_GPUS_TRAINER}" trainer.nnodes=1 \
      trainer.save_freq="${SAVE_FREQ}" trainer.test_freq="${TEST_FREQ}" \
      trainer.total_epochs="${TOTAL_EPOCHS}" \
      trainer.total_training_steps="${TOTAL_TRAINING_STEPS}" \
      trainer.resume_mode="${TRAINER_RESUME_MODE}" \
      trainer.default_local_dir="${OUTPUT_DIR}" \
      >"$VERL_LOG" 2>&1 &
  VERL_PID=$!
  echo "[async] veRL PID=$VERL_PID log=$VERL_LOG"

  # Discover vLLM HTTP URL (same as Path A — port scan since fully async uses vLLMHttpServer too)
  echo "[async] waiting for vLLM HTTP server (up to 900s, port scan)..."
  VERL_VLLM_URL=""
  for i in $(seq 1 450); do
    if ! kill -0 "$VERL_PID" 2>/dev/null; then
      echo "[async] FATAL: veRL exited before vLLM HTTP came up"
      tail -50 "$VERL_LOG" >&2
      exit 3
    fi
    LINE=$(grep -m1 "LLMServerManager:" "$VERL_LOG" 2>/dev/null || true)
    if [ -n "$LINE" ]; then
      ADDR=$(echo "$LINE" | grep -oE "'[0-9.]+:[0-9]+'" | head -1 | tr -d "'")
      [ -n "$ADDR" ] && VERL_VLLM_URL="http://${ADDR}/v1" && break
    fi
    if [ $((i % 5)) -eq 0 ]; then
      CANDIDATES=$(ss -tlnp 2>/dev/null | awk '/LISTEN/ && /vLLMHttp/ {print $4}' | sort -u)
      for ADDR_CAND in $CANDIDATES; do
        URL="http://${ADDR_CAND}/v1"
        if curl -sf --max-time 2 "${URL}/models" 2>/dev/null | grep -q '"data"'; then
          VERL_VLLM_URL="$URL"; break 2
        fi
      done
    fi
    sleep 2
  done
  if [ -z "$VERL_VLLM_URL" ]; then
    echo "[async] FATAL: timeout waiting for vLLM HTTP."
    tail -30 "$VERL_LOG" >&2
    kill -9 "$VERL_PID" 2>/dev/null || true
    exit 4
  fi
  echo "[async] discovered veRL vLLM HTTP: $VERL_VLLM_URL"
  # Multi-replica round-robin: parse ALL addrs from LLMServerManager line.
  # vLLM rollout splits across N_GPUS_ROLLOUT replicas, but base discovery
  # only picks the first. Distribute stacks across all replicas so GPU
  # utilization is balanced (otherwise N-1 vLLM GPUs sit idle).
  LLM_LINE=$(grep -m1 "LLMServerManager:" "$VERL_LOG" 2>/dev/null || true)
  VERL_VLLM_URLS=()
  if [ -n "$LLM_LINE" ]; then
    while IFS= read -r _addr; do
      [ -n "$_addr" ] && VERL_VLLM_URLS+=("http://${_addr}/v1")
    done < <(echo "$LLM_LINE" | grep -oE "[0-9.]+:[0-9]+")
  fi
  if [ ${#VERL_VLLM_URLS[@]} -eq 0 ]; then VERL_VLLM_URLS=("$VERL_VLLM_URL"); fi
  echo "[async] discovered ${#VERL_VLLM_URLS[@]} vLLM replicas: ${VERL_VLLM_URLS[*]}"
else
  VERL_VLLM_URL="${VERL_VLLM_URL:-http://127.0.0.1:614/v1}"
  echo "[async] DRY RUN — using $VERL_VLLM_URL"
fi

# Discover model id
DISCOVERED_MODEL=$(curl -sf --max-time 5 "${VERL_VLLM_URL%/}/models" 2>/dev/null \
  | python3 -c "import json,sys; d=json.load(sys.stdin); print(d['data'][0]['id']) if d.get('data') else None" 2>/dev/null)
[ -z "$DISCOVERED_MODEL" ] && DISCOVERED_MODEL="$MODEL_NAME"
echo "[async] using MODEL_NAME=$DISCOVERED_MODEL"

# Start N headless jiuwenclaw stacks for parallelism (B fix: each stack
# isolated workspace + WS port + agent/gateway ports; agent_loop round-robins).
JW_N_STACKS="${JW_N_STACKS:-2}"
JW_WS_URLS=""
JW_DATA_DIRS=""
for i in $(seq 0 $((JW_N_STACKS - 1))); do
  WS_PORT=$((611 + i))
  AGENT_SERVER_PORT=$((18092 + i))
  GATEWAY_PORT=$((19001 + i))
  JW_DATA_DIR="$HOME/.jiuwenclaw_${i}"
  echo "[async] launching headless jiuwenclaw stack #$i: ws://127.0.0.1:${WS_PORT}/ws data=${JW_DATA_DIR}"
  _STACK_API_BASE="${VERL_VLLM_URLS[$((i % ${#VERL_VLLM_URLS[@]}))]:-$VERL_VLLM_URL}"
  API_BASE="$_STACK_API_BASE" MODEL_NAME="$DISCOVERED_MODEL" \
    WS_PORT="$WS_PORT" AGENT_SERVER_PORT="$AGENT_SERVER_PORT" GATEWAY_PORT="$GATEWAY_PORT" \
    JIUWENCLAW_DATA_DIR="$JW_DATA_DIR" \
    LOG_DIR="${JW_LOG_DIR}/stack_${i}" \
    USE_RL_ONLINE_RAIL="$USE_RL_ONLINE_RAIL" \
    TRAJECTORY_GATEWAY_URL="$TRAJECTORY_GATEWAY_URL" \
    MEMORY_ENABLED="$MEMORY_ENABLED" \
    MAX_ITERATIONS="$MAX_TURNS" \
    JIUWENCLAW_PROGRESSIVE_TOOLS="${JIUWENCLAW_PROGRESSIVE_TOOLS:-1}" \
    JIUWENCLAW_PROGRESSIVE_VISIBLE="${JIUWENCLAW_PROGRESSIVE_VISIBLE:-read_file,write_file,list_files,edit_file,grep,glob,todo_create,todo_list,bash,memory_search,code}" \
    bash "$(dirname "$0")/start_jw_headless.sh"
  if ! (echo > /dev/tcp/127.0.0.1/${WS_PORT}) 2>/dev/null; then
    echo "[async] FATAL: jiuwenclaw stack #$i WS not up on ${WS_PORT}"
    exit 5
  fi
  [ -z "$JW_WS_URLS" ] && JW_WS_URLS="ws://127.0.0.1:${WS_PORT}/ws" || JW_WS_URLS="${JW_WS_URLS},ws://127.0.0.1:${WS_PORT}/ws"
  [ -z "$JW_DATA_DIRS" ] && JW_DATA_DIRS="$JW_DATA_DIR" || JW_DATA_DIRS="${JW_DATA_DIRS},${JW_DATA_DIR}"
done
echo "[async] $JW_N_STACKS stacks up. URLs=$JW_WS_URLS"
echo "[async] Async wiring complete."

if [ "$DRY_RUN" = "1" ]; then exit 0; fi

echo "[async] tailing veRL log (Ctrl-C exits tail; veRL continues as PID=$VERL_PID)"
tail -f "$VERL_LOG"
