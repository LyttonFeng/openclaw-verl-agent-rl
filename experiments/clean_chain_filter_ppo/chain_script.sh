#!/bin/bash
# 6-round clean chain: [filter + PPO + KL] from base Qwen3-4B
# - 每轮内 vLLM lifecycle 自动管理（rollout/bench 时启，训练时杀）
# - OOM/失败时自动重试 2 次
# - 断点续跑：每轮 bench 完成后 mark DONE，重启 chain 会跳过已完成轮

set -uo pipefail
REPO_ROOT=/workspace/openclaw-verl-agent-rl
CHAIN_DIR=/workspace/clean_chain_filter_ppo
mkdir -p $CHAIN_DIR/logs

source $HOME/.pinchbench_env

PREV_LORA=''
PREV_LORA_NAME=''

# ---- helpers ----
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $CHAIN_DIR/logs/chain.log; }

ensure_vllm() {
  if curl -sf http://127.0.0.1:8021/v1/models >/dev/null 2>&1; then return 0; fi
  log 'Starting vLLM...'
  pkill -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
  sleep 3
  nohup bash /workspace/repro/start_vllm.sh > $CHAIN_DIR/logs/vllm.log 2>&1 < /dev/null &
  disown
  for i in {1..60}; do
    if curl -sf http://127.0.0.1:8021/v1/models >/dev/null 2>&1; then
      log "vLLM ready (~$((i*5))s)"; return 0
    fi
    sleep 5
  done
  log 'ERROR: vLLM failed to start'; return 1
}

kill_vllm() {
  log 'Killing vLLM...'
  pkill -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
  sleep 8
}

hot_load() {
  local NAME=$1; local PATH_=$2
  curl -s -X POST http://127.0.0.1:8021/v1/load_lora_adapter \
    -H 'Content-Type: application/json' \
    -d "{\"lora_name\":\"$NAME\",\"lora_path\":\"$PATH_\"}" >/dev/null
}

retry() {
  local tries=$1; shift
  local n=0
  until "$@"; do
    n=$((n+1))
    if [ $n -ge $tries ]; then
      log "FAILED after $tries attempts: $*"
      return 1
    fi
    log "Attempt $n failed (rc=$?), retrying after 30s..."
    sleep 30
  done
}

run_rollout() {
  local ROLLOUT_MODEL=$1; local ROUND_DIR=$2
  python3 $REPO_ROOT/rl/train/generate_meeting_rollouts.py \
    --split-file $REPO_ROOT/rl/train/meeting_analysis_split.json \
    --split train \
    --tasks-dir $REPO_ROOT/pinchbench_tasks/meeting_analysis \
    --assets-dir $REPO_ROOT/assets \
    --vllm-base-url 'http://127.0.0.1:8021/v1' \
    --model "$ROLLOUT_MODEL" \
    --n-responses 2 \
    --output-dir "$ROUND_DIR/rollouts" \
    --judge-model 'deepseek-chat' \
    --num-workers 8 \
    --timeout 600 \
    2>&1 | tee "$ROUND_DIR/rollouts/generate.log"
}

run_compute_logprobs() {
  local LORA=$1; local IN=$2; local OUT=$3
  local ARG=''; [ -n "$LORA" ] && ARG="--lora-path $LORA"
  CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 -u /workspace/repro/compute_rollout_logprobs.py \
      --graded-file "$IN" \
      $ARG \
      --max-seq-length 65536 --rope-scaling-factor 2.0 \
      --output "$OUT"
}

run_train() {
  local PREV=$1; local TRAIN_FILE=$2; local LP=$3; local OUT=$4
  local ARG=''; [ -n "$PREV" ] && ARG="--lora-path $PREV"
  CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    python3 -u $REPO_ROOT/rl/train/train_meeting_grpo_step.py \
      --graded-file "$TRAIN_FILE" \
      --model-path Qwen/Qwen3-4B \
      $ARG \
      --output-dir "$OUT" \
      --logprobs-file "$LP" \
      --clip-eps 0.2 --kl-beta 0.02 \
      --lr 1e-6 --lora-rank 16 --grad-accum-steps 2 \
      --max-seq-length 65536 --rope-scaling-factor 2.0 \
      --prm-alpha 1.0 --prm-beta 0 --prm-mode additive
}

run_bench() {
  local NAME=$1; local OUT=$2
  cd $REPO_ROOT
  python3 scripts/benchmark.py \
    --suite 'task_meeting_advisory_stakeholders,task_meeting_council_votes,task_meeting_gov_speaker_summary,task_meeting_tech_action_items,task_meeting_sentiment_analysis' \
    --model "$NAME" \
    --base-url 'http://127.0.0.1:8021/v1' --api-key 'dummy' \
    --judge 'deepseek-chat' \
    --output-dir "$OUT" \
    --runs 3 --no-upload
}

# ---- main loop ----
for R in 1 2 3 4 5 6; do
  ROUND_DIR=$CHAIN_DIR/round_$R
  BENCH_DIR=$CHAIN_DIR/bench_$R
  NEW_LORA_NAME=clean-r${R}-fp32
  DONE_FLAG=$ROUND_DIR/DONE

  log "========== Round R${R}' =========="
  if [ -f "$DONE_FLAG" ]; then
    log "Round $R already done; skipping"
    PREV_LORA=$ROUND_DIR/checkpoint/lora_adapter
    PREV_LORA_NAME=$NEW_LORA_NAME
    continue
  fi

  mkdir -p $ROUND_DIR/rollouts $ROUND_DIR/selection $ROUND_DIR/checkpoint $BENCH_DIR

  # ---- Step 1: rollout ----
  ensure_vllm || exit 1
  if [ -n "$PREV_LORA" ]; then
    log "Hot-loading previous adapter $PREV_LORA_NAME..."
    hot_load "$PREV_LORA_NAME" "$PREV_LORA"
    sleep 5
    ROLLOUT_MODEL="$PREV_LORA_NAME"
  else
    ROLLOUT_MODEL='Qwen3-4B'
  fi
  GRADED=$ROUND_DIR/rollouts/graded_trajectories.jsonl
  if [ -s "$GRADED" ]; then
    log "Rollouts already exist ($(wc -l < $GRADED) lines), skipping"
  else
    log "Rolling out with model=$ROLLOUT_MODEL..."
    if ! retry 2 run_rollout "$ROLLOUT_MODEL" "$ROUND_DIR"; then exit 1; fi
    if [ ! -s "$GRADED" ]; then log 'ERROR: empty graded'; exit 1; fi
  fi

  # ---- Step 2: skip-PRM (synth zeros) + variance filter ----
  log 'Synth all-zero PRM + variance filter...'
  PYTHONPATH=$REPO_ROOT python3 -c "
import json
fin='$ROUND_DIR/rollouts/graded_trajectories.jsonl'
fout='$ROUND_DIR/rollouts/graded_trajectories_prm.jsonl'
with open(fin) as a, open(fout,'w') as b:
    for line in a:
        rec=json.loads(line)
        n=max(1, rec.get('prm_n_turns',3) or 3)
        rec['prm_turn_scores']=[0]*n
        rec['prm_milestones']=[]; rec['prm_pitfalls']=[]; rec['prm_reasons']=[]
        rec['prm_status']='skipped'; rec['prm_pos']=0; rec['prm_neg']=0; rec['prm_zero']=n
        b.write(json.dumps(rec)+'\n')
"
  if [ -s "$ROUND_DIR/selection/graded_trajectories_prm_valid.jsonl" ]; then
    log "Selection already done, skipping"
  else
    PYTHONPATH=$REPO_ROOT python3 $REPO_ROOT/rl/train/select_grpo_samples.py \
      --graded-file "$ROUND_DIR/rollouts/graded_trajectories_prm.jsonl" \
      --output-dir "$ROUND_DIR/selection" \
      --variance-threshold 1e-08 --alpha 1.0 \
      2>&1 | tee "$ROUND_DIR/selection/select.log"
  fi

  # ---- Step 3: quality filter ----
  if [ -s "$ROUND_DIR/selection/graded_trajectories_quality_filtered.jsonl" ]; then
    log 'Quality-filtered file already exists, skipping'
  else
    log 'Applying quality filter (race-to-bottom defense)...'
    python3 /workspace/repro/apply_quality_filter.py \
      --input "$ROUND_DIR/selection/graded_trajectories_prm_valid.jsonl" \
      --output "$ROUND_DIR/selection/graded_trajectories_quality_filtered.jsonl" \
      --report "$ROUND_DIR/selection/quality_report.json"
  fi

  TRAIN_FILE="$ROUND_DIR/selection/graded_trajectories_quality_filtered.jsonl"
  N_LINES=$(wc -l < "$TRAIN_FILE")
  log "Training samples after filter: $N_LINES"

  # ---- Step 4: kill vLLM ----
  kill_vllm

  # ---- Step 5: compute P_old ----
  if [ -s "$ROUND_DIR/rollout_logprobs.jsonl" ]; then
    log "P_old logprobs already exist, skipping"
  else
    log 'Computing P_old log_probs...'
    if ! retry 2 run_compute_logprobs "$PREV_LORA" "$TRAIN_FILE" "$ROUND_DIR/rollout_logprobs.jsonl"; then
      log 'compute_logprobs failed; abort'; exit 1
    fi
  fi

  # ---- Step 6: train PPO + KL ----
  if [ -f "$ROUND_DIR/checkpoint/lora_adapter/adapter_model.safetensors" ]; then
    log "Training adapter already exists, skipping"
  else
    log 'Training PPO + KL (multi-GPU fp32, rope=2)...'
    if ! retry 2 run_train "$PREV_LORA" "$TRAIN_FILE" "$ROUND_DIR/rollout_logprobs.jsonl" "$ROUND_DIR/checkpoint"; then
      log 'training failed; abort'; exit 1
    fi
    if [ ! -f "$ROUND_DIR/checkpoint/lora_adapter/adapter_model.safetensors" ]; then
      log 'ERROR: lora_adapter not saved'; exit 1
    fi
  fi

  # ---- Step 7: vLLM restart + hot-load ----
  ensure_vllm || exit 1
  hot_load "$NEW_LORA_NAME" "$ROUND_DIR/checkpoint/lora_adapter"
  sleep 5

  # ---- Step 8: bench ----
  if ls "$BENCH_DIR"/*.json >/dev/null 2>&1; then
    log "Bench already done, skipping"
  else
    log 'Benchmarking 5 task x 3 runs...'
    if ! retry 2 run_bench "$NEW_LORA_NAME" "$BENCH_DIR"; then
      log 'bench failed; abort'; exit 1
    fi
  fi

  # ---- Mark done ----
  touch "$DONE_FLAG"
  PREV_LORA=$ROUND_DIR/checkpoint/lora_adapter
  PREV_LORA_NAME=$NEW_LORA_NAME
  log "========== Round R${R}' DONE =========="
done

log 'ALL ROUNDS COMPLETE'
