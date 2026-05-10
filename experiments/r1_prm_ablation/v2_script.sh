#!/bin/bash
# R1' + PRM ablation：跟 clean_chain R1' 完全一样，但启用 PRM。
# 与 R1' (46.90%) 干净对比，看 PRM 是否在 filter+PPO 之上还有增量。
#
# 配置：
#   --prm-mode multiplicative
#   --prm-beta 0.5  (50% boost on prm=+1 turns)
#   pos-only clip:  apply_quality_filter 默认 max(0, int(s))
#   terminal gate:  judge.judge_terminal_completion 内置（默认开）

set -uo pipefail
REPO_ROOT=/workspace/openclaw-verl-agent-rl
SRC_ROUND_DIR=/workspace/clean_chain_filter_ppo/round_1   # 复用 R1' 已有 rollouts
ABLATE_DIR=/workspace/clean_chain_filter_ppo_prm_ablation_v2
ROUND_DIR=$ABLATE_DIR/round_1
BENCH_DIR=$ABLATE_DIR/bench_1
NEW_LORA_NAME=clean-r1-prm-ablate-v2

mkdir -p $ROUND_DIR/rollouts $ROUND_DIR/selection $ROUND_DIR/checkpoint $BENCH_DIR $ABLATE_DIR/logs

source $HOME/.pinchbench_env

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $ABLATE_DIR/logs/main.log; }

ensure_vllm() {
  if curl -sf http://127.0.0.1:8021/v1/models >/dev/null 2>&1; then return 0; fi
  log 'Starting vLLM...'
  pkill -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
  sleep 3
  nohup bash /workspace/repro/start_vllm.sh > $ABLATE_DIR/logs/vllm.log 2>&1 < /dev/null &
  disown
  for i in $(seq 1 60); do
    if curl -sf http://127.0.0.1:8021/v1/models >/dev/null 2>&1; then log 'vLLM ready'; return 0; fi
    sleep 5
  done
  log 'ERROR: vLLM start timeout'
  return 1
}

retry() {
  local tries=$1; shift
  local n=0
  until "$@"; do
    n=$((n+1))
    if [ $n -ge $tries ]; then log "FAILED: $*"; return 1; fi
    log "Retry $n/$tries after 30s..."
    sleep 30
  done
}

log '========== R1 PRM ablation =========='

# Step 1: Reuse R1' rollouts (干净对比同一份 rollout 数据)
log "Reusing R1' rollouts from $SRC_ROUND_DIR/rollouts"
cp -r $SRC_ROUND_DIR/rollouts/* $ROUND_DIR/rollouts/

# Step 2: PRM scoring (替换原来的 synth-zero)
log 'Running PRM per-turn scoring (DSv4 judge, this is the new step)...'
PYTHONPATH=$REPO_ROOT python3 $REPO_ROOT/agent_loop/roadmap_prm/scripts/score_trajectories.py \
  --graded-file $ROUND_DIR/rollouts/graded_trajectories.jsonl \
  --tasks-dir $REPO_ROOT/pinchbench_tasks/meeting_analysis \
  --roadmaps-dir $REPO_ROOT/agent_loop/roadmap_prm/roadmaps \
  --output-suffix _prm \
  --max-workers 4 \
  2>&1 | tee $ROUND_DIR/rollouts/prm_score.log

# Step 3: Variance filter
log 'Selection (variance filter)...'
PYTHONPATH=$REPO_ROOT python3 $REPO_ROOT/rl/train/select_grpo_samples.py \
  --graded-file $ROUND_DIR/rollouts/graded_trajectories_prm.jsonl \
  --output-dir $ROUND_DIR/selection \
  --variance-threshold 1e-08 --alpha 1.0 \
  2>&1 | tee $ROUND_DIR/selection/select.log

# Step 4: Quality filter（PRM 分数会被 apply_quality_filter 内部 max(0,s) 自动 pos-only clip）
log 'Quality filter...'
python3 /workspace/repro/apply_quality_filter.py \
  --input $ROUND_DIR/selection/graded_trajectories_prm_valid.jsonl \
  --output $ROUND_DIR/selection/graded_trajectories_quality_filtered.jsonl \
  --prm-reward-gate 0.5 --report $ROUND_DIR/selection/quality_report.json

# Step 5: kill vLLM
log 'Killing vLLM for training...'
pkill -f 'vllm.entrypoints.openai.api_server' 2>/dev/null || true
sleep 8

# Step 6: compute P_old (R1 from base, no LoRA)
log 'Computing P_old (base model, no LoRA)...'
retry 2 env CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 /workspace/repro/compute_rollout_logprobs.py \
    --graded-file $ROUND_DIR/selection/graded_trajectories_quality_filtered.jsonl \
    --max-seq-length 65536 --rope-scaling-factor 2.0 \
    --output $ROUND_DIR/rollout_logprobs.jsonl

# Step 7: Train PPO + KL + multiplicative PRM β=0.5 (起点 base，无 lora-path)
log 'Training PPO + KL + PRM (multiplicative β=0.5)...'
retry 2 env CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python3 -u $REPO_ROOT/rl/train/train_meeting_grpo_step.py \
    --graded-file $ROUND_DIR/selection/graded_trajectories_quality_filtered.jsonl \
    --model-path Qwen/Qwen3-4B \
    --output-dir $ROUND_DIR/checkpoint \
    --logprobs-file $ROUND_DIR/rollout_logprobs.jsonl \
    --clip-eps 0.2 --kl-beta 0.02 \
    --lr 1e-6 --lora-rank 16 --grad-accum-steps 2 \
    --max-seq-length 65536 --rope-scaling-factor 2.0 \
    --prm-alpha 1.0 --prm-beta 0.5 --prm-mode multiplicative --per-turn-loss

# Step 8: vLLM restart + bench
ensure_vllm || exit 1
log "Hot-loading $NEW_LORA_NAME..."
curl -s -X POST http://127.0.0.1:8021/v1/load_lora_adapter \
  -H 'Content-Type: application/json' \
  -d "{\"lora_name\":\"$NEW_LORA_NAME\",\"lora_path\":\"$ROUND_DIR/checkpoint/lora_adapter\"}"
sleep 5

log 'Benchmarking 5 task x 3 runs...'
cd $REPO_ROOT
retry 2 python3 scripts/benchmark.py \
  --suite 'task_meeting_advisory_stakeholders,task_meeting_council_votes,task_meeting_gov_speaker_summary,task_meeting_tech_action_items,task_meeting_sentiment_analysis' \
  --model $NEW_LORA_NAME \
  --base-url 'http://127.0.0.1:8021/v1' --api-key 'dummy' \
  --judge 'deepseek-chat' \
  --output-dir $BENCH_DIR \
  --runs 3 --no-upload

log '========== R1 PRM ablation DONE =========='
