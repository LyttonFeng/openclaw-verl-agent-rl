#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
# Meeting Analysis Roadmap-PRM Offline GRPO — single round
#
# Pipeline (one invocation = one round):
#   1. Preflight vLLM 80K rope=2 reachable
#   2. Generate N rollouts per task (base 4B for round 1, LoRA otherwise)
#   3. PRM-score trajectories (DSv4-flash judge → prm_turn_scores)
#   4. GRPO step with per-token credit assignment (α=1.0, β=0.1)
#   5. Save LoRA adapter for next round
#
# Design constraints (2026-05-06): explicit constraints:
#   - 起点 = Qwen3-4B base, NOT R5 LoRA
#   - train + rollout 都 rope=2 dynamic YARN factor=2.0
#   - α=1.0, β=0.1, lr=2e-6 (探针轮)
#
# Usage:
#   # First round from base 4B:
#   ROUND_NUM=1 bash rl/train/run_meeting_grpo_prm_round.sh
#
#   # Subsequent round, continuing from previous LoRA:
#   ROUND_NUM=2 PREV_LORA=/workspace/meeting_grpo_prm_v1/round_1/checkpoint/lora_adapter \
#     bash rl/train/run_meeting_grpo_prm_round.sh
#
#   # Override defaults:
#   BASE_DIR=$HOME/grpo_runs   # default /workspace/$EXPERIMENT — change for non-pod machines
#   PRM_BETA=0                 # disable PRM weight (terminal-only ablation; PRM scoring still runs)
#   PRM_MODE=multiplicative    # default 'additive'; switches to (1 + β·prm) advantage scaling
#   TASKS_DIR=...              # default $REPO_ROOT/pinchbench_tasks/meeting_analysis
#
# Environment:
#   DEEPSEEK_API_KEY   — for DSv4 PRM judge AND meeting_reward judge
#   DASHSCOPE_API_KEY  — only if using qwen-plus terminal judge (default DSv4)
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$REPO_ROOT"

# ── Load API keys from env file ────────────────────────────────────────
# Holds DEEPSEEK_API_KEY (and optionally DASHSCOPE_API_KEY).
if [ -f "$HOME/.pinchbench_env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "$HOME/.pinchbench_env"
    set +a
    echo "[env] Sourced $HOME/.pinchbench_env"
fi

# ── Required ────────────────────────────────────────────────────────────────
ROUND_NUM="${ROUND_NUM:?ROUND_NUM must be set, e.g. 1}"
PREV_LORA="${PREV_LORA:-}"   # empty = base 4B (round 1)

# ── Defaults (PRM v1 spec) ──────────────────────────────────────────────
EXPERIMENT="${EXPERIMENT:-meeting_grpo_prm_v1}"
BASE_DIR="${BASE_DIR:-/workspace/$EXPERIMENT}"
ROUND_DIR="$BASE_DIR/round_${ROUND_NUM}"

VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8021/v1}"
SERVED_MODEL="${SERVED_MODEL:-Qwen3-4B}"
LORA_NAME_IN_VLLM="${LORA_NAME_IN_VLLM:-meeting-lora-r5-rope}"   # vLLM-side LoRA module name

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
N_RESPONSES="${N_RESPONSES:-2}"
ROPE_FACTOR="${ROPE_FACTOR:-2.0}"
MAX_SEQ_LEN="${MAX_SEQ_LEN:-81920}"

LR="${LR:-2e-6}"
LORA_RANK="${LORA_RANK:-16}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"
PRM_ALPHA="${PRM_ALPHA:-1.0}"
PRM_BETA="${PRM_BETA:-0.1}"

# Judge for terminal grading (current default = DSv4)
JUDGE_MODEL="${JUDGE_MODEL:-deepseek-chat}"
export MEETING_JUDGE_PROVIDER="${MEETING_JUDGE_PROVIDER:-deepseek}"

# ── Paths ────────────────────────────────────────────────────────────────────
TRAIN_SPLIT="${TRAIN_SPLIT:-$REPO_ROOT/rl/train/meeting_analysis_split.json}"
TASKS_DIR="${TASKS_DIR:-$REPO_ROOT/pinchbench_tasks/meeting_analysis}"
ASSETS_DIR="${ASSETS_DIR:-$REPO_ROOT/assets}"
ROADMAPS_DIR="${ROADMAPS_DIR:-$REPO_ROOT/agent_loop/roadmap_prm/roadmaps}"

mkdir -p "$ROUND_DIR/rollouts" "$ROUND_DIR/checkpoint"

LOG_FILE="$ROUND_DIR/round.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "════════════════════════════════════════════════════════════════"
echo "  Meeting Analysis Roadmap-PRM GRPO — Round $ROUND_NUM"
echo "════════════════════════════════════════════════════════════════"
echo "  Experiment:       $EXPERIMENT"
echo "  Round dir:        $ROUND_DIR"
echo "  vLLM URL:         $VLLM_BASE_URL"
echo "  Served model:     $SERVED_MODEL"
echo "  Prev LoRA:        ${PREV_LORA:-(none, base 4B)}"
echo "  Rope factor:      $ROPE_FACTOR"
echo "  Max seq length:   $MAX_SEQ_LEN"
echo "  N responses:      $N_RESPONSES"
echo "  LR:               $LR"
echo "  PRM α / β:        $PRM_ALPHA / $PRM_BETA"
echo "  Terminal judge:   $JUDGE_MODEL ($MEETING_JUDGE_PROVIDER)"
echo "════════════════════════════════════════════════════════════════"

# ── Preflight ────────────────────────────────────────────────────────────────
echo ""
echo "[preflight] Checking vLLM at $VLLM_BASE_URL ..."
if ! curl -s "$VLLM_BASE_URL/models" | grep -q '"id"'; then
    echo "ERROR: vLLM /models endpoint did not return any models. Is the server up?"
    exit 1
fi
curl -s "$VLLM_BASE_URL/models" | python3 -c "
import json, sys
d = json.load(sys.stdin)
for m in d.get('data', []):
    print(f\"  - {m['id']}\")
"

echo ""
echo "[preflight] Checking DEEPSEEK_API_KEY ..."
if [ -z "${DEEPSEEK_API_KEY:-}" ]; then
    echo "ERROR: DEEPSEEK_API_KEY must be set (used by both PRM judge and terminal judge)."
    exit 1
fi
echo "  ✓ key set (length=${#DEEPSEEK_API_KEY})"

# ── Step 1: Generate rollouts ────────────────────────────────────────────────
# Round 1 → base 4B served name. Round ≥ 2 → LoRA module name in vLLM.
if [ -z "$PREV_LORA" ]; then
    if [ "$ROUND_NUM" != "1" ]; then
        echo "WARN: ROUND_NUM=$ROUND_NUM but PREV_LORA empty — running base 4B anyway."
    fi
    ROLLOUT_MODEL="$SERVED_MODEL"
else
    ROLLOUT_MODEL="$LORA_NAME_IN_VLLM"
    # Best-effort dynamic LoRA load (skip if vLLM doesn't support it)
    echo ""
    echo "[round $ROUND_NUM] Loading LoRA into vLLM: $LORA_NAME_IN_VLLM ← $PREV_LORA"
    curl -s -X POST "$VLLM_BASE_URL/v1/load_lora_adapter" \
        -H "Content-Type: application/json" \
        -d "{\"lora_name\": \"$LORA_NAME_IN_VLLM\", \"lora_path\": \"$PREV_LORA\"}" \
        || echo "  WARN: dynamic LoRA load endpoint failed (continuing — vLLM may already have it loaded)"
    sleep 3
fi

echo ""
echo "[round $ROUND_NUM] Step 1: Generating $N_RESPONSES rollouts per train task (model=$ROLLOUT_MODEL)..."
python3 "$REPO_ROOT/rl/train/generate_meeting_rollouts.py" \
    --split-file "$TRAIN_SPLIT" \
    --split train \
    --tasks-dir "$TASKS_DIR" \
    --assets-dir "$ASSETS_DIR" \
    --vllm-base-url "$VLLM_BASE_URL" \
    --model "$ROLLOUT_MODEL" \
    --n-responses "$N_RESPONSES" \
    --output-dir "$ROUND_DIR/rollouts" \
    --judge-model "$JUDGE_MODEL" \
    --timeout 600 \
    2>&1 | tee "$ROUND_DIR/rollouts/generate.log"

GRADED_FILE="$ROUND_DIR/rollouts/graded_trajectories.jsonl"
N_GRADED=$(wc -l < "$GRADED_FILE" 2>/dev/null || echo "0")
echo "[round $ROUND_NUM] Generated $N_GRADED graded trajectories"

if [ "$N_GRADED" -lt 4 ]; then
    echo "ERROR: too few graded trajectories ($N_GRADED). Aborting."
    exit 1
fi

# ── Step 2: PRM scoring ──────────────────────────────────────────────────────
# Set SKIP_PRM_SCORING=1 to bypass this step entirely (zero PRM judge calls);
# the file is then synthesized with all-zero per-turn scores so step 3 can
# still run. Useful for the strict terminal-only ablation when you want to
# avoid DeepSeek PRM costs altogether.
PRM_GRADED_FILE="$ROUND_DIR/rollouts/graded_trajectories_prm.jsonl"
if [ "${SKIP_PRM_SCORING:-0}" = "1" ]; then
    echo ""
    echo "[round $ROUND_NUM] Step 2: SKIPPED (SKIP_PRM_SCORING=1) — synthesizing all-zero PRM file"
    PYTHONPATH="$REPO_ROOT" python3 -c "
import json, sys
with open('$GRADED_FILE') as fin, open('$PRM_GRADED_FILE', 'w') as fout:
    for line in fin:
        rec = json.loads(line)
        # Stub all-zero PRM scores so downstream training tolerates the file.
        n = max(1, int(rec.get('num_assistant_turns') or 1))
        rec['prm_turn_scores'] = [0] * n
        rec['prm_status'] = 'skipped'
        fout.write(json.dumps(rec) + '\n')
print('SKIP_PRM_SCORING: synthesized all-zero PRM file', file=sys.stderr)
"
    N_PRM=$(wc -l < "$PRM_GRADED_FILE")
    echo "[round $ROUND_NUM] PRM-stub: $N_PRM records → $PRM_GRADED_FILE"
else
    echo ""
    echo "[round $ROUND_NUM] Step 2: PRM-scoring trajectories with DSv4-flash..."
    PYTHONPATH="$REPO_ROOT" python3 -m agent_loop.roadmap_prm.scripts.score_trajectories \
        --graded-file "$GRADED_FILE" \
        --tasks-dir "$TASKS_DIR" \
        --roadmaps-dir "$ROADMAPS_DIR" \
        --output-suffix _prm \
        --max-workers 4 \
        2>&1 | tee "$ROUND_DIR/rollouts/score_trajectories.log"

    if [ ! -f "$PRM_GRADED_FILE" ]; then
        echo "ERROR: PRM-scored file not found: $PRM_GRADED_FILE"
        exit 1
    fi
    N_PRM=$(wc -l < "$PRM_GRADED_FILE")
    echo "[round $ROUND_NUM] PRM-scored: $N_PRM records → $PRM_GRADED_FILE"
fi

# ── Step 3: GRPO training step ───────────────────────────────────────────────
echo ""
echo "[round $ROUND_NUM] Step 3: GRPO training step (α=$PRM_ALPHA, β=$PRM_BETA, lr=$LR, rope=$ROPE_FACTOR)..."
CKPT_DIR="$ROUND_DIR/checkpoint"

LORA_ARG=""
if [ -n "$PREV_LORA" ]; then
    LORA_ARG="--lora-path $PREV_LORA"
fi

CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" python3 -u "$REPO_ROOT/rl/train/train_meeting_grpo_step.py" \
    --graded-file "$PRM_GRADED_FILE" \
    --model-path "$MODEL_PATH" \
    $LORA_ARG \
    --output-dir "$CKPT_DIR" \
    --lr "$LR" \
    --lora-rank "$LORA_RANK" \
    --grad-accum-steps "$GRAD_ACCUM" \
    --max-seq-length "$MAX_SEQ_LEN" \
    --rope-scaling-factor "$ROPE_FACTOR" \
    --prm-alpha "$PRM_ALPHA" \
    --prm-beta "$PRM_BETA" \
    2>&1 | tee "$CKPT_DIR/train.log"

if [ ! -d "$CKPT_DIR/lora_adapter" ]; then
    echo "ERROR: LoRA adapter not saved at $CKPT_DIR/lora_adapter"
    exit 1
fi

echo ""
echo "════════════════════════════════════════════════════════════════"
echo "  Round $ROUND_NUM COMPLETE"
echo "════════════════════════════════════════════════════════════════"
echo "  LoRA:             $CKPT_DIR/lora_adapter"
echo "  PRM-graded:       $PRM_GRADED_FILE"
echo "  Train log:        $CKPT_DIR/train.log"
echo "  Round log:        $LOG_FILE"
echo ""
echo "  For round $((ROUND_NUM + 1)):"
echo "    ROUND_NUM=$((ROUND_NUM + 1)) PREV_LORA=$CKPT_DIR/lora_adapter \\"
echo "      bash rl/train/run_meeting_grpo_prm_round.sh"
echo "════════════════════════════════════════════════════════════════"
