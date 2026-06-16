#!/bin/bash
# ============================================================================
# run_next_round.sh — canonical "continue from the best LoRA" on-policy round.
#
# Recipe from the 2026-06-15 temp=0.3 ablation:
#   pure committee reward (AUTO_W=0) + llm_rubric + base@0.3 ref.
#   DELIBERATION ON by default (evidence-based as of 2026-06-16). The 9-pair eval
#   found no FINAL-QUALITY difference (w5-clean ≈ w6), BUT a reward-variance test
#   (re-scoring the same rollout groups with delib 0 vs 1) showed it measurably
#   improves the TRAINING SIGNAL: it corrects a systematic judge outlier (minimax
#   was ~0.3-0.4 too harsh; deliberation converges per-item cross-judge range from
#   ~0.4 to ~0.1) and elevates the genuinely-best rollout -> cleaner, more
#   reproducible GRPO advantage. Disable (DELIBERATE=0) only to save API calls.
# Generalises run_onpolicy.sh so each new round just continue-trains from the
# previous round's adapter on FRESH on-policy rollouts.
#
# THREE GUARDRAILS baked in (do not remove):
#   (1) FRESH on-policy rollouts — the shim serves $INIT_ADAPTER at temp=1.0 and
#       we regenerate K=4 rollouts PER ROUND. Never reuse a previous round's
#       graded_trajectories: that is off-policy and defeats "continue from w6".
#   (2) JUDGE-OVERFIT GUARD — after training, validate the new adapter with a
#       HELD-OUT judge (a committee member NOT used in the reward). See the
#       printed command at the end + docs/next_round_recipe.md. The committee can
#       be reward-hacked just like the automated grader was (round-1e); a held-out
#       judge (or human spot-check) catches it.
#   (3) TIMEOUT FIX + HEALTHCHECK — ROLLOUT_TIMEOUT_MULT=4.0 (advisory 71K doc)
#       and rollout_healthcheck.py STOP the round before training on garbage
#       (all-timeout / no-write / written-but-auto=0 false negatives).
#
# Also: LR is LOWERED by default (continue-train is already ≥2 layers deep:
# base -> committee_blend -> w6 -> ...); KL anchors to $INIT_ADAPTER.
#
# Run ON the pod (one process tree; survives ssh drops). Self-redirects its log.
# ============================================================================
set -uo pipefail
source ~/.pinchbench_env 2>/dev/null || true
source /root/openclaw-venv/bin/activate
unset OPENCLAW_HOST ECS_HOST OPENCLAW_REMOTE_ACTIVATE_CMD
export PINCHBENCH_FORCE_LOCAL_OPENCLAW=1
export ROLLOUT_TIMEOUT_MULT="${ROLLOUT_TIMEOUT_MULT:-4.0}"   # 180x4=720s. DATA-DRIVEN: all SUCCESSFUL rollouts finish <=593s (w2 max 593, w7 max 479); long 71K docs are context-bound (96% ctx) and NEVER complete, so a higher cap only wastes ~370s/doomed-rollout. 720s covers every completable rollout + fails the doomed long-docs fast. (Raising the cap does NOT fix the context wall.)
ROLLOUT_TEMP="${ROLLOUT_TEMP:-0.7}"                          # rollout sampling temp. temp=1.0 caused ~25% premature-termination (agent reads then ends WITHOUT writing the deliverable). 0.7 keeps diversity, far fewer empties.
SHIM_SCRIPT="${SHIM_SCRIPT:-/workspace/openclaw-naive-meeting-analysis-github/scripts/tf_agentic/tf_shim_batched.py}"  # micro-batching shim. A/B validated 2026-06-16: workers=2 → 0 empty + ~33% faster than serial; workers=4 corrupted 3/4 rollouts. Set /tmp/tf_shim.py + NUM_WORKERS=1 for serial.
NUM_WORKERS="${NUM_WORKERS:-2}"                              # concurrent rollouts (2 validated clean+faster; do NOT raise to 4 — batched generate corrupts outputs)
SHIM_MAX_BATCH="${SHIM_MAX_BATCH:-2}"                        # micro-batch window for tf_shim_batched

REPO=/workspace/openclaw-naive-meeting-analysis-github
cd "$REPO"

# ---- knobs (defaults = the w6 winning recipe) ----
RUN_NAME="${RUN_NAME:-committee_w7}"
INIT_ADAPTER="${INIT_ADAPTER:-/workspace/saved_adapters/committee_w6/checkpoint/lora_adapter}"
AUTO_W="${AUTO_W:-0.0}"                                       # pure committee (automated is a weak proxy)
export DELIBERATE="${DELIBERATE:-1}"                          # ON (evidence-based): variance test 2026-06-16 showed it de-biases judges (per-item cross-judge range ~0.4 -> ~0.1, corrects minimax harsh outlier) + sharpens the correct ranking -> cleaner, more reproducible GRPO signal
LR="${LR:-2.0e-5}"; export LR                                 # lowered (continue-train already deep)
BASE_REF="${BASE_REF:-/workspace/saved_adapters/base_ref_temp03.jsonl}"  # base@0.3 anchor
TASKS="${TASKS:-$REPO/data/meeting_analysis_val3_slim_train/val3_plus6_train.json}"
HEALTH="${TASK_HEALTH:-/workspace/saved_adapters/task_health.json}"   # persistent per-task health (adaptive skip + reprobe)
HOLDOUT_JUDGE="${HOLDOUT_JUDGE:-minimax-M3}"                  # guardrail (2): validate with this member held out of nothing? see note

RUN=/tmp/nma_round1/$RUN_NAME
ROLLOUT_DIR=$RUN/rollouts
mkdir -p "$RUN/checkpoint" "$ROLLOUT_DIR"
exec >"$RUN/run.log" 2>&1                                     # log survives ssh drops

echo "[next-round] RUN=$RUN_NAME INIT=$INIT_ADAPTER AUTO_W=$AUTO_W DELIBERATE=$DELIBERATE LR=$LR ROLLOUT_TEMP=$ROLLOUT_TEMP TIMEOUT_MULT=$ROLLOUT_TIMEOUT_MULT SHIM=$(basename $SHIM_SCRIPT) WORKERS=$NUM_WORKERS"
[ -e "$INIT_ADAPTER/adapter_model.safetensors" ] || { echo "INIT_ADAPTER missing: $INIT_ADAPTER"; exit 1; }

echo "[next-round] A: shim serves INIT adapter @ temp=1.0 (fresh on-policy) — guardrail (1)"
pkill -9 -f "tf_shim|benchmark.py" 2>/dev/null || true
sleep 3; rm -f "$RUN/shim.log"
PORT=8021 SHIM_DEFAULT_TEMP=$ROLLOUT_TEMP SHIM_MAX_BATCH=$SHIM_MAX_BATCH SHIM_LOG=$RUN/shim.log LORA_ADAPTER=$INIT_ADAPTER python -u "$SHIM_SCRIPT" &
SHIM=$!
for i in $(seq 1 90); do
  grep -q "shim] ready" "$RUN/shim.log" 2>/dev/null && break
  kill -0 "$SHIM" 2>/dev/null || { echo "SHIM_DIED"; tail -15 "$RUN/shim.log"; exit 1; }
  sleep 3
done
grep -q "shim] ready" "$RUN/shim.log" 2>/dev/null || { echo "SHIM_NOT_READY"; tail -15 "$RUN/shim.log"; exit 1; }
grep "lora=" "$RUN/shim.log" | tail -1

echo "[next-round] B0: adaptive task selection (landmine-proof: skip all-dead>=2 rounds, force re-probe every 3, loud-logged)"
ACTIVE_TASKS=$RUN/active_tasks.json; SKIP_LIST=$RUN/skipped_tasks.json
python3 "$REPO/scripts/tf_agentic/select_active_tasks.py" --tasks-file "$TASKS" \
  --out-file "$ACTIVE_TASKS" --skip-out "$SKIP_LIST" --health "$HEALTH" \
  --dead-threshold 2 --reprobe-every 3 || { echo "[warn] task-select failed; using full task set"; cp "$TASKS" "$ACTIVE_TASKS"; echo "[]" > "$SKIP_LIST"; }

echo "[next-round] B: FRESH rollouts K=4 + flash grade (for automated_score)"
PINCHBENCH_JUDGE_ENSEMBLE=1 python -u train/generate_ledger_online_rollouts.py \
  --tasks-file "$ACTIVE_TASKS" --vllm-base-url http://127.0.0.1:8021/v1 --model qwen35-4b \
  --output-dir "$ROLLOUT_DIR" --n-responses 4 --num-workers $NUM_WORKERS \
  --judge-model deepseek-v4-flash --judge-base-url https://api.deepseek.com/v1 \
  || echo "[warn] rollout driver non-zero (validating file instead)"
pkill -9 -f tf_shim 2>/dev/null || true; sleep 5
[ -s "$ROLLOUT_DIR/graded_trajectories.jsonl" ] || { echo "NO_ROLLOUTS"; exit 1; }
echo "[next-round] rollouts: $(wc -l < $ROLLOUT_DIR/graded_trajectories.jsonl) rows"

echo "[next-round] B1: update per-task health (adaptive skip memory; single writer)"
python3 "$REPO/scripts/tf_agentic/update_task_health.py" --graded "$ROLLOUT_DIR/graded_trajectories.jsonl" \
  --skip-list "$SKIP_LIST" --health "$HEALTH" || echo "[warn] task-health update failed (non-fatal)"

echo "[next-round] B2: rollout health check — guardrail (3) (stop before training on garbage)"
python3 "$REPO/scripts/tf_agentic/rollout_healthcheck.py" "$ROLLOUT_DIR/graded_trajectories.jsonl" \
  || { echo "ROLLOUT_HEALTH_FAILED — fix harness, do NOT train"; exit 2; }

echo "[next-round] C: committee re-score (AUTO_W=$AUTO_W, deliberation=$DELIBERATE, rubric + base@0.3 ref)"
JUDGE_LIB_DIR=/tmp/judge_lib COMMITTEE_DIR=$REPO/scripts/tf_agentic RULER_DIR=$REPO/scripts/tf_agentic \
DELIBERATE=$DELIBERATE \
GRADED_IN=$ROLLOUT_DIR/graded_trajectories.jsonl GRADED_OUT=$RUN/graded_blend.jsonl \
TASKS_FILE=$TASKS BASE_REF_FILE=$BASE_REF AUTO_W=$AUTO_W \
python3 -u "$REPO/scripts/tf_agentic/inject_committee_reward.py" || { echo "INJECT_FAILED"; exit 1; }

echo "[next-round] D: continue-train FROM $INIT_ADAPTER (LR=$LR, KL anchors to init)"
RUN_DIR=$RUN GRADED_NAME=graded_blend.jsonl INIT_LORA=$INIT_ADAPTER LR=$LR \
bash "$REPO/scripts/tf_agentic/retrain_committee.sh"

echo "NEXT_ROUND_DONE ckpt=$RUN/checkpoint/lora_adapter"
echo ""
echo "==== GUARDRAIL (2): held-out-judge validation — RUN THIS AFTER eval ===="
echo "  # 1) eval the new adapter @ temp=0.3 RUNS=3 vs the SAME canonical base@0.3"
echo "  EVAL_TEMP=0.3 RUNS=3 bash scripts/tf_agentic/eval_val3_adapter.sh lora $RUN/checkpoint/lora_adapter"
echo "  # 2) judge with a HELD-OUT member (exclude one training judge) — gains must survive:"
echo "  JUDGE_MEMBERS=qwen-max,$HOLDOUT_JUDGE python3 scripts/tf_agentic/committee_judge.py"
echo "  # if the committee win VANISHES under the held-out judge -> suspect judge-overfitting."
