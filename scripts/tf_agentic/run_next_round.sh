#!/bin/bash
# ============================================================================
# run_next_round.sh — canonical "continue from the best LoRA" on-policy round.
#
# Recipe from the 2026-06-15 temp=0.3 ablation:
#   pure committee reward (AUTO_W=0) + llm_rubric + base@0.3 ref.
#   DELIBERATION OFF by default. The ablation found NO measurable quality benefit
#   (w5-clean ≈ w6); the only evidence we have is "no effect", so don't carry an
#   unproven knob by default. Its effect on training reward-VARIANCE was never
#   measured — if a variance test later shows it helps, re-enable with DELIBERATE=1.
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
export ROLLOUT_TIMEOUT_MULT="${ROLLOUT_TIMEOUT_MULT:-4.0}"   # guardrail (3)

REPO=/workspace/openclaw-naive-meeting-analysis-github
cd "$REPO"

# ---- knobs (defaults = the w6 winning recipe) ----
RUN_NAME="${RUN_NAME:-committee_w7}"
INIT_ADAPTER="${INIT_ADAPTER:-/workspace/saved_adapters/committee_w6/checkpoint/lora_adapter}"
AUTO_W="${AUTO_W:-0.0}"                                       # pure committee (automated is a weak proxy)
export DELIBERATE="${DELIBERATE:-0}"                          # OFF: no measured quality benefit; re-enable (=1) only if a reward-variance test shows it helps
LR="${LR:-2.0e-5}"; export LR                                 # lowered (continue-train already deep)
BASE_REF="${BASE_REF:-/workspace/saved_adapters/base_ref_temp03.jsonl}"  # base@0.3 anchor
TASKS="${TASKS:-$REPO/data/meeting_analysis_val3_slim_train/val3_plus6_train.json}"
HOLDOUT_JUDGE="${HOLDOUT_JUDGE:-minimax-M3}"                  # guardrail (2): validate with this member held out of nothing? see note

RUN=/tmp/nma_round1/$RUN_NAME
ROLLOUT_DIR=$RUN/rollouts
mkdir -p "$RUN/checkpoint" "$ROLLOUT_DIR"
exec >"$RUN/run.log" 2>&1                                     # log survives ssh drops

echo "[next-round] RUN=$RUN_NAME INIT=$INIT_ADAPTER AUTO_W=$AUTO_W DELIBERATE=$DELIBERATE LR=$LR"
[ -e "$INIT_ADAPTER/adapter_model.safetensors" ] || { echo "INIT_ADAPTER missing: $INIT_ADAPTER"; exit 1; }

echo "[next-round] A: shim serves INIT adapter @ temp=1.0 (fresh on-policy) — guardrail (1)"
pkill -9 -f "tf_shim|benchmark.py" 2>/dev/null || true
sleep 3; rm -f "$RUN/shim.log"
PORT=8021 SHIM_DEFAULT_TEMP=1.0 SHIM_LOG=$RUN/shim.log LORA_ADAPTER=$INIT_ADAPTER python -u /tmp/tf_shim.py &
SHIM=$!
for i in $(seq 1 90); do
  grep -q "shim] ready" "$RUN/shim.log" 2>/dev/null && break
  kill -0 "$SHIM" 2>/dev/null || { echo "SHIM_DIED"; tail -15 "$RUN/shim.log"; exit 1; }
  sleep 3
done
grep -q "shim] ready" "$RUN/shim.log" 2>/dev/null || { echo "SHIM_NOT_READY"; tail -15 "$RUN/shim.log"; exit 1; }
grep "lora=" "$RUN/shim.log" | tail -1

echo "[next-round] B: FRESH rollouts K=4 + flash grade (for automated_score)"
PINCHBENCH_JUDGE_ENSEMBLE=1 python -u train/generate_ledger_online_rollouts.py \
  --tasks-file "$TASKS" --vllm-base-url http://127.0.0.1:8021/v1 --model qwen35-4b \
  --output-dir "$ROLLOUT_DIR" --n-responses 4 --num-workers 1 \
  --judge-model deepseek-v4-flash --judge-base-url https://api.deepseek.com/v1 \
  || echo "[warn] rollout driver non-zero (validating file instead)"
pkill -9 -f tf_shim 2>/dev/null || true; sleep 5
[ -s "$ROLLOUT_DIR/graded_trajectories.jsonl" ] || { echo "NO_ROLLOUTS"; exit 1; }
echo "[next-round] rollouts: $(wc -l < $ROLLOUT_DIR/graded_trajectories.jsonl) rows"

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
