#!/usr/bin/env bash
# Evaluate a trained LoRA on Val3, with an action-item behavior smoke gate.
#
# Flow:
#   1. Cold-start vLLM serving base + the LoRA (named $LORA_NAME).
#   2. Behavior smoke: run ONLY task_meeting_tech_action_items N times. If ANY
#      run scores 0, the LoRA still has the completion-behavior problem -> do
#      NOT proceed to full Val3 (per the "0 => no overall compare" rule).
#   3. Full Val3 (3 tasks x VAL3_RUNS), print per-task mean + overall pct and
#      compare to the base baseline (overall 50.9% / mean 0.5088).
#
# Usage: eval_lora_val3.sh <lora_adapter_path> [lora_name]
set -euo pipefail

LORA_PATH="${1:?usage: eval_lora_val3.sh <lora_adapter_path> [lora_name]}"
LORA_NAME="${2:-eval-lora}"
REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/root/openclaw-venv/bin/python}"
BASE_MODEL="${BASE_MODEL:-/workspace/qwen_models/qwen3-4b}"
PORT="${PORT:-8021}"
BASE_URL="http://127.0.0.1:${PORT}/v1"
SMOKE_RUNS="${SMOKE_RUNS:-5}"
VAL3_RUNS="${VAL3_RUNS:-3}"
MAX_LORA_RANK="${MAX_LORA_RANK:-16}"
OUT_ROOT="${OUT_ROOT:-/workspace/naive_meeting_analysis_runs/eval_${LORA_NAME}_$(date +%Y%m%d_%H%M%S)}"
mkdir -p "$OUT_ROOT"

echo "== eval_lora_val3 =="
echo "  lora:      $LORA_PATH ($LORA_NAME)"
echo "  out:       $OUT_ROOT"

# --- 1. cold-start vLLM base + LoRA ---
echo "[1/3] (re)starting vLLM base + LoRA"
pkill -9 -f vllm.entrypoints 2>/dev/null || true
sleep 5
setsid "$PYTHON_BIN" -m vllm.entrypoints.openai.api_server \
  --model "$BASE_MODEL" --served-model-name Qwen3-4B-base \
  --host 127.0.0.1 --port "$PORT" --max-model-len 40960 --gpu-memory-utilization 0.85 \
  --dtype bfloat16 --trust-remote-code --enable-auto-tool-choice \
  --tool-call-parser hermes --reasoning-parser qwen3 \
  --enable-lora --max-loras 1 --max-lora-rank "$MAX_LORA_RANK" \
  --lora-modules "${LORA_NAME}=${LORA_PATH}" \
  > "$OUT_ROOT/vllm.log" 2>&1 < /dev/null &
for i in $(seq 1 40); do
  curl -s -m5 "$BASE_URL/models" 2>/dev/null | grep -q "$LORA_NAME" && { echo "  vLLM ready (~$((i*8))s)"; break; }
  sleep 8
done
curl -s -m5 "$BASE_URL/models" 2>/dev/null | grep -q "$LORA_NAME" || { echo "ERROR: vLLM did not load LoRA"; tail -20 "$OUT_ROOT/vllm.log"; exit 1; }

run_bench() {  # $1=suite  $2=runs  $3=out_subdir
  MODEL="$LORA_NAME" BASE_URL="$BASE_URL" VAL3_TASKS="$1" RUNS="$2" \
    OUTPUT_DIR="$OUT_ROOT/$3" PYTHON_BIN="$PYTHON_BIN" \
    bash "$REPO_ROOT/scripts/run_val3_bench_isolated.sh" 2>&1 | tail -5
}

parse_scores() {  # $1=result_dir -> prints per-task 0-count + overall
  "$PYTHON_BIN" - "$1" <<'PY'
import json, sys, glob, os
d = sys.argv[1]
files = sorted(glob.glob(os.path.join(d, "*.json")))
if not files:
    print("NO_RESULT_JSON"); sys.exit(0)
data = json.load(open(files[-1]))
per = {}
for t in data.get("tasks", []):
    tid = t["task_id"]
    for r in t.get("grading", {}).get("runs", [{"score": 0}]):
        per.setdefault(tid, []).append(float(r.get("score", 0)))
for tid, ss in per.items():
    zeros = sum(1 for s in ss if s <= 1e-9)
    print(f"TASK {tid} runs={len(ss)} zeros={zeros} mean={sum(ss)/len(ss):.3f} scores={[round(s,3) for s in ss]}")
cat = data.get("category_scores", {})
for k, v in cat.items():
    print(f"OVERALL {k} pct={v.get('pct')} raw={v.get('score')}/{v.get('max_score')}")
PY
}

# --- 2. tech_action_items behavior smoke ---
echo "[2/3] tech_action_items behavior smoke (${SMOKE_RUNS} runs)"
run_bench "task_meeting_tech_action_items" "$SMOKE_RUNS" "smoke"
SMOKE_DIR="$(ls -dt "$OUT_ROOT"/smoke/*/ 2>/dev/null | head -1 || echo "$OUT_ROOT/smoke")"
echo "--- smoke scores ---"
SMOKE_OUT="$(parse_scores "$SMOKE_DIR")"
echo "$SMOKE_OUT"
TECH_ZEROS="$(echo "$SMOKE_OUT" | awk '/tech_action_items/{for(i=1;i<=NF;i++) if($i ~ /^zeros=/){split($i,a,"="); print a[2]}}')"
if [ "${TECH_ZEROS:-1}" != "0" ]; then
  echo "RESULT: tech_action_items still produces ${TECH_ZEROS} zero(s) -> NOT promoting, skipping full Val3."
  exit 2
fi

# --- 3. full Val3 ---
echo "[3/3] full Val3 (${VAL3_RUNS} runs/task)"
run_bench "task_meeting_advisory_stakeholders,task_meeting_gov_speaker_summary,task_meeting_tech_action_items" "$VAL3_RUNS" "val3"
VAL3_DIR="$(ls -dt "$OUT_ROOT"/val3/*/ 2>/dev/null | head -1 || echo "$OUT_ROOT/val3")"
echo "--- Val3 scores (baseline: overall 50.9% / mean 0.5088) ---"
parse_scores "$VAL3_DIR"
