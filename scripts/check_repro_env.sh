#!/usr/bin/env bash
# Reproduction preflight for naive_meeting_analysis.
#
# This script does not start vLLM, run rollouts, or call the judge. It only
# checks that the local environment has the pieces needed to run the quickstart.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO_ROOT"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VLLM_BASE_URL="${VLLM_BASE_URL:-http://127.0.0.1:8021/v1}"
ROLLOUT_MODEL="${ROLLOUT_MODEL:-Qwen3-4B}"
CHECK_VLLM="${CHECK_VLLM:-1}"
CHECK_OPENCLAW_PATCH="${CHECK_OPENCLAW_PATCH:-1}"
OC_PROVIDER_JS="${OC_PROVIDER_JS:-/usr/local/lib/node_modules/openclaw/node_modules/@mariozechner/pi-ai/dist/providers/openai-completions.js}"

if [ -f "$HOME/.pinchbench_env" ]; then
  set -a
  # shellcheck disable=SC1090
  source "$HOME/.pinchbench_env"
  set +a
fi

FAILURES=0
WARNINGS=0

ok() {
  printf '[OK]   %s\n' "$1"
}

warn() {
  WARNINGS=$((WARNINGS + 1))
  printf '[WARN] %s\n' "$1"
}

fail() {
  FAILURES=$((FAILURES + 1))
  printf '[FAIL] %s\n' "$1"
}

check_file() {
  local path="$1"
  if [ -e "$path" ]; then
    ok "$path exists"
  else
    fail "$path missing"
  fi
}

echo "== naive_meeting_analysis reproducibility preflight =="
echo "repo:        $REPO_ROOT"
echo "python:      $PYTHON_BIN"
echo "vllm url:    $VLLM_BASE_URL"
echo "model name:  $ROLLOUT_MODEL"
echo

echo "[1/7] Required files"
for path in \
  README.md \
  env/requirements.txt \
  env/training_environment.md \
  env/benchmark_environment.md \
  scripts/start_qwen3_vllm.sh \
  scripts/apply_oc_hermes_patch.sh \
  scripts/run_val5_bench_isolated.sh \
  train/run_naive_ppo_round.sh \
  train/generate_meeting_rollouts.py \
  train/select_grpo_samples.py \
  train/compute_rollout_logprobs.py \
  train/train_meeting_grpo_step.py \
  data/train/meeting_analysis_all_samples_split.json \
  results/isolated_val5_temp0_baseline_results.md
do
  check_file "$path"
done

echo
echo "[2/7] Python and packages"
if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  "$PYTHON_BIN" --version
  ok "python executable available"
else
  fail "python executable not found: $PYTHON_BIN"
fi

if command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  "$PYTHON_BIN" - <<'PY' || FAILURES=$((FAILURES + 1))
import importlib
packages = ["yaml", "torch", "transformers", "peft", "accelerate", "vllm", "huggingface_hub"]
missing = []
for name in packages:
    try:
        mod = importlib.import_module(name)
        version = getattr(mod, "__version__", "unknown")
        print(f"[OK]   python package {name}={version}")
    except Exception as exc:
        missing.append((name, str(exc)))
for name, err in missing:
    print(f"[FAIL] python package {name} unavailable: {err}")
if missing:
    raise SystemExit(1)
PY
fi

echo
echo "[3/7] OpenClaw"
if command -v openclaw >/dev/null 2>&1; then
  openclaw --version || fail "openclaw --version failed"
  ok "openclaw command available"
else
  fail "openclaw command not found on PATH"
fi

if [ "$CHECK_OPENCLAW_PATCH" = "1" ]; then
  if [ -f "$OC_PROVIDER_JS" ]; then
    if grep -q "PATCH-B" "$OC_PROVIDER_JS"; then
      ok "OpenClaw hermes PATCH-B is applied"
    else
      fail "OpenClaw hermes PATCH-B not applied; run scripts/apply_oc_hermes_patch.sh"
    fi
  else
    fail "OpenClaw provider JS not found: $OC_PROVIDER_JS"
  fi
else
  warn "OpenClaw hermes PATCH-B check skipped"
fi

echo
echo "[4/7] Secrets"
if [ -n "${DEEPSEEK_API_KEY:-}" ]; then
  ok "DEEPSEEK_API_KEY is set"
else
  fail "DEEPSEEK_API_KEY is not set; add it to ~/.pinchbench_env or the environment"
fi

echo
echo "[5/7] vLLM endpoint"
if [ "$CHECK_VLLM" = "1" ]; then
  if command -v curl >/dev/null 2>&1; then
    MODELS_JSON="$(curl -sSf "$VLLM_BASE_URL/models" 2>/dev/null || true)"
    if [ -n "$MODELS_JSON" ]; then
      printf '%s\n' "$MODELS_JSON" | grep -q "$ROLLOUT_MODEL" \
        && ok "vLLM endpoint serves $ROLLOUT_MODEL" \
        || fail "vLLM endpoint responded, but served model name did not include $ROLLOUT_MODEL"
    else
      fail "vLLM endpoint not reachable: $VLLM_BASE_URL/models"
    fi
  else
    fail "curl command not found"
  fi
else
  warn "vLLM endpoint check skipped"
fi

echo
echo "[6/7] Task data"
"$PYTHON_BIN" - <<'PY' || FAILURES=$((FAILURES + 1))
import json
import sys
from pathlib import Path

root = Path(".")
sys.path.insert(0, str((root / "scripts").resolve()))
from lib_tasks import TaskLoader

split = json.loads((root / "data/train/meeting_analysis_all_samples_split.json").read_text())
train = TaskLoader(root / "data/train/tasks").load_all_tasks()
val5 = TaskLoader(root / "data/eval/val5").load_all_tasks()
missing = []

for task in train + val5:
    for wf in task.workspace_files:
        src = wf.get("source") or wf.get("src") or wf.get("path")
        if not src:
            continue
        candidates = [root / "data/eval/assets" / src, task.file_path.parent / src]
        if not any(p.exists() for p in candidates):
            missing.append((task.task_id, src))

checks = [
    ("split train ids", len(split.get("train", [])), 28),
    ("loaded train tasks", len(train), 28),
    ("loaded val5 tasks", len(val5), 5),
]
for label, got, want in checks:
    if got == want:
        print(f"[OK]   {label}: {got}")
    else:
        print(f"[FAIL] {label}: got {got}, want {want}")
        raise SystemExit(1)

if missing:
    for task_id, src in missing[:20]:
        print(f"[FAIL] missing workspace file for {task_id}: {src}")
    raise SystemExit(1)
print("[OK]   workspace file references resolved")
PY

echo
echo "[7/7] Script syntax"
for path in scripts/start_qwen3_vllm.sh scripts/apply_oc_hermes_patch.sh scripts/run_val5_bench_isolated.sh train/run_naive_ppo_round.sh; do
  if bash -n "$path"; then
    ok "$path syntax"
  else
    fail "$path syntax"
  fi
done

echo
echo "== summary =="
echo "failures: $FAILURES"
echo "warnings: $WARNINGS"

if [ "$FAILURES" -ne 0 ]; then
  echo "Preflight failed."
  exit 1
fi

echo "Preflight passed."
