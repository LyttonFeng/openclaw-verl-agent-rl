#!/usr/bin/env bash
# Bring a fresh GPU pod to a runnable training/bench state for this repo.
#
# Goal: 10-15 min from clean Ubuntu/Debian pod with CUDA 12.x to:
#   - python3.12 venv with vllm 0.10.2 / transformers 4.57.1 / peft / torch 2.8
#   - openclaw@2026.4.5 globally installed via npm, with PATCH-B applied
#   - HF cache populated with Qwen3-4B
#   - ~/.pinchbench_env present (you must fill in the DeepSeek key)
#
# Usage:
#   bash scripts/setup_pod.sh                         # default everything
#   SKIP_APT=1 bash scripts/setup_pod.sh              # skip system pkg install
#   SKIP_HF_DOWNLOAD=1 bash scripts/setup_pod.sh      # skip Qwen3 download
#   RSYNC_FROM=user@old-pod:/workspace/hf_cache bash scripts/setup_pod.sh
#
# Idempotent: re-runs are safe — each step short-circuits if already done.

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}"
PY="${PY:-python3.12}"
VENV="${VENV:-$REPO_ROOT/.venv}"
HF_HOME="${HF_HOME:-/workspace/hf_cache}"
OC_VERSION="${OC_VERSION:-2026.4.5}"
QWEN3_MODEL_ID="${QWEN3_MODEL_ID:-Qwen/Qwen3-4B}"
ENV_FILE="${ENV_FILE:-$HOME/.pinchbench_env}"

log() { echo "[$(date +%T)] $*"; }

# ---------------------------------------------------------------------------
# 1. system packages (apt)
# ---------------------------------------------------------------------------
if [ "${SKIP_APT:-0}" != "1" ]; then
  log "step 1/7  system packages"
  if command -v apt-get >/dev/null 2>&1; then
    apt-get update -y
    apt-get install -y --no-install-recommends \
      build-essential git curl ca-certificates rsync \
      software-properties-common
    # python3.12
    if ! command -v "$PY" >/dev/null 2>&1; then
      add-apt-repository -y ppa:deadsnakes/ppa || true
      apt-get update -y
      apt-get install -y python3.12 python3.12-venv python3.12-dev
    fi
    # node 20.x (openclaw needs >= 20)
    if ! command -v node >/dev/null 2>&1; then
      curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
      apt-get install -y nodejs
    fi
  else
    log "  apt-get not found — skipping (assume system pkgs already present)"
  fi
else
  log "step 1/7  system packages  SKIPPED"
fi

# ---------------------------------------------------------------------------
# 2. python venv + pip deps
# ---------------------------------------------------------------------------
log "step 2/7  python venv at $VENV"
if [ ! -d "$VENV" ]; then
  "$PY" -m venv "$VENV"
fi
# shellcheck disable=SC1091
source "$VENV/bin/activate"
python -m pip install --upgrade pip wheel setuptools
pip install -r "$REPO_ROOT/requirements.txt"
# editable install for our package layout
pip install -e "$REPO_ROOT" || true

# ---------------------------------------------------------------------------
# 3. openclaw (npm global) + node symlink
# ---------------------------------------------------------------------------
log "step 3/7  openclaw@$OC_VERSION"
if command -v openclaw >/dev/null 2>&1 && openclaw --version 2>/dev/null | grep -q "$OC_VERSION"; then
  log "  openclaw $OC_VERSION already installed"
else
  # Install on LOCAL disk, NOT under /workspace (network FS makes this take ~50min)
  npm install -g "openclaw@$OC_VERSION"
fi
# nvm-installed node may not be on the openclaw shebang's PATH
if [ ! -e /usr/local/bin/node ]; then
  ln -sf "$(which node)" /usr/local/bin/node
fi
openclaw --version || { log "ERROR: openclaw not callable"; exit 1; }

# ---------------------------------------------------------------------------
# 4. PATCH-B (OC + vLLM hermes parser fallback)
# ---------------------------------------------------------------------------
log "step 4/7  applying PATCH-B to openclaw pi-ai provider"
bash "$REPO_ROOT/scripts/sft/apply_oc_hermes_patch.sh"

# ---------------------------------------------------------------------------
# 5. ~/.pinchbench_env  (DeepSeek judge API key)
# ---------------------------------------------------------------------------
log "step 5/7  $ENV_FILE"
if [ ! -f "$ENV_FILE" ]; then
  cat > "$ENV_FILE" <<'EOF'
# Fill in your DeepSeek API key — used as the meeting-analysis judge.
export DEEPSEEK_API_KEY=REPLACE_ME
EOF
  chmod 600 "$ENV_FILE"
  log "  WROTE TEMPLATE — edit $ENV_FILE and replace DEEPSEEK_API_KEY"
else
  log "  $ENV_FILE already exists, leaving untouched"
fi

# ---------------------------------------------------------------------------
# 6. HuggingFace cache (Qwen3-4B ~8GB)
# ---------------------------------------------------------------------------
log "step 6/7  HF cache at $HF_HOME"
mkdir -p "$HF_HOME"
export HF_HOME

if [ -n "${RSYNC_FROM:-}" ]; then
  log "  rsync from $RSYNC_FROM -> $HF_HOME"
  rsync -aP --partial "$RSYNC_FROM/" "$HF_HOME/"
elif [ "${SKIP_HF_DOWNLOAD:-0}" != "1" ]; then
  if [ -d "$HF_HOME/hub/models--${QWEN3_MODEL_ID/\//--}" ]; then
    log "  $QWEN3_MODEL_ID already in cache"
  else
    pip install -q "huggingface_hub[cli]"
    huggingface-cli download "$QWEN3_MODEL_ID" \
      --cache-dir "$HF_HOME/hub" \
      --max-workers 8
  fi
else
  log "  HF download skipped (SKIP_HF_DOWNLOAD=1)"
fi

# ---------------------------------------------------------------------------
# 7. verification
# ---------------------------------------------------------------------------
log "step 7/7  verification"
python - <<'PYEOF'
import importlib, sys
mods = [("torch", "2.8"), ("transformers", "4.57"), ("peft", None),
        ("vllm", "0.10.2"), ("accelerate", None)]
for name, want in mods:
    try:
        m = importlib.import_module(name)
        v = getattr(m, "__version__", "?")
        ok = (want is None) or v.startswith(want)
        print(f"  {'OK ' if ok else 'WARN'} {name:14s} {v}{f'  (want {want})' if not ok else ''}")
    except Exception as e:
        print(f"  MISS {name}: {e}")
        sys.exit(1)
PYEOF

grep -q 'PATCH-B' /usr/local/lib/node_modules/openclaw/node_modules/@mariozechner/pi-ai/dist/providers/openai-completions.js \
  && log "  OK   PATCH-B in place" \
  || { log "  FAIL PATCH-B not found in OC provider"; exit 1; }

cat <<'EOF'

═══════════════════════════════════════════════════════════════════════
Pod setup complete.

Next:
  1. source ~/.pinchbench_env      (after filling in DEEPSEEK_API_KEY)
  2. source .venv/bin/activate
  3. bash scripts/sft/bench_base.sh /workspace/verl_port/bench_smoke
═══════════════════════════════════════════════════════════════════════
EOF
