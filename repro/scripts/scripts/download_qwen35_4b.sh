#!/usr/bin/env bash
# Download Qwen3.5-4B to LOCAL disk (/tmp). MUST NOT target MFS /workspace:
# the MFS network disk silently corrupts large safetensors downloads (Errno5
# short writes), which produced a checkpoint that loaded fine but garbled exact
# token sequences (e.g. emitted "meeting/cript.md" instead of
# "meeting-transcript.md") and scored 0 on Val3. `hf download` hash-verifies
# against the Hub, so a clean local pull is the integrity check.

set -euo pipefail

HF_REPO="${HF_REPO:-Qwen/Qwen3.5-4B}"
DEST="${DEST:-/tmp/qwen3.5-4b}"   # LOCAL disk only
HF_BIN="${HF_BIN:-/root/openclaw-venv/bin/hf}"
[ -x "$HF_BIN" ] || HF_BIN="hf"

case "$DEST" in
  /workspace/*) echo "[ERROR] refusing to download to MFS /workspace ($DEST). Use local /tmp." >&2; exit 1;;
esac

echo "Downloading $HF_REPO -> $DEST (local disk)"
"$HF_BIN" download "$HF_REPO" --local-dir "$DEST"
echo "done. size=$(du -sh "$DEST" | cut -f1)"
echo "NOTE: serve with scripts/start_qwen35_vllm.sh (it applies the non-think template patch)."
