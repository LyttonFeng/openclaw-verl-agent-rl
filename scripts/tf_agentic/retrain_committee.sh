#!/bin/bash
# Cold-start GRPO from base qwen3.5-4b, using the RULER-style COMMITTEE reward
# (score field already replaced with committee_score). Reuses round-1e rollouts;
# recomputes logprobs for this exact filtered set. New ckpt dir -- does NOT touch
# the round-1e adapter.
set -uo pipefail
source ~/.pinchbench_env 2>/dev/null || true
source /root/openclaw-venv/bin/activate
cd /workspace/openclaw-naive-meeting-analysis-github

RUN="${RUN_DIR:-/tmp/nma_round1/committee_w1}"
GRADED=$RUN/"${GRADED_NAME:-graded_committee.jsonl}"
MODEL_PATH=/tmp/qwen3.5-4b
CKPT=$RUN/checkpoint
LOGPROBS=$RUN/rollout_logprobs.jsonl
mkdir -p "$CKPT"

echo "[train] filter (drop empty rows + low-variance groups)..."
python - "$GRADED" <<'PYF'
import json, sys
from collections import defaultdict
p = sys.argv[1]
rows = [json.loads(l) for l in open(p) if l.strip()]
g = defaultdict(list)
for r in rows:
    g[r["task_id"]].append(r)
kept = []; de = 0; dg = 0
for t, rs in g.items():
    nz = [r for r in rs if len((r.get("response") or "")) > 0]
    de += len(rs) - len(nz)
    sc = [float(r.get("score", 0)) for r in nz]
    if len(nz) >= 2 and (max(sc) - min(sc)) > 0.02:
        kept.extend(nz)
    else:
        dg += 1; print("  dropped group:", t, "scores=", sc)
open(p, "w").write("\n".join(json.dumps(r) for r in kept) + "\n")
print("[filter] %d -> %d rows (empty %d, dead groups %d)" % (len(rows), len(kept), de, dg))
PYF
[ -s "$GRADED" ] || { echo "ERROR: graded empty"; exit 1; }

echo "[train] kill shim/vllm to free GPU..."
pkill -9 -f "tf_shim|vllm|benchmark.py" 2>/dev/null || true
sleep 6

# INIT_LORA: continue-train from an existing adapter (on-policy iteration). Empty -> cold start.
# When set, old-policy logprobs MUST also be computed WITH that adapter (correct PPO ratio).
INIT="${INIT_LORA:-}"
[ -n "$INIT" ] && echo "[train] continue-train from $INIT (on-policy)" || echo "[train] cold start from base"

echo "[train] step 2: logprobs (old policy${INIT:+ = init LoRA})"
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python train/compute_rollout_logprobs.py \
  --graded-file "$GRADED" --model-path "$MODEL_PATH" --output "$LOGPROBS" \
  ${INIT:+--lora-path "$INIT"} \
  --max-seq-length 32768 --dtype bf16 --max-memory-per-gpu 75GiB || { echo "LOGPROBS FAILED"; exit 1; }

echo "[train] step 3: GRPO (LR 2.5e-5, committee/blend reward)"
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python -u train/train_meeting_grpo_step.py \
  --graded-file "$GRADED" --model-path "$MODEL_PATH" --output-dir "$CKPT" \
  ${INIT:+--lora-path "$INIT"} \
  --lr ${LR:-2.5e-5} --lora-rank 32 --grad-accum-steps 2 --max-seq-length 32768 \
  --prm-alpha 1.0 --prm-beta 0.0 --prm-mode additive \
  --logprobs-file "$LOGPROBS" --clip-eps 0.2 --kl-beta 0.05
echo "RETRAIN_COMMITTEE_DONE ckpt=$CKPT/lora_adapter"
