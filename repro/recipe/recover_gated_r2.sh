#!/bin/bash
set -uo pipefail
source ~/.pinchbench_env 2>/dev/null || true
source /root/openclaw-venv/bin/activate
REPO=/workspace/openclaw-naive-meeting-analysis-github
R=/tmp/nma_round1/gated_r2
INIT=/workspace/saved_adapters/base_gated_r1
TASKS=$REPO/data/meeting_analysis_val3_slim_train/val3_plus6_train_mem0_gated_nonasa.json
exec >"$R/recover.log" 2>&1
echo "[recover] drop advisory (all-timeout) from gated_r2 rollouts, inject committee on 6 groups, retrain from base_gated_r1"
python3 -c "
import json
src=\"$R/rollouts/graded_trajectories.jsonl\"; out=\"$R/rollouts/graded_6grp.jsonl\"
rows=[json.loads(l) for l in open(src)]
keep=[r for r in rows if r.get(\"task_id\")!=\"task_meeting_advisory_stakeholders\"]
open(out,\"w\").write(\"\n\".join(json.dumps(r,ensure_ascii=False) for r in keep)+\"\n\")
import collections; print(\"kept\",dict(collections.Counter(r.get(\"task_id\") for r in keep)))
"
JUDGE_LIB_DIR=/tmp/judge_lib COMMITTEE_DIR=$REPO/scripts/tf_agentic RULER_DIR=$REPO/scripts/tf_agentic \
DELIBERATE=1 GRADED_IN=$R/rollouts/graded_6grp.jsonl GRADED_OUT=$R/graded_blend.jsonl \
TASKS_FILE=$TASKS BASE_REF_FILE=/workspace/saved_adapters/base_ref_temp03.jsonl AUTO_W=0.0 \
python3 -u $REPO/scripts/tf_agentic/inject_committee_reward.py || { echo INJECT_FAILED; exit 1; }
RUN_DIR=$R GRADED_NAME=graded_blend.jsonl INIT_LORA=$INIT LR=2.0e-5 bash $REPO/scripts/tf_agentic/retrain_committee.sh
echo "RECOVER_GATED_R2_DONE ckpt=$([ -e $R/checkpoint/lora_adapter/adapter_model.safetensors ] && echo yes || echo NO)"
