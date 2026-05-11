#!/bin/bash
# Bench base Qwen3-4B via JiuwenClaw stack on 5 meeting analysis test tasks.
# Requires stack already running (start_jw_pod.sh).

set -a
source /root/.pinchbench_env
set +a

# Resolve ws url from stack meta
META=/tmp/pinchbench_jw/logs/stack_latest.env
if [[ -f "$META" ]]; then
  source "$META"
fi
WS_URL="${WS_URL:-ws://127.0.0.1:611/ws}"

OUT_DIR="/workspace/verl_port/bench/jiuwen_base_4b"
mkdir -p "$OUT_DIR"

cd /root/jiuwen_work/jiuwenclaw
exec uv run --frozen python /root/jiuwen_work/pinchbench/scripts/run_pinchbench_jiuwenclaw.py \
  --skill-root /workspace/openclaw-verl-agent-rl \
  --ws-url "$WS_URL" \
  --suite task_meeting_advisory_stakeholders,task_meeting_council_votes,task_meeting_gov_speaker_summary,task_meeting_tech_action_items,task_meeting_sentiment_analysis \
  --judge-model deepseek-chat \
  --judge-backend api \
  --output-dir "$OUT_DIR"
