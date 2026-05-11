#!/bin/bash
set -a
source /root/.pinchbench_env
set +a
export PINCHBENCH_FORCE_LOCAL_OPENCLAW=1
export OPENCLAW_HOST=localhost
cd /workspace/openclaw-verl-agent-rl
exec python3 scripts/benchmark.py \
  --model step8-lora \
  --base-url http://127.0.0.1:8000/v1 \
  --api-key dummy \
  --judge deepseek-chat \
  --suite task_meeting_advisory_stakeholders,task_meeting_council_votes,task_meeting_gov_speaker_summary,task_meeting_tech_action_items,task_meeting_sentiment_analysis \
  --timeout-multiplier 3 \
  --no-upload \
  --output-dir /workspace/verl_port/bench/results_step8
