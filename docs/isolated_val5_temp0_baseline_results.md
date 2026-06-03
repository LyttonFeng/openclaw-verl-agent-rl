# Isolated Val5 Temp-0 Baseline Results

Scope: meeting-analysis Val5 benchmark, isolated OpenClaw runtime, `RUNS=3`, tested model temperature `0`.

Judge: `deepseek-v4-pro`.

| Model | advisory | council votes | speaker summary | action items | sentiment | Total |
|---|---:|---:|---:|---:|---:|---:|
| deepseek-v4-pro | 100.0% | 72.3% | 98.8% | 92.7% | 87.2% | 90.2% |
| deepseek-v4-flash | 98.3% | 77.7% | 81.0% | 79.3% | 94.4% | 86.1% |
| qwen3.5-4b | 89.7% | 34.6% | 90.6% | 74.3% | 90.5% | 75.9% |
| qwen3-4b | 43.9% | 24.6% | 42.3% | 60.6% | 53.7% | 45.0% |

Task mapping:

- `advisory`: `task_meeting_advisory_stakeholders`
- `council votes`: `task_meeting_council_votes`
- `speaker summary`: `task_meeting_gov_speaker_summary`
- `action items`: `task_meeting_tech_action_items`
- `sentiment`: `task_meeting_sentiment_analysis`

Protocol notes:

- Isolated wrapper: `scripts/run_val5_bench_isolated.sh`
- Tested model temperature: `PINCHBENCH_MODEL_TEMPERATURE=0`
- Judge cache disabled.
- Parallel judge disabled.
- Upload disabled.
