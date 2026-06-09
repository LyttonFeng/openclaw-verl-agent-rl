# Isolated Val3 Temp-0 Baseline Results

Scope: meeting-analysis Val3 benchmark, isolated OpenClaw runtime, `RUNS=3`, tested agent temperature `0`.

Judge: `deepseek-v4-pro` with judge temperature `0.0`.

## Summary

| Model | advisory stakeholders | speaker summary | action items | Val3 mean |
|---|---:|---:|---:|---:|
| deepseek-v4-pro | 99.33% | 96.99% | 91.48% | 95.93% |
| deepseek-v4-flash | 100.00% | 98.15% | 84.44% | 94.20% |
| Qwen3-4B-base | 43.83% | 48.46% | 66.80% | 53.03% |

Raw aggregate:

| Model | Score | Max | Percent |
|---|---:|---:|---:|
| deepseek-v4-pro | 8.634056 | 9.0 | 95.9% |
| deepseek-v4-flash | 8.477778 | 9.0 | 94.2% |
| Qwen3-4B-base | 4.772889 | 9.0 | 53.0% |

## Per-Task Runs

Std is sample standard deviation across the 3 runs.

| Model | Task | Runs | Mean | Std |
|---|---|---:|---:|---:|
| deepseek-v4-pro | `task_meeting_advisory_stakeholders` | `[1.0000, 0.9800, 1.0000]` | 0.9933 | 0.0115 |
| deepseek-v4-pro | `task_meeting_gov_speaker_summary` | `[0.9097, 1.0000, 1.0000]` | 0.9699 | 0.0521 |
| deepseek-v4-pro | `task_meeting_tech_action_items` | `[0.9417, 0.9027, 0.9000]` | 0.9148 | 0.0233 |
| deepseek-v4-flash | `task_meeting_advisory_stakeholders` | `[1.0000, 1.0000, 1.0000]` | 1.0000 | 0.0000 |
| deepseek-v4-flash | `task_meeting_gov_speaker_summary` | `[0.9444, 1.0000, 1.0000]` | 0.9815 | 0.0321 |
| deepseek-v4-flash | `task_meeting_tech_action_items` | `[0.8760, 0.8240, 0.8333]` | 0.8444 | 0.0277 |
| Qwen3-4B-base | `task_meeting_advisory_stakeholders` | `[0.5083, 0.3307, 0.4760]` | 0.4383 | 0.0946 |
| Qwen3-4B-base | `task_meeting_gov_speaker_summary` | `[0.3772, 0.5256, 0.5511]` | 0.4846 | 0.0939 |
| Qwen3-4B-base | `task_meeting_tech_action_items` | `[0.6520, 0.6760, 0.6760]` | 0.6680 | 0.0139 |

## Result Files

Reference pod outputs:

```text
/workspace/naive_meeting_analysis_runs/dsv4_pro_val3_temp0_3run/val3_runs/0001_deepseek-v4-pro.json
/workspace/naive_meeting_analysis_runs/dsv4_flash_val3_temp0_3run/val3_runs/0001_deepseek-v4-flash.json
/workspace/naive_meeting_analysis_runs/qwen3_4b_base_val3_temp0_shortid/val3_runs/0001_qwen3-4b-base.json
```

## Task Mapping

- `advisory stakeholders`: `task_meeting_advisory_stakeholders`
- `speaker summary`: `task_meeting_gov_speaker_summary`
- `action items`: `task_meeting_tech_action_items`

## Protocol Notes

- Isolated wrapper: `scripts/run_val3_bench_isolated.sh`
- Runs per task: `3`
- Tested agent temperature: `PINCHBENCH_MODEL_TEMPERATURE=0`
- Judge model: `deepseek-v4-pro`
- Judge temperature: `0.0` explicitly set by the OpenAI-compatible judge call.
- Judge cache disabled.
- Parallel judge disabled.
- Upload disabled.

Use the 3-run mean, not a single run, as the baseline reference.
