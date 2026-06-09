# Meeting Analysis Val3 Slim Train

This directory contains the slim training data for the current meeting-analysis
Val3 reproduction setting.

Primary online RL task registry:

```text
claude_code_14_tasks.json
```

Each entry is an executable RL task seed with:

- `workspace_files`
- `prompt`
- `expected_output_file`
- `grading.grade_function`
- `grading.llm_rubric`
- `reward_contract`
- `rl_grouping`

The online RL driver consumes this file:

```bash
python rl/train/generate_ledger_online_rollouts.py \
  --tasks-file data/meeting_analysis_val3_slim_train/claude_code_14_tasks.json \
  --vllm-base-url http://127.0.0.1:8021/v1 \
  --model Qwen3-4B \
  --output-dir /path/to/run \
  --n-responses 4
```

Offline grouped rollout data:

```text
claude_code_14_grpo.jsonl
```

This file contains 14 groups with 4 responses per group, each with scalar
scores and reward breakdowns. It is useful for offline GRPO sanity checks.

Reference best-response sample file:

```text
claude_code_14_samples.jsonl
```

This file contains the best response per task. It is mainly for inspection and
visualization, not the main online RL entry point.

## Expected Schema

Each line is one JSON object:

```json
{
  "id": "claude_code_val3_0001",
  "source_task_id": "task_meeting_gov_qa_extract",
  "target_capability": "speaker_claim_ledger",
  "variant": "evidence_ledger",
  "prompt": "...",
  "response": "...",
  "score": 1.0,
  "metadata": {
    "meeting_family": "gov",
    "uses_val_task": false,
    "notes": "constructed from train task only"
  }
}
```

## Capability Coverage

| Capability | Samples | Primary Target |
| --- | ---: | --- |
| `stakeholder_evidence_ledger` | 4 | `task_meeting_advisory_stakeholders` |
| `speaker_claim_ledger` | 4 | `task_meeting_gov_speaker_summary` |
| `commitment_evidence_ledger` | 4 | `task_meeting_tech_action_items` |
| `decision_evidence_ledger` | 2 | action / decision tracking support |

## Validation Boundary

All samples should satisfy:

- `metadata.uses_val_task == false`
- no Val3/Val5 grading code
- no expected behavior / gold answer / judge-note leakage
- transcript-grounded response with evidence quotes or source spans

This dataset is designed for the fast Val3 diagnostic suite:

- `task_meeting_advisory_stakeholders`
- `task_meeting_gov_speaker_summary`
- `task_meeting_tech_action_items`
