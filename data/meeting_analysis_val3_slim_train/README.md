# Meeting Analysis Val3 Slim Train

This directory contains the slim training data for the current meeting-analysis
Val3 reproduction setting.

Expected sample file:

```text
claude_code_14_samples.jsonl
```

The file should contain 14 Qwen3-4B OpenClaw rollout samples selected from
non-validation tasks. It is intended to train Val3-relevant capabilities without
using Val3/Val5 grading, expected behavior, gold answers, or judge notes.

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
