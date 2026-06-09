# Data

This directory holds meeting-analysis training and evaluation data used by the
RL and swarm-policy reproduction workflow.

The original split definition lives in
`rl/train/meeting_analysis_split.json` (23 train tasks / 5 held-out tasks across
4 real meeting transcripts). For the current fast reproduction setting, we use a
smaller Val3 diagnostic suite instead of the full Val5 suite.

## Current Reproduction Setting

### Training Data

Primary slim training data:

```text
data/meeting_analysis_val3_slim_train/claude_code_14_samples.jsonl
```

Visualization:

```text
docs/data_visualizations/meeting_val3_training_data_14_plus_12.html
```

Online RL task registry:

```text
data/meeting_analysis_val3_slim_train/claude_code_14_tasks.json
```

This is the main entry point for the 14-task online RL smoke run. Each task
contains `workspace_files`, `prompt`, `expected_output_file`, embedded
`grade_function`, `llm_rubric`, `reward_contract`, and `rl_grouping`.

Offline grouped rollout data:

```text
data/meeting_analysis_val3_slim_train/claude_code_14_grpo.jsonl
```

This contains 14 groups with 4 responses per group and can be used for offline
GRPO sanity checks or seed data conversion.

Reference best-response samples:

```text
data/meeting_analysis_val3_slim_train/claude_code_14_samples.jsonl
```

This file contains the best response per task selected from Qwen3-4B OpenClaw
rollouts. It is mainly used for inspection and visualization; the online RL
path should use `claude_code_14_tasks.json`.

Coverage:

| Capability | Samples | Primary Target |
| --- | ---: | --- |
| `stakeholder_evidence_ledger` | 4 | `task_meeting_advisory_stakeholders` |
| `speaker_claim_ledger` | 4 | `task_meeting_gov_speaker_summary` |
| `commitment_evidence_ledger` | 4 | `task_meeting_tech_action_items` |
| `decision_evidence_ledger` | 2 | action / decision tracking support |

Data constraints:

- `uses_val_task` must be `false` for every sample.
- Samples must not use Val3/Val5 grading, expected behavior, gold answers, or
  judge notes.
- Responses should contain transcript-grounded evidence, quotes, or source
  spans.
- The current 14-sample file reports score range `0.58-0.96`, mean `0.71`.

### Evaluation Data

The fast diagnostic evaluation suite is Val3:

| Suite | Task ID | Capability |
| --- | --- | --- |
| Val3 | `task_meeting_advisory_stakeholders` | stakeholder / entity / stance / evidence |
| Val3 | `task_meeting_gov_speaker_summary` | speaker attribution / quote-backed summary |
| Val3 | `task_meeting_tech_action_items` | owner / action / deadline / dependency |

The previous Val5 suite is still useful for full reporting, but it is no longer
the default fast iteration target. The two tasks excluded from Val3 are:

| Task ID | Reason |
| --- | --- |
| `task_meeting_council_votes` | Requires stronger vote-recall scaffold / skill / swarm support than pure single-agent RL. |
| `task_meeting_sentiment_analysis` | Highest cross-round variance; less suitable as a quick training-trend signal. |

## Full Meeting-Analysis Split

The full split remains:

- 23 train tasks.
- 5 held-out Val5 tasks.
- 4 transcript families: NTIA advisory, Tampa council, NASA government hearing,
  and GitLab product marketing meeting.

To regenerate the parquet from task definitions:

```bash
python rl/train/build_meeting_analysis_prompts.py \
    --tasks-dir pinchbench_tasks/meeting_analysis \
    --split-file rl/train/meeting_analysis_split.json \
    --output-dir data/meeting_prompts
```

Inputs consumed by the build:

- `pinchbench_tasks/meeting_analysis/*.md` — 28 task definitions
- `assets/meetings/*.md` — 4 transcripts (NTIA spectrum advisory, GitLab PMM,
  Tampa City Council, NASA UAP hearing)
- `rl/train/meeting_analysis_split.json` — train/test split

Outputs:

- `meeting_prompts/train.parquet` — 23 train tasks
- `meeting_prompts/val.parquet` — 5 held-out test tasks

The compiled parquet is intentionally not committed — re-run the build whenever
the task definitions or split change.
