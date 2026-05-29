# DSv4-Flash Val5 Task Scores

Date: 2026-05-29

Scope: PinchBench `meeting_analysis` Val5, 3 runs per task.

| Task | DSv4-Flash single agent | DSv4-Flash + 4B base sub | DSv4-Flash + LoRA sub |
|---|---:|---:|---:|
| advisory_stakeholders | 0.9583 | 0.8556 | 0.9333 |
| council_votes | 0.7000 | 0.4292 | 0.6042 |
| gov_speaker_summary | 0.6248 | 0.7491 | 0.7389 |
| tech_action_items | 0.8333 | 0.8267 | 0.8444 |
| sentiment_analysis | 0.8646 | 0.8750 | 0.8750 |
| **Overall** | **0.7962** | **0.7471** | **0.7992** |

## Run-Level Details

| Task | DSv4-Flash single agent | DSv4-Flash + 4B base sub | DSv4-Flash + LoRA sub |
|---|---:|---:|---:|
| advisory_stakeholders | 0.8750 / 1.0000 / 1.0000 | 0.8333 / 0.9000 / 0.8333 | 1.0000 / 0.9000 / 0.9000 |
| council_votes | 0.6000 / 0.7875 / 0.7125 | 0.7125 / 0.5750 / 0.0000 | 0.5750 / 0.7125 / 0.5250 |
| gov_speaker_summary | 0.6979 / 0.5604 / 0.6160 | 0.6917 / 0.7639 / 0.7917 | 0.7472 / 0.8472 / 0.6222 |
| tech_action_items | 0.9000 / 0.8000 / 0.8000 | 0.8333 / 0.8133 / 0.8333 | 0.8333 / 0.8000 / 0.9000 |
| sentiment_analysis | 0.8750 / 0.8750 / 0.8438 | 0.8750 / 0.8750 / 0.8750 | 0.8750 / 0.8750 / 0.8750 |

## Notes

- `DSv4-Flash + 4B base sub` uses the raw 3-run result. The third `council_votes` run timed out and is kept as `0.0000`.
- `DSv4-Flash + LoRA sub` uses the raw 3-run result. No single-task rerun replacement is applied.
- Additional council reruns exist, but are excluded from this table to keep the comparison objective.
