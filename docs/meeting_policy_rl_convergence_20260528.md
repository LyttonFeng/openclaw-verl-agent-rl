# Meeting Policy RL Convergence

Date: 2026-05-28

This is the real validation curve from the mini-round GRPO run. It is not smoothed and not forced to be monotonic.

## Curve Data

| Point | Checkpoint | Val runs | Score |
|---|---:|---:|---:|
| Baseline report | Qwen3-4B rope2 | report | 44.7% |
| Baseline observed | Qwen3-4B rope2 | 1 | 47.8-49.8% |
| MR2 | policy_mr2 | 1 | 54.0% |
| MR4 | policy_mr4 | 1 | 52.9% |
| MR6 | policy_mr6 | 1 | 53.7% |
| MR8 | policy_mr8 | 1 | 55.1% |
| MR8 formal | policy_mr8 | 3 | 54.2% |
| MR9 | policy_mr9 | 1 | 47.4% |
| MR10 | policy_mr10 | 1 | 50.8% |
| MR11b invalid | policy_mr11b | 1 | 33.5% |
| MR11b | policy_mr11b | 1 | 56.3% |
| MR11b formal | policy_mr11b | 3 | 47.2% |

## Readout

The run did improve over the 4B baseline. The honest shape is an early jump, a small dip, a recovery, a new best point after the council/tech focused MR8 round, a regression after MR9/MR10, and a new best single-run point after rolling back to MR8 and running a targeted group=4 round.

MR8 is the current best checkpoint:

`/workspace/meeting_policy_rl/run_20260527_mr/round_08_temp07/checkpoint/lora_adapter`

The formal 3-run MR8 result is lower than the single-run best, but still confirms the checkpoint is in the improved band rather than a one-off failure.

MR9 and MR10 are not the best checkpoints. MR9 fell to 47.4%, with `council_votes` at 10% and `tech_action_items` hallucinating action items. MR10 partially recovered to 50.8%, but `council_votes` still reverted to the shallow-read failure mode: the judge noted that the agent only read the first 1500 lines and missed the key votes.

The first MR11b validation produced 33.5%, but that point is marked invalid because `gov_speaker_summary` had a judge transport failure: `IncompleteRead(8186 bytes read)`. A rerun with per-task judge-output inspection produced 56.3%. All five rerun tasks had non-empty breakdowns and no `Grading failed` / `IncompleteRead` notes.

The MR11b formal 3-run result was 47.2%. All 15 task-runs had non-empty judge breakdowns and no `Grading failed` / `IncompleteRead` notes, so the formal result should be treated as valid. The 56.3% single-run point was not confirmed.

## Source Files

| Result | Path |
|---|---|
| MR2 1-run | `/workspace/meeting_policy_rl/run_20260527_mr/val5_mr2_temp02_1run/0001_policy_mr2.json` |
| MR4 1-run | `/workspace/meeting_policy_rl/run_20260527_mr/val5_mr4_temp02_1run/0001_policy_mr4.json` |
| MR6 1-run | `/workspace/meeting_policy_rl/run_20260527_mr/val5_mr6_temp02_1run/0001_policy_mr6.json` |
| MR8 1-run | `/workspace/meeting_policy_rl/run_20260527_mr/val5_mr8_temp02_1run/0001_policy_mr8.json` |
| MR8 3-run | `/workspace/meeting_policy_rl/run_20260527_mr/val5_mr8_temp02_3run/0001_policy_mr8.json` |
| MR9 1-run | `/workspace/meeting_policy_rl/run_20260527_mr/val5_mr9_temp02_1run/0001_policy_mr9.json` |
| MR10 1-run | `/workspace/meeting_policy_rl/run_20260527_mr/val5_mr10_temp02_1run/0001_policy_mr10.json` |
| MR11b invalid 1-run | `/workspace/meeting_policy_rl/run_20260527_mr/val5_mr11b_temp02_1run/0001_policy_mr11b.json` |
| MR11b 1-run | `/workspace/meeting_policy_rl/run_20260527_mr/val5_mr11b_temp02_1run_rerun/0001_policy_mr11b.json` |
| MR11b 3-run | `/workspace/meeting_policy_rl/run_20260527_mr/val5_mr11b_temp02_3run/0001_policy_mr11b.json` |

## Per-Task MR8 1-Run

| Task | Score |
|---|---:|
| advisory_stakeholders | 64.2% |
| council_votes | 24.4% |
| gov_speaker_summary | 52.4% |
| tech_action_items | 75.3% |
| sentiment_analysis | 59.1% |

## Per-Task MR8 Formal 3-Run Mean

| Task | Score |
|---|---:|
| advisory_stakeholders | 63.2% |
| council_votes | 20.2% |
| gov_speaker_summary | 59.1% |
| tech_action_items | 64.9% |
| sentiment_analysis | 63.5% |

## Per-Task MR10 1-Run

| Task | Score |
|---|---:|
| advisory_stakeholders | 56.0% |
| council_votes | 22.0% |
| gov_speaker_summary | 52.0% |
| tech_action_items | 68.0% |
| sentiment_analysis | 56.0% |

## Per-Task MR9 1-Run

| Task | Score |
|---|---:|
| advisory_stakeholders | 57.0% |
| council_votes | 10.0% |
| gov_speaker_summary | 58.0% |
| tech_action_items | 56.0% |
| sentiment_analysis | 57.0% |

## Per-Task MR11b 1-Run

| Task | Score |
|---|---:|
| advisory_stakeholders | 51.3% |
| council_votes | 41.3% |
| gov_speaker_summary | 63.9% |
| tech_action_items | 71.7% |
| sentiment_analysis | 53.1% |

## Practical Conclusion

MR11b is the current best single-run checkpoint, but it still needs 3-run confirmation before replacing MR8 as the formal checkpoint. Do not present the curve as a smooth monotonic convergence curve; the honest claim is baseline-to-improved-band convergence, regression from noisy/off-target rounds, and recovery after MR8 rollback plus targeted group=4 training.

After formal confirmation, keep MR8 as the best formal checkpoint: MR8 3-run is 54.2%, while MR11b 3-run is 47.2%.
