# Agent Team Loop Coach Experiment

Date: 2026-05-24

This note records the first `DSv4-Pro policy teacher -> Qwen3 agent team`
loop experiment before the pod was shut down.

## Goal

Test whether a strong teacher model can improve a weak Qwen3-4B agent team by
looking only at Qwen3 trajectories and grader feedback, then updating the team
policy. The teacher does not execute the task and does not write the final
answer.

The intended loop is:

```text
Qwen3 single-agent trajectory
  + Qwen3 self-team trajectory
  + previous Qwen3 policy-run trajectory
  + grader breakdown
    -> DSv4-Pro coach writes next team policy
    -> Qwen3 workers/final execute that policy
    -> repeat until saturated
```

Important constraint: the mainline setting is **Pure Coach Loop**. It does not
use DSv4-Pro's own successful task trajectory as a reference. We briefly tried
reference augmentation, then stopped it because it risks answer-path leakage and
weakens the claim that the coach can improve from Qwen3 failures alone.

## Code Added

- `scripts/team_agent_benchmark.py`
  - Standalone team-agent benchmark path; does not touch the single-agent path.
  - Supports separate policy/worker/final model endpoints.
  - Supports fixed policies through `--policy-file`.
  - Uses isolated OpenClaw home by default:
    `<output-dir>/_openclaw_home/<run-id>/.openclaw`
  - Adds namespace-scoped cleanup and OpenClaw-home lock.

- `scripts/coach_team_policy.py`
  - Calls DSv4-Pro through OpenAI-compatible API.
  - Reads task prompt, prior result JSONs, transcript summaries, and optional
    base policy.
  - Outputs `{ "tasks": { task_id: policy } }` for `team_agent_benchmark.py`.

- `scripts/run_pure_coach_loop_batch.py`
  - Runs the Pure Coach Loop sequentially over tasks.
  - Stops each task on target score, low improvement, or repeated timeout.

## Environment Used

Remote pod was:

```text
root@154.54.102.52 -p 17949
```

Qwen3 base service was:

```text
http://127.0.0.1:8770/v1
served model: qwen3-base
context: 65536
rope scaling: yarn factor 2.0
```

The pod was shut down after the experiment, so remote output paths below are
historical references.

## Key Results

Baseline and comparison settings:

| Setting | Val-5 mean / task score |
|---|---:|
| Qwen3 self-team val_5 mean | `0.5368` |
| DSv4 Flash policy + Qwen3 val_5 mean | `0.4401` |
| Pure Coach Loop best-of-task val_5 mean | `0.7209` |

Best-of-loop val_5:

| Task | Best score | Best iteration / note |
|---|---:|---|
| `task_meeting_advisory_stakeholders` | `0.8750` | Pure v2 |
| `task_meeting_council_votes` | `0.3250` | v1; still weak |
| `task_meeting_gov_speaker_summary` | `0.7882` | v2 |
| `task_meeting_tech_action_items` | `0.9600` | v2; saturated target |
| `task_meeting_sentiment_analysis` | `0.6563` | v1 |

Best-of-loop mean:

```text
(0.8750 + 0.3250 + 0.7882 + 0.9600 + 0.6563) / 5 = 0.7209
```

## Advisory Stakeholders Loop Detail

`task_meeting_advisory_stakeholders` showed the clearest loop signal.

| Setting | Score | Status |
|---|---:|---|
| Qwen3 autonomous team | `0.5233` | success |
| DSv4-Pro cold-start policy + Qwen3 | `0.8583` | success |
| Pure Coach v1 + Qwen3 | `0.4667` | timeout |
| Pure Coach v2 + Qwen3 | `0.8750` | success |
| Pure Coach v3 + Qwen3 | `0.7250` | success |

v2 exceeded the DSv4-Pro cold-start policy without using DSv4-Pro reference
trajectory.

v3 regressed because the coach over-optimized for line-numbered citations. Qwen3
started fabricating citation-looking evidence such as generic `L123: ...`
snippets. This is an important failure mode: asking Qwen3 for exact citations
without a reliable line-indexing tool can reduce quality.

## Per-Task Notes

### `task_meeting_advisory_stakeholders`

Loop works strongly. v2 made all automated checks pass:

```text
report_created = 1
gov_stakeholders = 1
commercial_stakeholders = 1
sharing_preference = 1
relocation_cost = 1
sharing_vs_relocation = 1
common_parameters = 1
conflicts_identified = 1
member_positions = 1
```

Remaining weakness: evidence quality and citations.

### `task_meeting_council_votes`

Loop did not solve it:

```text
v1 = 0.3250
v2 = 0.2000
```

Likely issue: exhaustive vote extraction needs a more tool-grounded protocol.
Natural-language worker instructions still miss many vote events.

### `task_meeting_gov_speaker_summary`

Loop improved substantially:

```text
v1 = 0.3056
v2 = 0.7882
v3 = 0.6875
```

Best is v2. v3 regression suggests overfitting to a narrower failure diagnosis
or overburdening Qwen3 with too much evidence format complexity.

### `task_meeting_tech_action_items`

Strong result:

```text
v1 = 0.8667
v2 = 0.9600
```

This is the cleanest positive case. The loop reached the target threshold.

### `task_meeting_sentiment_analysis`

Moderate result:

```text
v1 = 0.6563
v2 = 0.6165
```

Best is v1. Additional coach iterations can regress; select best policy by
score, not latest policy.

## Operational Lessons

1. Use isolated OpenClaw homes for all team experiments.

   Historical global agents under `/root/.openclaw/agents` can pollute or slow
   experiments. The new team runner avoids this by default.

2. Do not use DSv4-Pro reference trajectories in the mainline loop.

   They may help as an upper-bound diagnostic, but they risk answer-path
   leakage. The mainline claim should be based on Qwen3 trajectories only.

3. Select best-of-loop, not latest.

   Several tasks improved in v2 then regressed in v3.

4. Avoid asking Qwen3 for exact line citations unless the runner provides line
   numbers as structured evidence.

   Otherwise it may fabricate citation-looking snippets.

5. `council_votes` needs a different execution substrate.

   Policy coaching alone is not enough. It likely needs runner-provided
   evidence packets or a line-index/search primitive to ensure exhaustive vote
   coverage.

## Historical Remote Paths

These paths were on the shut-down pod:

```text
/workspace/verl_port/bench/team_policy_advisory_dsv4pro_loop/
/workspace/verl_port/bench/pure_coach_loop_val5_remaining/
/workspace/openclaw-verl-agent-rl/experiments/team_policies/
```

Important historical outputs:

```text
/workspace/verl_port/bench/team_policy_advisory_dsv4pro_loop/v2_pure/adv_loop_v2_pure_qwen_team.json
/workspace/verl_port/bench/pure_coach_loop_val5_remaining/summary.json
```

## Suggested Next Steps

1. Re-run val_5 on a fresh pod to reproduce the `0.7209` best-of-loop result.
2. Run a small train subset, not the full train split first.
   Suggested sample size: 5-10 tasks.
3. Save coach policies and result JSONs locally after every run.
4. Add task-specific failure tags for training data:
   - coverage failure
   - timeout / overbroad worker scope
   - evidence hallucination
   - final synthesis omission
   - tool/parser failure
5. For `council_votes`, test a tool-grounded evidence extractor before more
   policy iterations.

## Resume Commands

After starting a new pod and Qwen3 service, copy local scripts to the pod:

```bash
rsync -av scripts/team_agent_benchmark.py scripts/coach_team_policy.py scripts/run_pure_coach_loop_batch.py \
  root@<pod-host>:/workspace/openclaw-verl-agent-rl/scripts/
```

Then run val_5 Pure Coach Loop sequentially:

```bash
cd /workspace/openclaw-verl-agent-rl
set -a; source /root/.pinchbench_env; set +a
python3 scripts/run_pure_coach_loop_batch.py \
  --tasks task_meeting_advisory_stakeholders task_meeting_council_votes task_meeting_gov_speaker_summary task_meeting_tech_action_items task_meeting_sentiment_analysis \
  --output-root /workspace/verl_port/bench/pure_coach_loop_val5_rerun \
  --max-iters 4 \
  --timeout-multiplier 8 \
  --workers 4
```

