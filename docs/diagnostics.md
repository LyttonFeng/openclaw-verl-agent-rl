# Diagnostics module

`agent_loop/diagnostics/` is a per-task-family trajectory analyzer. It runs
both at rollout time (for the fatal-skip heuristic that saves judge API cost)
and at bench time (post-process, generates the markdown diagnosis report).

It does **not** re-grade. It consumes the already-computed grading breakdown
from `result.json` and surfaces:

- structural failures (timeout, output not written, transcript not read, ...)
- output budget allocation (file vs chat-reply char ratio — catches the
  reward-hacking pattern where model talks in chat but skimps on the file)
- transcript truncation (read tool returns ≥ 39900 chars and model didn't paginate)
- automated grading breakdown (which checks failed, stable across runs?)
- PRM signals (per-turn scores, negatives, gate decision)

## CLI

```bash
python -m agent_loop.diagnostics analyze \
    --result-json /path/to/bench/result.json \
    --transcripts-dirs results/0071_transcripts [results/0070_transcripts ...] \
    --output diagnosis.md \
    --output-json diagnosis.json
```

Output structure (markdown):

```
## Overall                       — fatal/warning/healthy counts + score
## Failure-tag distribution      — sortable tag → tasks affected
## Per-task                      — one row per task: turns, reads, writes,
                                   thinking, output_len, budget_ratio, tags
## Failed automated checks       — per-run breakdown, marks "stable across
                                   runs" patterns
## Notable trajectories          — detailed view of any tagged trajectory
```

## In-process API

For rollout-time use (`rl/train/generate_meeting_rollouts.py`):

```python
from agent_loop.diagnostics import diagnose

diag = diagnose(
    trajectory=transcript_entries,
    workspace_path="/tmp/pinchbench/.../agent_workspace",
    task_id="task_meeting_council_votes",
    execution_time=82.0,
    timed_out=False,
)
if diag.fatal:
    reward = 0.0  # skip the expensive judge call
```

`diag.fatal` covers timeout / output_not_written / empty_response / transcript_not_read.

## Failure tags

| Tag | Layer | Meaning |
|---|---|---|
| `timeout` | 1 (fatal) | episode hit timeout |
| `output_not_written` | 1 (fatal) | expected output file missing |
| `empty_response` | 1 (fatal) | no assistant tool calls at all |
| `transcript_not_read` | 1 (fatal) | model didn't read the meeting |
| `read_loop` | 1 | reads same file ≥3× without writing |
| `serial_read_no_write` | 1 | ≥4 read-only assistant turns |
| `excessive_thinking` | 1 | thinking > 5000 chars total |
| `output_too_short` | 1 | output file < 50 chars |
| `output_budget_misallocated` | 2 | file/(file+chat) ratio < 0.70 |
| `transcript_read_truncated` | 2 | read returned ≥39900 chars, no pagination |
| `output_below_min` | 3 | output file shorter than plugin's expected min |

Layer 1 fatal tags skip the judge (rollout time). Layer 2/3 are warnings only.

## Plugin model

Each task family registers a `TaskPlugin`:

```python
# agent_loop/diagnostics/plugins/meeting_analysis.py
from agent_loop.diagnostics.protocol import TaskPlugin, register_plugin

PLUGIN = TaskPlugin(
    family_id="meeting_analysis",
    expected_output_file={"task_meeting_council_votes": "votes_report.md", ...},
    expected_input_files={"meeting_transcript.md", "transcript.md", ...},
    task_id_prefix_match=("task_meeting_",),
)
register_plugin(PLUGIN)
```

To support a new family (e.g. `task_email_*`), add a new file under
`agent_loop/diagnostics/plugins/` and import it from
`plugins/__init__.py`. No core changes needed.

## Layered design

| Layer | What it knows |
|---|---|
| **L1 — structural** | What the trajectory did (turns, tool calls, files written/read, thinking chars) |
| **L2 — budget** | Where the output chars went (file vs chat reply, read truncation) |
| **L3 — grading** | Consume what's already computed (automated breakdown, PRM scores) — does NOT re-grade |

`fatal=True` only triggers on L1 fatal tags. L2/L3 surface warnings in the
report but don't break the training loop.
