# Swarm Policy Overnight Run — 2026-05-28

## TL;DR (final overnight results)

**40 rollouts collected, 9/10 task groups have GRPO signal** — vs MR2 round_02's
0/15 zero-variance failure. The composite reward `(1-γ)*terminal + γ*swarm`
unlocked variance that pure terminal couldn't produce.

Top variance task groups:
- `task_meeting_tech_decisions`: var **0.0836** (T3_verify hit comp 0.740 vs T1_solo 0.389)
- `task_meeting_advisory_attendees`: var **0.0526** (T1_solo hit comp 0.668)
- `task_meeting_gov_data_sources`: var 0.0242
- `task_meeting_gov_controversy`: var 0.0157

Only `task_meeting_council_budget` got filtered (all 4 rollouts scored 0; task
too hard for base 4B).

Training is **NOT auto-triggered**. Use the one-command kickoff in section
"Morning checklist" step 4 when ready.

## What was built

- `scripts/swarm_policy/templates.py` — 4 prompt templates (T1_solo,
  T2_decompose2, T3_verify, T4_evidence) that force the agent to emit a
  `<plan>` block with a specific swarm policy style.
- `scripts/swarm_policy/judge.py` — DSv4-Pro judge that rates the
  agent's emitted swarm policy on four dimensions
  (decomposition_quality / verification / adherence / efficiency).
  Returns swarm_policy_score in [0, 1] + breakdown + retry on transient API
  failures.
- `rl/train/generate_swarm_rollouts.py` — wrapper around codex's
  `generate_meeting_rollouts.py` that injects templates, extracts the
  emitted plan, and adds composite reward
  `(1-γ)*terminal + γ*swarm_policy`. γ=0.4.

## Pod-side artifacts (after overnight run completes)

- vLLM serves `qwen3-mr2-lora` on GPU 3, port 8766 (reused from MR2 bench)
- Rollout data: `/workspace/meeting_policy_rl/run_20260528_swarm/graded_swarm_trajectories.jsonl`
  - schema: `task_id, template_id, terminal_score, swarm_policy_score, composite_reward, ...`
- Per-task variance summary appended to `/tmp/swarm_overnight.log`

## Tasks selected (10 train tasks, swarm-policy-friendly)

- task_meeting_council_public_comment (multi-stakeholder)
- task_meeting_gov_controversy (multi-aspect)
- task_meeting_tech_decisions (multi-decision tracking)
- task_meeting_tech_competitors (multi-entity)
- task_meeting_tech_product_features (multi-feature)
- task_meeting_council_budget (numbers + verification)
- task_meeting_gov_data_sources (extraction + verify)
- task_meeting_executive_summary (synthesis)
- task_meeting_gov_recommendations (synthesis)
- task_meeting_advisory_attendees (multi-stakeholder)

10 tasks × 4 templates × 1 resp = **40 rollouts** total.

## Key trick: composite reward unlocks variance

MR2 round_02 died from `variance=0`: K rollouts of same task got identical
scores → no advantage signal → no learning. The new pipeline:

- Different template per rollout → different swarm policy → different
  trajectory → terminal score may still collapse, but **swarm_policy_score
  varies** because DSv4-Pro grades the policy text directly.
- Composite reward = `0.6 * terminal + 0.4 * swarm_policy`.
- Even when terminal flatlines, swarm signal provides advantage.

## Morning checklist

1. **Check MR2 3-run bench result** at
   `/workspace/meeting_policy_rl/run_20260527_mr/val5_mr2_temp02_3run/summary.txt`
   to confirm MR2 val5 3-run mean.

2. **Read overnight log**:
   ```bash
   ssh root@154.54.102.40 -p 11102 -i ~/.ssh/id_ed25519 \
     'cat /tmp/swarm_overnight.log'
   ```
   Look for the "Per-task variance check" block at the bottom — count groups
   with non-zero composite variance. Target: ≥ 8 / 10 groups have signal.

3. **Inspect a few rollouts** to verify quality:
   ```bash
   ssh ... 'jq "select(.task_id == \"task_meeting_tech_decisions\") | {template_id, terminal_score, swarm_policy_score, composite_reward, swarm_breakdown}" /workspace/meeting_policy_rl/run_20260528_swarm/graded_swarm_trajectories.jsonl'
   ```

4. **If data looks good**: one-command GRPO + bench (only uses GPU 3):
   ```bash
   ssh root@154.54.102.40 -p 11102 -i ~/.ssh/id_ed25519
   # On pod:
   bash /tmp/run_swarm_grpo_round.sh && bash /tmp/bench_mr_swarm.sh
   ```
   - `run_swarm_grpo_round.sh`: converts swarm JSONL → GRPO schema, runs one
     GRPO step on GPU 3 (Qwen3-4B base + composite_reward), saves LoRA at
     `/workspace/meeting_policy_rl/run_20260528_swarm/checkpoint/lora_adapter`.
   - Score-field knobs:
     - `SCORE_FIELD=composite_reward` (default; what overnight saved)
     - `SCORE_FIELD=terminal_score` (ablation: terminal-only)
     - `SCORE_FIELD=composite GAMMA=0.5` (recompute with custom γ)
   - `bench_mr_swarm.sh`: dynamic-loads the new LoRA into the running vLLM
     (port 8766) as `qwen3-swarm`, runs val5 3-run.

5. **If variance still poor on some tasks**: re-run those tasks with
   `--resp-per-template 2` (8 rollouts per task instead of 4).

## Risks / known gotchas

- **vLLM hermes parser warns** during streaming tool calls — non-fatal, MR2
  bench proceeded despite the WARNING lines. Same applies to overnight.
- **DSv4-Pro swarm judge cost** ≈ $0.001 per rollout (~ 1k input tokens, 200
  output). 40 rollouts ≈ $0.04. Negligible.
- **OpenClaw long-tail** — some rollouts may hit 600s timeout. Records still
  saved with execution_status="timeout".
- **MR2 (the base for these rollouts)** still uses the buggy template-less
  prompt. After GRPO step on the new composite reward, the new LoRA learns
  to bias toward better policy emission.

## Composite reward formula

```python
gamma = 0.4
composite = (1 - gamma) * terminal_score + gamma * swarm_policy_score
```

To change γ (e.g., emphasize terminal more), pass --gamma to a downstream
script when re-running judge/composite. The raw fields are kept in the JSONL
so re-weighting is one-line cheap.

## Files added (commit if desired)

Local repo (already in `experiment/verl-async-openclaw` branch):
- `scripts/swarm_policy/__init__.py` (empty)
- `scripts/swarm_policy/templates.py` — prompt templates + plan extractor
- `scripts/swarm_policy/judge.py` — DSv4-Pro behavior-based swarm judge
- `rl/train/generate_swarm_rollouts.py` — rollout collector wrapping codex's helpers
- `docs/swarm_policy_overnight_20260528.md` (this file)
- `docs/val5_benchmark_table_20260527.md` — val5 comparison table (4B base / 8B / MR2 / DSv4)

Pod-only scripts (`/tmp/` — push to local if useful):
- `/tmp/launch_swarm_overnight.sh` — overnight rollout launcher
- `/tmp/bench_mr2_rerun.sh` — MR2 3-run bench rerun
- `/tmp/swarm_to_grpo.py` — convert swarm JSONL → GRPO schema
- `/tmp/run_swarm_grpo_round.sh` — one-command GRPO training
- `/tmp/bench_mr_swarm.sh` — bench new LoRA on val5

These do not touch codex's running pipeline.

## Important context from this session

### Real val5 numbers (3-run mean, retry-fixed judge)

| Model | val5 |
|---|---:|
| Qwen3-4B base (rope=2) | 48.45% |
| MR2 LoRA (terminal-only RL) | **45.97%** ← lower than base |
| MR-Swarm (target) | > 48% to beat base; ideally > 53% to defend article |

MR2 LoRA's 1-run history result (53.98%) turned out to be a high-variance
lucky sample. With 3-run averaging it's actually -2.5pp below base. So the
swarm-policy track is the only remaining path to beat base in this branch.

### Why MR2 didn't help (analysis)

MR2 trade-off pattern:
- council_votes: +9.17pp (vs base) — wins on one task
- gov_speaker_summary: −9.33pp
- tech_action_items: −7.30pp
- sentiment_analysis: −6.08pp

Net: MR2 specializes for one task at the cost of three others. Likely cause:
the round_01 training data (10 records, 4 opt steps) was too narrow.

### Judge bug fix (still relevant)

Earlier audit of DSv4-Pro and DSv4 Flash bench results showed `_combine_grades`
in `scripts/lib_grading.py` had no retry — when judge API returned status=200
but content="" (empty), the parser returned `{}`, score collapsed to 0.

Fix: added `PINCHBENCH_JUDGE_MAX_RETRIES=3` retry loop with backoff. All bench
runs in this session use the fixed version.
