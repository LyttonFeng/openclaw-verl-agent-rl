# Meeting Policy RL Handoff - 2026-05-27

## Goal

Use newly built meeting-analysis policy RL data to improve Qwen3-4B on Val5 meeting tasks, without training on Val5 originals.

Current target behavior:

- Read long transcripts with targeted passes instead of guessing after one scan.
- Build intermediate evidence/ledger state before final output.
- Do pre-final verification against task requirements.
- Keep staleness low by alternating small rollout rounds and LoRA updates.

## Data

Local/pod dataset path:

- Pod: `/root/openclaw-verl-agent-rl/rl/data/meeting_policy_rl/`
- Main mixed file: `meeting_grpo_mixed_40_40_20.jsonl`
- Mini-round files: `mini_rounds/mini_round_01.jsonl` through `mini_round_04.jsonl`

Dataset construction:

- Source: 23 non-Val5 `meeting_analysis` train tasks.
- Generated task variants: `base`, `evidence_ledger`, `pre_final_audit`.
- Gold: DeepSeek + Qwen teacher consensus, merged after retrying failed teacher cases.
- Full GRPO dataset: 69 rows.
- HQ dataset: 65 rows.
- Mixed schedule: 100 rows, sampled as:
  - 40% base
  - 40% evidence / targeted extraction
  - 20% pre-final audit / verification

Important: Val5 original tasks were not used as training tasks.

## Training Plan

Conservative mini-round loop:

1. Roll out 15 prompts with group size 2.
2. Score with consensus-gold reward plus trajectory policy features.
3. Keep only groups with reward variance.
4. Compute rollout-time logprobs `P_old`.
5. Train one PPO/GRPO LoRA step.
6. Use the new LoRA for the next mini-round.
7. Run Val5 1-run every 2 mini-rounds.

Hyperparameters:

- Model: Qwen3-4B, rope factor 2.0, max seq len 65536.
- LoRA rank: 16, alpha 32.
- LR: `1e-6`.
- Group size: 2.
- Grad accum: 2.
- PPO clip eps: 0.2.
- KL beta: 0.02.
- PRM beta: 0 for this first pass.
- Rollout sampling intended: temp 0.7, top_p 0.9.
- Validation sampling: temp 0.2, top_p 0.9.

Report guidance followed from `docs/experiment_report.md`:

- Do not train zero-variance groups.
- Avoid vanilla PG; use quality/variance filtering plus PPO/KL.
- Watch for race-to-bottom groups where all samples are poor.
- Healthy KL range from prior clean-chain runs is about 0.001-0.003.
- Stop early if Val5 starts regressing after a few rounds.

## GPU / Runtime Setup

Pod:

- SSH: `root@154.54.102.40 -p 11102`
- Repo: `/root/openclaw-verl-agent-rl`
- Base model copied to container disk:
  - `/root/hf_cache/qwen3-4b-rope2`
- LoRA/checkpoints on network/workspace disk:
  - `/workspace/meeting_policy_rl/run_20260527_mr/`

Initial assumption was 3 rollout GPUs, then kill vLLM and train. Observed:

- 3 single-GPU vLLM replicas work.
- vLLM cold start with LoRA takes around 2-3 minutes after cache warmup.
- Training script currently uses only one GPU for GRPO update.
- Rollout dominates wall clock once OpenClaw runs real local multi-turn tasks.

Possible alternative:

- Keep 2 GPUs for rollout and 1 GPU for train/logprob to reduce lifecycle switching.
- This only helps if LoRA hot-loading or endpoint lifecycle is reliable; otherwise stale rollout risk remains.

## Code Changes Made

1. Added policy rollout runner:
   - `rl/train/generate_meeting_policy_rollouts.py`
   - Consumes GRPO JSONL records instead of original split files.
   - Outputs `graded_trajectories.jsonl`, transcripts, workspaces, and shard summaries.

2. Fixed R1/base logprob computation:
   - `rl/train/compute_rollout_logprobs.py`
   - Plain HF `Qwen3ForCausalLM` also exposes `.base_model`; script now distinguishes plain model from PEFT wrapper.

3. Tightened consensus-gold scorer:
   - `scripts/rl_data/score_meeting_rollout.py`
   - Old scorer saturated too easily.
   - New scorer uses stricter claim hit threshold, low-agreement weak signal, continuous read/write/verify/output features.

## Progress So Far

### Mini-Round 1

Rollout:

- Path: `/workspace/meeting_policy_rl/run_20260527_mr/round_01/`
- 15 prompts x 2 responses = 30 trajectories.
- Initial old reward was saturated:
  - Shard means: 0.9712 / 0.982 / 1.0
  - 15/15 groups had zero variance.
- After tightening reward:
  - Mean score: 0.805
  - Valid GRPO groups: 5 / 15
  - Kept records: 10

Training:

- Logprobs: `/workspace/meeting_policy_rl/run_20260527_mr/round_01/rollout_logprobs.jsonl`
- Checkpoint: `/workspace/meeting_policy_rl/run_20260527_mr/round_01/checkpoint/lora_adapter`
- Training meta:
  - Records: 10
  - Skipped near-zero advantage: 2
  - Optimizer steps: 4
  - Avg KL: 0.002077
  - LR: 1e-6

This produced `policy_mr1`.

### Mini-Round 2 First Attempt

Rollout with `policy_mr1`, but without explicit temp/top_p and with OpenClaw trying remote ECS:

- Path: `/workspace/meeting_policy_rl/run_20260527_mr/round_02/`
- 30 trajectories completed very fast.
- Mean score: 0.779.
- Valid GRPO groups: 0 / 15.

Diagnosis:

- Same prompt responses were effectively same-score.
- Sampling diversity was insufficient.
- Logs showed OpenClaw was trying ECS sync and falling back; this made logs noisy and likely affected execution mode.

Decision: discard this attempt for training.

### Mini-Round 2 Temp 0.7 Rerun

Rerun with:

- `PINCHBENCH_MODEL_TEMPERATURE=0.7`
- `PINCHBENCH_MODEL_TOP_P=0.9`
- `PINCHBENCH_FORCE_LOCAL_OPENCLAW=1`

Path:

- `/workspace/meeting_policy_rl/run_20260527_mr/round_02_temp07/`

Result:

- 30 / 30 trajectories completed.
- Mean score: 0.945.
- Score range: 0.75 to 1.00.
- Valid GRPO groups: 3 / 15.
- Kept records: 6.

Training:

- Logprobs: `/workspace/meeting_policy_rl/run_20260527_mr/round_02_temp07/rollout_logprobs.jsonl`
- Checkpoint: `/workspace/meeting_policy_rl/run_20260527_mr/round_02_temp07/checkpoint/lora_adapter`
- Training meta:
  - Records: 6
  - Skipped near-zero advantage: 2
  - Optimizer steps: 2
  - Avg KL: 0.002521
  - LR: 1e-6

This produced `policy_mr2`.

## Current Validation

MR2 Val5 1-run completed.

Config:

- Model served by vLLM as `policy_mr2`.
- Base URL: `http://127.0.0.1:8100/v1`
- Output dir: `/workspace/meeting_policy_rl/run_20260527_mr/val5_mr2_temp02_1run`
- Runs: 1
- Validation temperature: 0.2
- Judge: `deepseek-chat`

Result:

- Overall: 2.70 / 5.00 = 54.0%.
- Result file: `/workspace/meeting_policy_rl/run_20260527_mr/val5_mr2_temp02_1run/0001_policy_mr2.json`
- Per-task:
  - `task_meeting_advisory_stakeholders`: about 0.58.
  - `task_meeting_council_votes`: about 0.24.
  - `task_meeting_gov_speaker_summary`: about 0.55.
  - `task_meeting_tech_action_items`: about 0.68.
  - `task_meeting_sentiment_analysis`: about 0.66.

Interpretation:

- This is only 1 run, so do not treat it as a final score.
- It is directionally promising relative to several Qwen3-4B baseline records.
- Council votes remains the weakest task and still needs targeted improvement.

Baseline references found:

- Report official baseline: Qwen3-4B rope=2 / 64K = 44.68%.
- Pod file with Qwen3-4B clean baseline:
  - `/workspace/verl_port/bench_val5_baselines_20260526235136_clean/qwen3_4b_rope2/0001_qwen3-4b.json`
  - Shows 47.8%, but likely not the same pure-base setting as the report baseline.
- Other isolated Qwen3-4B reruns on pod:
  - `/root/openclaw-verl-agent-rl/results/val5_isolated/20260527_034702_31388/0001_qwen3-4b.json`: 40.0%
  - `/root/openclaw-verl-agent-rl/results/val5_isolated/20260527_041145_5479/0001_qwen3-4b.json`: 49.8%

Takeaway: Val5 is high variance. Compare MR2 1-run against several same-wrapper baselines, not a single number.

## Key Issues / Lessons

1. Reward saturation was real.
   - Old claim overlap scorer made almost everything score near 1.0.
   - Tightened reward recovered useful variance in MR1.

2. Group size 2 may be too weak after MR1.
   - MR2 had only 3 valid groups after temp 0.7.
   - If next rounds continue to produce sparse variance, use group size 4 for high-value / hard prompts.

3. Explicit sampling env is required.
   - Set `PINCHBENCH_MODEL_TEMPERATURE=0.7` and `PINCHBENCH_MODEL_TOP_P=0.9` for training rollout.
   - Set `PINCHBENCH_MODEL_TEMPERATURE=0.2` for validation.

4. Force local OpenClaw for this training loop.
   - Use `PINCHBENCH_FORCE_LOCAL_OPENCLAW=1`.
   - Otherwise OpenClaw tries ECS sync and emits permission warnings.

5. Local OpenClaw is slower but cleaner.
   - Some trajectories take 90-150 seconds.
   - One `gov_qa_extract` trajectory took about 555 seconds.
   - Consider training once enough valid groups exist instead of waiting for all 30 trajectories.

6. KL is healthy so far.
   - MR1 avg KL: 0.0021.
   - MR2 avg KL: 0.0025.
   - This matches the clean-chain healthy range.

## Suggested Next Steps

1. Finish MR2 Val5 1-run and record per-task scores.

2. If MR2 Val5 improves or is neutral:
   - Continue to mini-round 3.
   - Use explicit temp/top_p and force local OpenClaw from the start.
   - Consider early cutoff once enough valid groups are present.

3. If MR2 Val5 regresses:
   - Compare MR1 vs MR2 if possible.
   - MR2 only had 2 optimizer steps; regression may be noise, but check transcripts for over-verbose ledger behavior.

4. For future rollouts:
   - Try group size 4 on council / tech / gov hard subsets.
   - Or keep group size 2 but sample more prompts per mini-round and train only valid groups.

5. Improve reward:
   - Add a continuous output-quality dimension that is not trivially maxed.
   - Add task-specific automated checks where available.
   - Avoid rewarding scaffold compliance alone.

6. Runtime optimization:
   - Investigate vLLM LoRA hot-load for `policy_mr{k}`.
   - If stable, keep rollout endpoints warm and avoid repeated cold starts.
   - Otherwise, 3-GPU rollout then 1-GPU train remains simple and auditable.
