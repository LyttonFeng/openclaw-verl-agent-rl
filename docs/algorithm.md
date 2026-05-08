# Algorithm

This repo runs **offline GRPO** on Qwen3-4B for the PinchBench `meeting_analysis`
task family, with an optional **Roadmap PRM** providing per-turn process reward.

It is intentionally simple:

- **Async, off-policy.** Rollout (a vLLM serving process) and training (a
  one-step GRPO updater) are decoupled. A round = collect rollouts → grade → score → train → hot-load adapter back into vLLM.
- **No critic, no GAE, no KL.** GRPO uses group-relative advantage; PRM, when
  enabled, only re-weights the per-token advantage.
- **Two reward layers** (both can be enabled together):
  1. **Terminal reward** — automated check + LLM judge on the final artifact.
  2. **Process reward (PRM)** — DSv4-flash judges each rollout turn against a
     task-specific roadmap, giving +1 / 0 / -1 per turn.

## End-to-end flow

```mermaid
flowchart LR
    subgraph "Inference (vLLM, GPU 1)"
        VLLM[vLLM server<br/>Qwen3-4B + rope=2 80K<br/>+ optional LoRA]
    end

    subgraph "Rollout (CPU/GPU 1, parallel)"
        ROLL[generate_meeting_rollouts.py<br/>OpenClaw multi-turn agents<br/>N workers]
        TERM[meeting_reward.py<br/>automated + LLM-judge]
    end

    subgraph "PRM scoring (optional)"
        PRM[score_trajectories.py<br/>DSv4-flash judge + roadmap<br/>terminal-completion gate]
    end

    subgraph "Training (GPU 0)"
        SEL[select_grpo_samples.py<br/>variance filter]
        TRAIN[train_meeting_grpo_step.py<br/>GRPO step + LoRA save]
    end

    VLLM -->|sample N responses| ROLL
    ROLL --> TERM
    TERM -->|graded.jsonl| PRM
    PRM -->|graded_prm.jsonl| SEL
    TERM -.->|graded.jsonl<br/>(terminal-only path)| SEL
    SEL --> TRAIN
    TRAIN -->|LoRA adapter| VLLM
```

The dashed path skips the PRM stage entirely — that's terminal-only training.

## Reward formula

Per token-span `k` in response `i`, GRPO advantage is computed group-relatively
within each prompt's group of `N` responses:

```
group_mean_i = mean(score_j for j in group)
terminal_adv_i = (score_i - group_mean_i) / max(std_group, 1.0)
```

Then the per-token advantage depends on `prm_mode`:

```
# Terminal-only (β = 0)
per_token_adv[k] = α · terminal_adv_i

# Additive (default, β = 0.10)
per_token_adv[k] = α · terminal_adv_i + β · prm_turn_score[k]

# Multiplicative (β = 1.0; only amplifies positive advantages)
if terminal_adv_i > 0:
    per_token_adv[k] = terminal_adv_i · (1 + β · prm_turn_score[k])
else:
    per_token_adv[k] = terminal_adv_i   # failures keep pure terminal gradient
```

`prm_turn_score[k] ∈ {-1, 0, +1}` is the roadmap judge's score on the assistant
turn that produced token `k`.

**Pos-only clip** (recommended, on by default): -1 is treated as 0 — only
encourage progress, never penalize. Rationale: per-turn judges have
non-negligible false-negative rate on hard tasks (the judge sees a partial
trajectory and can't always tell if a turn is a wrong path or just a slow
exploration step). Letting -1 directly subtract from advantage punishes
exploration steps the judge mis-classified, hurting the very tasks where
PRM should help most. Empirically (`experiment_report.md` §"Judge-gate
ablation"), pos-only clip is the single largest design lever in our setup
(+3.1pp over the no-clip variant, holding everything else fixed).

## Roadmap PRM design

Three features distinguish the PRM:

1. **Roadmaps come from successful trajectories** — the "key milestones to
   reach" for each task are extracted from real expert runs (the calibrated
   yaml files under `agent_loop/roadmap_prm/roadmaps/`), or can be supplied by
   a stronger teacher model. No human-written rubric per task.
2. **Per-turn judging.** DSv4-flash takes the roadmap + one rollout turn at a
   time, decides which milestone (if any) it advanced, returns +1/0/-1.
3. **Terminal-completion gate** ("only help the lost trajectories"). Before
   running per-turn scoring, the judge first decides whether the trajectory
   already accomplished the terminal goal (`mostly_done: yes/no`). If yes →
   `prm_turn_scores = [0]*n_turns`, training falls through to pure terminal
   gradient. If no → run the per-turn judge. **Successful trajectories never
   receive PRM amplification**, which is the most common reward-hacking source
   in PRM training.

Code: `agent_loop/roadmap_prm/judge.py:judge_terminal_completion`,
`judge.py:judge_trajectory`.

## Why offline + async

Rollout cost dominates wall-clock for agentic tasks (multi-turn, tool use,
real workspace I/O). Coupling rollout to gradient steps is wasteful. Instead:

- vLLM stays up across rounds; a new LoRA is hot-loaded after each training
  step (`/v1/load_lora_adapter` API). No restart.
- Rollouts and PRM scoring run in parallel workers (default 4) against the
  same vLLM endpoint.
- Training reads the collected JSONL — a single vectorized GRPO step on the
  whole rollout batch. No multi-step PPO loop.

A round in our setup runs end-to-end in roughly:

| Stage | Time (Qwen3-4B, 30 records, single A100) |
|---|---|
| Rollout (4 workers, 23 tasks × 2 responses) | ~25 min |
| PRM scoring (DSv4-flash, 4 workers) | ~5 min |
| Variance filter + pos-only clip | <1 min |
| GRPO step (15 updates, batch=2) | ~10 min |
| Hot-load + 3-run bench | ~25 min |
| **Total** | ~65 min |
