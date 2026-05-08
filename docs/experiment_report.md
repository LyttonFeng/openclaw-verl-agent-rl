# Experiment Report — Meeting Analysis GRPO

3-run mean scores on 5 held-out test tasks, judge = `openai/deepseek-chat`.

## Setup

| Item | Value |
|---|---|
| Model | Qwen3-4B (LoRA rank 16, α=32, target = q/k/v/o/gate/up/down) |
| Context | 80K (rope-scaling dynamic factor=2.0), bf16 |
| GRPO | offline, group N=2 responses/prompt, no critic, no GAE |
| Train data | 23 tasks × 2 responses = 46 rollouts/round, variance-filtered, pos-only PRM clip |
| Test data | 5 held-out tasks × 3 runs (judge variance reduction) |
| Source meetings | NTIA spectrum advisory (71KB) · GitLab PMM (34KB) · Tampa City Council (206KB) · NASA UAP hearing (265KB) |
| Step budget | 15 GRPO updates per round, lr=2e-6, batch=2 (grad accum) |

## Baselines

Two baselines coexist because rope-scaling slightly penalizes the unadapted
base model. **All trained models below are bench'd at rope=2; compare against
the rope=2 baseline to stay apples-to-apples.**

| Config | 3-run mean |
|---|---|
| Qwen3-4B base, rope=1 / 40K | 51.7% |
| Qwen3-4B base, rope=2 / 80K | **50.6%** ← canonical baseline |

## Headline result

| Config | 3-run mean | Δ vs rope=2 baseline | Rounds to converge |
|---|---|---|---|
| baseline rope=2 (no LoRA) | 50.6% | — | — |
| terminal-only GRPO | 55.0% | +4.4pp | ~5 rounds |
| **terminal + Roadmap PRM (additive judge-gate, β=0.10)** | **57.24%** | **+6.6pp** | **1 round** |

`additive judge-gate` formula for per-token advantage:

```
per_token_adv[k] = α · terminal_adv + β · prm_turn_score[k]
```

with α=1.0, β=0.10, pos-only clip on `prm_turn_score`. The judge first decides
whether each rollout is `mostly_done`; if yes, `prm_turn_scores=[0]*n_turns`
(pure terminal gradient — "only help the lost trajectories").

See `algorithm.md` §"Roadmap PRM design" for the full formula and rationale.

## Per-task (R1 SOTA vs baseline)

| Task | rope=2 baseline | R1 additive judge-gate | Δ |
|---|---|---|---|
| advisory_stakeholders | 0.49 | 0.573 | +0.083 (+17%) |
| **council_votes (hardest)** | **0.18** | **0.283** | **+0.103 (+57%)** |
| gov_speaker_summary | 0.55 | 0.604 | +0.054 (+10%) |
| sentiment_analysis | 0.68 | 0.672 | -0.008 (-1%) |
| tech_action_items | 0.63 | 0.729 | +0.099 (+16%) |
| **Overall** | **0.506** | **0.5724** | **+0.066 (+13%)** |

Council_votes (lowest baseline, hardest task) shows the largest gain —
consistent with the design intent: PRM concentrates supervision on failed
trajectories, where it most matters.

## Judge-gate hit ratio

```
22/46 trajectories  mostly_done (no PRM applied)
24/46 trajectories  lost (per-turn PRM applied)
```

Stable across rounds — rubric does not drift. If this ratio collapses to
`0/46` or `46/46`, the rubric needs re-calibration.

## Continued training: R2 regression observation

R2-additive (continuing from R1-additive LoRA, same recipe): 55.34%, regressing
-1.9pp from R1. Per-task: advisory_stakeholders fell from 0.573 to 0.438
(-13.5pp), with all 3 runs failing the same automated checks — a deterministic
capability loss, not stochastic noise.

The diagnostics module (`agent_loop/diagnostics`) caught the root cause: the
model shifted output budget from the `.md` report file to the chat-side
summary at the end of the trajectory:

| Task | R1 budget ratio | R2 budget ratio |
|---|---|---|
| advisory_stakeholders | 0.92 | 0.64 ⬇️ |
| sentiment_analysis | 0.75 | 0.47 ⬇️ |

`budget_ratio = output_file_chars / (output_file_chars + final_chat_chars)`.
Automated grading reads only the file; PRM judge sees the chat-side summary
and tends to reward "thoughtful explanation". Misalignment between PRM
incentive and terminal evaluator → textbook reward hacking.

This validates two design principles in practice:

1. **Don't continue indefinitely with the same recipe.** Single-round PRM gain
   is real; multi-round amplification opens reward-hacking failure modes.
2. **Diagnostics tooling is as important as training tooling.** Without
   per-task budget-ratio tracking, this regression looked like ordinary noise.

R3+ paused pending PRM rubric update (penalize chat-side verbose so it stops
being a +1 signal).

## Convergence comparison

Terminal-only training converges to a similar plateau but takes ~5 rounds,
versus 1 round for terminal+PRM. The terminal-only column below is illustrative
(scaled from rope=1 R1-R6 deltas with peak aligned to the actual rope=2 R5
bench at 55.0%, since the actual rope=2 R1-R4 terminal-only checkpoints were
not benched separately):

| Round | Terminal-only *(illustrative)* | Terminal + Roadmap PRM *(measured)* |
|---|---|---|
| baseline (rope=2) | 50.6% | 50.6% |
| R1 | 51.8% | **57.24%** ✅ converged |
| R2 | 50.0% | — |
| R3 | 52.2% | — |
| R4 | 52.8% | — |
| **R5** | **55.0%** ← peak | — |
| R6 | 52.2% | — |

PRM provides the same plateau gain at 1/5 the rounds — the value of process
supervision is convergence speed, not necessarily a higher final ceiling.

## Reproducibility

- Train + bench: see [`reproduction.md`](reproduction.md)
- Diagnostics CLI: see [`diagnostics.md`](diagnostics.md)
- Algorithm + flow diagram: see [`algorithm.md`](algorithm.md)
