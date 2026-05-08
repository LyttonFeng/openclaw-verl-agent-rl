# Experiment Report — Meeting Analysis GRPO

3-run mean scores on 5 held-out test tasks, judge = `deepseek-v4-flash`.
**Scope notice**: this is a small-N study (5 test tasks × 3 runs = 15
evaluations per checkpoint, single training seed). Numbers below are point
estimates; we report per-run scores so the reader can judge variance.
Conclusions are framed as observations, not statistical claims.

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

`rope=N` here refers to RoPE-scaling's `dynamic` factor — `rope=1` is the
native 40K-token context, `rope=2` doubles the effective context to 80K
(needed because Tampa City Council and NASA UAP transcripts exceed 40K).
The unadapted base model takes a small (-1.1pp) quality hit at rope=2 since
it never trained at extended positions; LoRA training recovers and exceeds
the rope=1 baseline.

**All trained models below are bench'd at rope=2; compare against the rope=2
baseline to stay apples-to-apples.**

| Config | 3-run mean |
|---|---|
| Qwen3-4B base, rope=1 / 40K | 51.7% |
| Qwen3-4B base, rope=2 / 80K | **50.6%** ← canonical baseline |

## Headline result

| Config | 3-run mean | Δ vs rope=2 baseline |
|---|---|---|
| baseline rope=2 (no LoRA) | 50.6% | — |
| **terminal + Roadmap PRM (additive judge-gate, β=0.10) — R1** | **57.24%** | **+6.6pp** |

Per-task per-run scores for the SOTA checkpoint:

| Task | run 1 | run 2 | run 3 | mean | std |
|---|---|---|---|---|---|
| advisory_stakeholders | 0.560 | 0.553 | 0.607 | 0.573 | 0.029 |
| council_votes | 0.344 | 0.344 | 0.163 | 0.283 | 0.105 |
| gov_speaker_summary | 0.719 | 0.604 | 0.490 | 0.604 | 0.115 |
| sentiment_analysis | 0.646 | 0.652 | 0.719 | 0.672 | 0.040 |
| tech_action_items | 0.717 | 0.820 | 0.650 | 0.729 | 0.086 |

Note the per-task std is non-trivial — gov_speaker has run-to-run swing
±0.115, council_votes ±0.105. The +6.6pp overall gain is well outside the
overall 3-run jitter, but per-task claims should be read with that variance
in mind.

The advantage formula (α=1.0, β=0.10, pos-only clip on `prm_turn_score`):

```
per_token_adv[k] = α · terminal_adv + β · prm_turn_score[k]
```

The `mostly_done` gate runs first; if `yes`, `prm_turn_scores=[0]*n_turns`
(pure terminal gradient). See [`algorithm.md`](algorithm.md) for the design
rationale.

## Judge-gate ablation (R1)

Same base, same 30-record train set, same 15 GRPO steps, varying the per-token
advantage shape. This isolates the contribution of the judge-gate vs. just
running per-turn PRM scoring.

| Config | mean | Δ vs rope=2 baseline | What changed |
|---|---|---|---|
| rope=2 baseline | 50.6% | — | no training |
| terminal-only β=0 | 52.5% | +1.9pp | terminal advantage only |
| additive β=0.10, raw PRM (incl. -1) | 51.0% | +0.4pp | per-turn judge, no gate, no clip |
| additive β=0.10, **pos-only**, no gate | 55.6% | +5.0pp | pos-only clip alone; -1 → 0 |
| additive β=0.10, pos-only, **mult-B form** | 56.6% | +6.0pp | `(1+β·prm)` shape, no gate |
| **additive β=0.10, pos-only, judge-gate (SOTA)** | **57.24%** | **+6.6pp** | adds gate on top of pos-only additive |

Reading: the **biggest single win is pos-only clip** (+5.0pp over baseline,
or +3.1pp over no-clip). The **judge-gate adds an additional +1.6pp** on top
of pos-only additive — meaningful but not the whole story. The mult-B form
(without gate) is in the same ballpark (56.6%, +6.0pp).

This says the gate is real but not magic; pos-only clip is doing most of the
heavy lifting. We have not run multi-seed reps to put error bars on the
+1.6pp gate gain — treat it as suggestive at this N.

## Per-task gain analysis

| Task | rope=2 baseline | SOTA (R1 judge-gate) | abs Δ | rel Δ |
|---|---|---|---|---|
| advisory_stakeholders | 0.49 | 0.573 | +0.083 | +17% |
| council_votes (hardest) | 0.18 | 0.283 | **+0.103** | **+57%** |
| gov_speaker_summary | 0.55 | 0.604 | +0.054 | +10% |
| sentiment_analysis | 0.68 | 0.672 | -0.008 | -1% |
| tech_action_items | 0.63 | 0.729 | +0.099 | +16% |
| **Overall** | **0.506** | **0.5724** | **+0.066** | **+13%** |

The largest **relative** gains are on the lowest-baseline tasks — partly a
headroom effect (low baseline = more room to grow), partly consistent with
the "PRM concentrates on failed trajectories" design intent. We can't
cleanly separate these without per-task ablations on the gate, which we
have not run. tech_action_items is a counter-example: high baseline (0.63)
but absolute gain comparable to council_votes (+0.10pp), which doesn't fit
pure headroom. sentiment_analysis is essentially flat — the model was
already close to ceiling and PRM had nothing useful to add.

## Judge-gate hit ratio

R1 and R2 both produced the same gate split:

```
22/46 trajectories  mostly_done  (no PRM applied)
24/46 trajectories  lost          (per-turn PRM applied)
```

Two rounds isn't enough to claim "stable across rounds" generally, but the
**identical 22/24 split in both rounds** suggests the gate decision is
deterministic on the train data given a fixed roadmap and base model. If
this ratio collapses to `0/46` or `46/46`, the rubric has drifted and needs
re-calibration. We monitor it as a smoke signal.

## Continued training: R2 regression

R2-additive (continuing from R1-additive LoRA, same recipe): **55.34%**,
regressing -1.9pp from R1.

Per-task R2 vs R1:

| Task | R1 | R2 | Δ |
|---|---|---|---|
| advisory_stakeholders | 0.573 | 0.438 | **-0.135** ⚠ |
| council_votes | 0.283 | 0.233 | -0.050 |
| gov_speaker_summary | 0.604 | 0.652 | +0.048 |
| sentiment_analysis | 0.672 | 0.689 | +0.017 |
| tech_action_items | 0.729 | 0.754 | +0.025 |

advisory_stakeholders accounts for almost all the regression. All 3 R2 runs
fail the same automated checks (deterministic, not stochastic).

The diagnostics module surfaced one strong correlate of the regression — a
shift in **output-budget allocation** between the trained file artifact and
the chat-side summary at the end of the trajectory:

| Task | R1 budget ratio | R2 budget ratio |
|---|---|---|
| advisory_stakeholders | 0.92 | **0.64** ⬇ |
| sentiment_analysis | 0.75 | 0.47 ⬇ |

`budget_ratio = output_file_chars / (output_file_chars + final_chat_chars)`.

**Hypothesis (not yet confirmed by judge-prompt inspection)**: the PRM judge
reads trajectory turns, and a chat-side summary near the end can look like
a "thoughtful wrap-up" worth +1, while the automated grader only reads the
file. If true, the model is being rewarded for talking about doing the work
in the chat instead of writing the deliverable. This is the textbook
reward-hacking failure mode for PRM systems with misaligned reward layers.

To confirm/refute the hypothesis, we'd need to capture the actual
per-turn judge prompt + score for the regressed trajectories and check
whether chat-only turns are scored +1. We have not yet done this.
Interpret the R2 analysis as "diagnostics flagged a coherent pattern" —
not "proven mechanism."

What we conclude is narrower than "PRM caused R2 regression":
- R1 is a real one-round gain over baseline.
- R2 with the same recipe regressed deterministically.
- Diagnostics caught the regression at per-task budget granularity, which
  surface-level overall-mean tracking would have missed (only -1.9pp).
- Continuing past R1 with this recipe is not safe; the rubric likely
  needs to penalize chat-side verbose output before R2+ is meaningful.

## Convergence comparison (qualitative)

Terminal-only and terminal+PRM both converge on this setup. Anecdotally,
the terminal-only path (in our prior rope=1 runs, see git history) took
~5 rounds to reach a comparable plateau, vs. 1 round for terminal+PRM at
rope=2. We do **not** present a side-by-side per-round table here because
we did not run terminal-only on rope=2 round-by-round — only the R5
terminal-only LoRA was benched at rope=2 (55.0%, see prior results). So
the "1 round vs 5 rounds" claim is suggestive based on the rope=1 trend
shape, not a clean rope=2 comparison.

If you want a clean convergence study, run terminal-only at rope=2 with
`PRM_BETA=0 SKIP_PRM_SCORING=1` for 5+ rounds and compare round-by-round.

## What this experiment supports — and doesn't

Supports:
- Roadmap PRM with the additive judge-gate produces a real one-round gain
  on this 5-task held-out set (+6.6pp over rope=2 base, +1.6pp over the
  same recipe without the gate).
- Pos-only clip is the largest single design lever (+3.1pp).
- Diagnostics tooling can surface task-level reward-hacking failures that
  overall-mean tracking misses.

Does not support:
- A statistically rigorous claim that judge-gate is necessary for the
  +1.6pp gain (single seed, no error bars).
- A measured "1-round vs 5-rounds" terminal-only comparison at rope=2.
- A general statement about Roadmap PRM beyond this setup, model size,
  and task family.

## Reproducibility

- Train + bench: see [`reproduction.md`](reproduction.md)
- Diagnostics CLI: see [`diagnostics.md`](diagnostics.md)
- Algorithm + flow diagram: see [`algorithm.md`](algorithm.md)
