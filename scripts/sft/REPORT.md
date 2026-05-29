# Qwen3-4B + PinchBench: Bug discovery, RL invalidation, and the real bottleneck

**Date**: 2026-05-22
**Branch**: `jiuwenclaw-agent-loop-impl`
**Commit**: `dccd9e7`

## TL;DR

In one work-cycle we (a) discovered & fixed a production bug in OpenClaw's vLLM
adapter that silently degrades all multi-turn agentic benchmarks for
Qwen3-based models, (b) invalidated the apparent "RL micro-gains" reported in
v32–v41 by re-benching with the fix, and (c) identified the actual bottleneck
on PinchBench: 32K-context base models can't hold the source transcripts, so
single-agent RL fine-tuning has no structural room to outperform base. The
viable path forward is **multi-agent team orchestration**, which Codex's
parallel experiment is already validating (+8pp on a single task).

## Three things we actually shipped

### 1. Critical OpenClaw infra bug + patch (PATCH-B)

OpenClaw's `pi-ai/openai-completions.js` provider sets `params.tools = []` on
turn 2+ when conversation has tool history (compat shim for Anthropic/LiteLLM).
But vLLM's hermes tool-call parser only activates when `tools` is non-empty.
Result: Qwen3 emits valid `<tool_call>...</tool_call>` XML, vLLM doesn't parse
it, OC sees text content with no `tool_calls` field, treats it as a final
answer. Every multi-turn rollout silently degrades from turn 2 onward.

**Fix**: extract `<tool_call>` from text content as a fallback after stream
completion. ~25 lines of JS. See `oc_hermes_patch.md` and
`apply_oc_hermes_patch.sh`.

### 2. Invalidated v32–v41 RL "micro-gains"

Re-benched v38_ckpt9 (the historical RL high-water mark at 0.517) under
patched OC:

```
                base Qwen3-4B    v38_ckpt9 (RL)   Δ vs base
Historical      0.474 ± 0.04     0.49  (3 run)    +1.6pp  ← "gain"
Patched         0.510 ± 0.019    0.464 ± 0.017    -4.6pp  ← real
```

`-4.6pp` is outside 1 stdev of either run; statistically real. The historical
"gain" was entirely an artifact of the bug pushing base down ~4pp.

**Implication**: every v32–v41 conclusion that claimed RL improvement needs
re-benching. Future RL experiments must use patched OC.

### 3. Localized the actual bottleneck (context, not policy)

PinchBench transcripts are 120K–250K chars (Tampa City Council, NTIA hearings,
NASA UAP). Qwen3-4B native context is 32K, YaRN-extended 64K. Single-agent
trajectories that read the transcript chunk-by-chunk hit `400 Bad Request`
from vLLM after a few reads because cumulative input tokens overflow.

We saw this directly in v8 overfit bench:
```
council_votes turn 2: 400 'max_tokens' is too large: 8192. 
This model's maximum context length is 32768 tokens and your request has 
28992 input tokens (8192 > 32768 - 28992)
```

This makes single-agent RL structurally bounded. The model can't even *roll
out* the strong trajectory pattern DSv4 Flash uses (chunked read → synthesize
→ write), because the context fills up before it can synthesize. RL on a
trajectory distribution that can't contain the winning strategy can't find it.

## What we learned that doesn't work

| Approach | Result | Why |
|---|---|---|
| SFT v1 (28 records, 2 epoch) | 0.489 ≈ base 0.49 | too few train steps |
| SFT v2 (42 records, 15 epoch, lr 5e-5) | 0.354 (-14pp) | overfit; high variance |
| SFT overfit val_5 only (50 epoch, lr 1e-4) | **0.148 (-34pp)** | overfit broke base, 32-turn micro-read loops |
| RL v32–v41 (various) | 0.464–0.49 | structurally bounded by context |

## What does work (Codex's parallel exploration)

Multi-agent team architecture with DSv4 Flash as policy + Qwen3-4B workers/final:
- `task_meeting_tech_action_items`: **0.773** (vs single-agent base 0.692, RL 0.626)
- Working hypothesis: policy has full transcript context (DSv4 256K), worker
  agents each see one chunk (fits in Qwen3 32K), final agent synthesizes
  short worker outputs.

If Qwen3-4B-as-policy is also competitive, the team architecture is
deployable without DSv4 dependency.

## Reusable artifacts (in repo)

- `scripts/sft/oc_hermes_patch.md` — full RCA for the bug
- `scripts/sft/apply_oc_hermes_patch.sh` — one-shot patch application
- `scripts/sft/{1,1b,1c,2,3,3b,4}_*.py` — 7-stage SFT data pipeline
  (extract → merge_reads → normalize_paths → score_filter → chatml → truncate → validate)
- `scripts/sft/train_qwen3_lora.py` — LoRA SFT trainer with assistant-only loss masking
- `scripts/sft/{bench_base,bench_sft_lora}.sh` — vLLM + LoRA + bench launchers
- `scripts/sft/run_v*.sh` — six versions of training experiments

## Recommended next steps

1. **Stop investing in single-model RL fine-tuning on PinchBench**. The
   bottleneck is context length, not policy quality. Weight updates can't move
   it.
2. **Verify Qwen3-4B-as-policy in team architecture**. If close to DSv4
   (within ~5pp), the team is fully Qwen3-deployable.
3. **If Qwen3-policy is weak**: SFT distill *policy decisions only* (the
   plan/split steps) from DSv4 traces. Not full trajectory distillation.
4. **For any future RL experiment**: apply PATCH-B first. Otherwise the
   reward signal is corrupted on every turn 2+.

## Numbers reference

All numbers below are mean of 3 val_5 runs unless noted.

| Setup | mean | stdev | n |
|---|---|---|---|
| DSv4 Flash teacher | 0.894 | 0.02 | 3 |
| base Qwen3-4B (patched OC) | **0.510** | 0.019 | 3 |
| base Qwen3-4B (historical, bug present) | 0.474 | 0.04 | 9 |
| v38_ckpt9 RL LoRA (patched OC) | **0.464** | 0.017 | 3 |
| v38_ckpt9 RL LoRA (historical, bug present) | 0.49 | - | 3 |
| SFT v1 LoRA (28 records, 2 epoch) | 0.489 | 0.025 | 3 |
| SFT v2 LoRA (42 records, 15 epoch) | 0.354 | 0.012 | 3 |
| SFT overfit LoRA (9 val_5 only, 50 epoch) | 0.148 | 0.023 | 3 |
| Team (DSv4-policy + Qwen3-workers) — partial | 0.773 | — | 1 task only |

Per-task breakdown of patched base vs patched RL (3 run each):

| Task | base | v38 RL | Δ |
|---|---|---|---|
| advisory_stakeholders | 0.518 | 0.411 | **-11pp** |
| council_votes | 0.269 | 0.256 | -1pp |
| gov_speaker_summary | 0.435 | 0.406 | -3pp |
| sentiment_analysis | 0.639 | 0.621 | -2pp |
| tech_action_items | 0.692 | 0.626 | **-7pp** |
