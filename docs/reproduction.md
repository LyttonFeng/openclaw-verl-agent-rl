# Reproduction

End-to-end recipe to reproduce the meeting_analysis GRPO training in this repo.
Targets: same 5-task bench score as reported in [`experiment_report.md`](experiment_report.md).

Two training paths are supported via the same wrapper:

- **Terminal-only** (no PRM, baseline): `PRM_BETA=0`
- **Terminal + PRM** (Roadmap PRM, judge-gate): default settings

## 1. Environment

| Component | Version | Notes |
|---|---|---|
| Python | 3.12 | venv recommended |
| veRL | 0.7.1 | offline GRPO trainer dependency |
| vLLM | 0.10.2 | with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` for hot-load |
| Transformers | 4.57.1 | |
| PEFT | latest compatible | LoRA training |
| Torch | 2.8.0+cu128 | tested on A100 |
| OpenClaw CLI | `2026.4.5` (3e72c03) | runs the multi-turn agent locally |
| PinchBench | 1.2.1 | task definitions + grader |
| GPU | 2 × A100-80GB | GPU 0 = training, GPU 1 = vLLM |

> **No ECS / external runtime needed.** OpenClaw runs locally on the same pod
> as training. SSH-to-OpenClaw mode (used by older task16 path) is not used here.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

API keys (DeepSeek required for both terminal LLM-judge and PRM judge):

```bash
cat > ~/.pinchbench_env <<'EOF'
export DEEPSEEK_API_KEY=sk-xxxxxxxxxxxx
EOF
chmod 600 ~/.pinchbench_env
```

## 2. Data

Train/test split definition:

```text
rl/train/meeting_analysis_split.json
```

| Split | Tasks | Source meetings (4 real transcripts) |
|---|---|---|
| **Train** | 23 tasks | NTIA spectrum advisory (71KB), GitLab PMM (34KB), Tampa City Council (206KB), NASA UAP hearing (265KB) |
| **Test** | 5 tasks | same 4 transcripts, held-out tasks: `advisory_stakeholders`, `council_votes`, `gov_speaker_summary`, `tech_action_items`, `sentiment_analysis` |

Task definitions live under `pinchbench_tasks/meeting_analysis/`.
Transcripts under `assets/meetings/`.
Roadmaps (per-task expert milestones for PRM) under `agent_loop/roadmap_prm/roadmaps/`.

Build training prompts (parquet) once:

```bash
python rl/train/build_meeting_analysis_prompts.py \
    --tasks-dir pinchbench_tasks/meeting_analysis \
    --split-file rl/train/meeting_analysis_split.json \
    --output-dir data/meeting_prompts
```

## 3. Start vLLM (GPU 1)

```bash
CUDA_VISIBLE_DEVICES=1 \
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B \
    --served-model-name Qwen3-4B \
    --host 0.0.0.0 --port 8021 \
    --max-model-len 81920 \
    --rope-scaling '{"type":"dynamic","factor":2.0}' \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 \
    --trust-remote-code \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    --enable-lora --max-loras 1 --max-lora-rank 16
```

> Important: `rope-scaling factor=2.0` and `max-model-len=81920` must match
> what training uses. Mismatched rope between rollout and train is the most
> common reproducibility trap.

## 4. Training

### One round, terminal + PRM (default, recommended)

```bash
ROUND_NUM=1 bash rl/train/run_meeting_grpo_prm_round.sh
```

This runs the full pipeline:

1. Generate 23 × 2 = 46 rollouts (4 parallel workers)
2. Grade with terminal reward (automated + DeepSeek LLM judge)
3. PRM-score each trajectory (judge-gate + per-turn judge)
4. Variance filter + pos-only clip (-1 → 0)
5. Single GRPO step (15 updates, batch=2, lr=2e-6)
6. Save LoRA adapter to `/workspace/meeting_grpo_prm_v1/round_1/checkpoint/lora_adapter`

### One round, terminal-only (baseline)

```bash
PRM_BETA=0 \
ROUND_NUM=1 \
EXPERIMENT=meeting_grpo_terminal_v1 \
bash rl/train/run_meeting_grpo_prm_round.sh
```

`PRM_BETA=0` makes the PRM scores have no effect on advantage; PRM scoring
still runs (it's cheap) but the GRPO update reduces to terminal-only.

### Continue from a previous LoRA

```bash
ROUND_NUM=2 \
PREV_LORA=/workspace/meeting_grpo_prm_v1/round_1/checkpoint/lora_adapter \
bash rl/train/run_meeting_grpo_prm_round.sh
```

### Key knobs

| Variable | Default | Meaning |
|---|---|---|
| `ROUND_NUM` | required | round counter |
| `PREV_LORA` | empty | start from base if empty, else LoRA path |
| `EXPERIMENT` | `meeting_grpo_prm_v1` | output subdir name |
| `PRM_ALPHA` | `1.0` | terminal weight |
| `PRM_BETA` | `0.10` | PRM weight (0 = terminal-only) |
| `PRM_MODE` | `additive` | `additive` or `multiplicative` |
| `N_RESPONSES` | `2` | rollouts per prompt (GRPO group size) |
| `NUM_WORKERS` | `4` | parallel rollout workers |
| `MAX_SEQ_LEN` | `81920` | training sequence cap (must match vLLM) |

## 5. Bench / evaluate

After training, the wrapper hot-loads the new LoRA into vLLM and runs a 3-run
bench on the 5 test tasks. Result lands at:

```text
/workspace/<EXPERIMENT>/bench_<round_tag>_v2_<timestamp>/result.json
```

To run bench manually on a saved LoRA:

```bash
curl -X POST http://127.0.0.1:8021/v1/load_lora_adapter \
    -H "Content-Type: application/json" \
    -d '{"lora_name":"meeting-r1","lora_path":"<lora_path>"}'

OPENAI_API_KEY=$DEEPSEEK_API_KEY \
OPENAI_BASE_URL=https://api.deepseek.com/v1 \
python scripts/benchmark.py \
    --suite "task_meeting_advisory_stakeholders,task_meeting_council_votes,task_meeting_gov_speaker_summary,task_meeting_tech_action_items,task_meeting_sentiment_analysis" \
    --model "custom/meeting-r1" \
    --base-url "http://127.0.0.1:8021/v1" \
    --api-key "dummy" \
    --judge "openai/deepseek-chat" \
    --output-dir /workspace/bench_meeting_r1 \
    --runs 3
```

## 6. Diagnose the run

After bench, run the diagnostics module to get per-task analysis (failure
modes, output budget allocation, automated check stability across runs):

```bash
python -m agent_loop.diagnostics analyze \
    --result-json /workspace/<bench_dir>/result.json \
    --transcripts-dirs results/<NNNN>_transcripts \
    --output /workspace/<bench_dir>/diagnosis.md
```

See [`diagnostics.md`](diagnostics.md) for what the report covers.

## 7. Expected results

Reference scores (3-run mean, judge = `openai/deepseek-chat`):

| Config | Overall | Notes |
|---|---|---|
| Baseline (rope=2, no LoRA) | **50.6%** | apples-to-apples baseline |
| Terminal-only, R5 LoRA | 55.0% | ~5 rounds to converge |
| **Terminal + PRM (additive judge-gate, R1)** | **57.24%** | converges in 1 round |

See [`experiment_report.md`](experiment_report.md) for full per-task breakdown
and ablation history.

## 8. Common pitfalls

- **rope mismatch**: train at rope=2 but bench at rope=1 (or vice versa) gives
  inconsistent scores. Both must be `factor=2.0, max_model_len=81920`.
- **Single-run validation is noisy**: always 3-run for the test set.
- **Don't continue past R2 with the default recipe**: R2-additive regresses
  -1.9pp due to reward hacking (model shifts output chars from `.md` file to
  chat reply). The diagnostics module catches this; see `experiment_report.md` §15.
- **Workspace overwrite**: rollouts share `/tmp/pinchbench/<NNNN>/agent_workspace`.
  Each rollout snapshots its workspace before the next task overwrites it.
- **Don't skip thinking**: Qwen3-4B tool calling depends on `<think>...</think>`.
  Disabling thinking breaks rollout silently.
