# slim12 → Val3 RL: reproducible run recipe

Round-by-round GRPO/PPO LoRA training on the 12 `task_meeting_*` slim tasks,
evaluated on the 3 Val3 tasks (transfer). Validated on a single A100 with
Qwen3-4B-base served by vLLM.

## Why the bare runner needs env overrides

`run_naive_ppo_round.sh` does NOT work out-of-the-box; its defaults are wrong
for this setup. Always pass:

| env | why | example (this pod) |
|-----|-----|--------------------|
| `PYTHON_BIN` | default `python3` lacks `yaml`/`torch`; use the project venv | `/root/openclaw-venv/bin/python` |
| `MODEL_PATH` | default `Qwen/Qwen3-4B` downloads from HF; use local weights | `/workspace/qwen_models/qwen3-4b` |
| `ROLLOUT_MODEL` | **must equal the vLLM `--served-model-name`**, else rollouts 404 → all fatal | `Qwen3-4B-base` (round 1) / `r<N-1>-lora` (flywheel) |
| `TRAIN_SPLIT` | default is `all_samples`, not slim12 | `data/train/meeting_analysis_slim12_split.json` |
| hyperparams | conservative to avoid base-policy drift (see below) | `LR=2e-6 KL_BETA=0.05 REF_KL_BETA=0.05 LORA_RANK=16 N_RESPONSES=4` |

## vLLM lifecycle (critical)

- The runner kills vLLM before `[3/4]` (single-GPU). So **vLLM is down after every
  round** — you must (re)start it before the next rollout / before eval.
- **LoRA cannot be hot-loaded.** To roll out or eval a LoRA you must COLD-START
  vLLM with base + LoRA loaded together (`--enable-lora --lora-modules name=path`).
  A long-lived / hot-modified vLLM instance degrades and yields all-fatal rollouts.
- Keep the agent suffix short (OpenClaw truncates long ids → transcript lookup fails).

## Round 1 (from base)

```bash
# 1. fresh base-only vLLM
setsid $PYTHON_BIN -m vllm.entrypoints.openai.api_server \
  --model $MODEL_PATH --served-model-name Qwen3-4B-base --host 127.0.0.1 --port 8021 \
  --max-model-len 40960 --gpu-memory-utilization 0.85 --dtype bfloat16 --trust-remote-code \
  --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 &

# 2. rollout + select(process gate) + train
RUN_NAME=r1 PYTHON_BIN=... MODEL_PATH=... ROLLOUT_MODEL=Qwen3-4B-base \
  TRAIN_SPLIT=.../meeting_analysis_slim12_split.json \
  LR=2e-6 KL_BETA=0.05 REF_KL_BETA=0.05 LORA_RANK=16 N_RESPONSES=4 \
  bash train/run_naive_ppo_round.sh        # -> results/train/<run>/checkpoint/lora_adapter

# 3. eval (cold-starts vLLM with the LoRA, tech smoke gate, then Val3)
bash train/eval_lora_val3.sh results/train/<run>/checkpoint/lora_adapter r1-lora
```

## Round N>1 (flywheel: roll out the previous LoRA, continue-train from it)

```bash
# fresh COLD-START vLLM with base + previous LoRA together (NOT hot-load)
setsid $PYTHON_BIN -m vllm.entrypoints.openai.api_server ... \
  --enable-lora --max-loras 1 --max-lora-rank 16 \
  --lora-modules r<N-1>-lora=<prev_lora_path> &

RUN_NAME=rN ROLLOUT_MODEL=r<N-1>-lora PREV_LORA=<prev_lora_path> \
  LR=1e-6 KL_BETA=0.05 REF_KL_BETA=0.05 ... bash train/run_naive_ppo_round.sh
# ROLLOUT_MODEL and PREV_LORA must be the SAME policy (on-policy: P_old matches rollouts).
```

## What the fixes do

- `f5690a3` **per-rollout agent**: each rollout gets a fresh OpenClaw agent, so one
  wedged rollout (e.g. a long-doc context overflow) can't cascade the rest to fatal.
- `2b77b3f` **process gate**: `generate_meeting_rollouts` records agentic-behavior
  features (`context_overflow`, `compaction_before_write`, `read_without_write`, …);
  `select_grpo_samples` drops trajectories whose process is not worth imitating
  (no output / read-only / overflow / compaction-before-write / output<100 chars).
- `eval_lora_val3.sh` gates on a `tech_action_items` behavior smoke: any 0-score run
  ⇒ do not promote / skip full Val3.

## Result log (this pod)

- baseline (Qwen3-4B-base): Val3 overall **50.9%** (advisory 0.441 / gov_speaker 0.443 / tech 0.642)
- prior broken LoRA: 48.0% (tech reproducibly 0)
- R1 (LR 2e-6, from base): **50.9%** — tech 0.629 (no 0s, collapse fixed), gov_speaker 0.542 (+0.10),
  advisory 0.356 (regressed, ~1/3 no-output). Net flat; advisory completion is the bottleneck.
