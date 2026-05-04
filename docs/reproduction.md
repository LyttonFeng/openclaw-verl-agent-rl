# Reproduction

This document expands the README checklist for anyone who needs to run, inspect,
or port the task16 RL path.

## 1. Environment

Training machine:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Expected package versions:

- `verl==0.7.1`
- `vllm==0.10.2`
- `transformers==4.57.1`
- `torch==2.8.0+cu128`
- `pyarrow` installed

Runtime services:

- OpenClaw reachable over SSH
- DashScope `qwen-plus` judge reachable from the training machine
- Hugging Face model weights available for `Qwen/Qwen3-1.7B` or `Qwen/Qwen3-4B`

## 2. Configuration

Use `configs/task16_rl.env.example` as the complete list of knobs:

```bash
cp configs/task16_rl.env.example ~/.pinchbench_env
chmod 600 ~/.pinchbench_env
```

Required values:

```bash
export OPENCLAW_HOST=<ecs-or-local-host>
export OPENCLAW_USER=root
export OPENCLAW_PORT=22
export OPENCLAW_SSH_KEY=/root/.ssh/id_ed25519
export DASHSCOPE_API_KEY=<dashscope-api-key>
```

Recommended task16 runtime values:

```bash
export PINCHBENCH_AGENT_LOOP_RUNTIME=benchmark
export PINCHBENCH_DISABLE_DEFAULT_SKILLS=1
export PINCHBENCH_CLEANUP_REMOTE_AGENT=1
export PINCHBENCH_CLEANUP_REMOTE_WORKSPACE=1
export OPENCLAW_MODEL_REASONING=0
export MAX_PROMPT_LENGTH=20000
export MAX_RESPONSE_LENGTH=12000
export VLLM_MAX_MODEL_LEN=32768
export PINCHBENCH_OPENCLAW_MAX_TOKENS=8192
```

Why these matter:

- The training rollout executes OpenClaw episodes on the configured runtime host.
- Each task16 episode should get a fresh OpenClaw home and fresh workspace.
- The workspace is synced back before PinchBench grading.
- `triage_report.md` must exist for task16 terminal grading.

## 3. Data

The repo commits the data needed for reproduction:

```text
data/task16_prompts/train.parquet
data/task16_prompts/val.parquet
```

Expected row counts:

```text
train.parquet                         91
train_small.parquet                   32
train_tiny.parquet                    16
train_canonical32_readexplicit.parquet 32
train_synth20.parquet                 20
train_stage2_balanced.parquet         32
val.parquet                           11
val_canonical5_readexplicit.parquet    5
val_synth5.parquet                     5
```

Check:

```bash
python scripts/check_data.py data/task16_prompts
python scripts/test_task16.py --data-dir data/task16_prompts
```

Regenerate:

```bash
python scripts/build_task16_prompts.py \
  --tasks-dir pinchbench_tasks \
  --output-dir data/task16_prompts
```

## 4. Training

Dry run:

```bash
bash scripts/run_task16_rl.sh --dry-run
```

One-step smoke:

```bash
TOTAL_TRAINING_STEPS=1 TEST_FREQ=999 SAVE_FREQ=999 bash scripts/run_task16_rl.sh
```

Short run:

```bash
TOTAL_TRAINING_STEPS=32 TEST_FREQ=4 SAVE_FREQ=4 bash scripts/run_task16_rl.sh
```

Important defaults:

- REINFORCE++ via veRL
- LoRA rank 32
- batch size 1
- stochastic training rollout
- deterministic validation rollout
- turn-level reward broadcast
- terminal reward from PinchBench task16 grader
- no `masked_whiten`

Checkpoints are written under `checkpoints/`.

## 5. Evaluation

Basic eval checklist:

```bash
bash scripts/run_task16_eval.sh
```

Checkpoint eval flow:

1. Start vLLM with the base model and LoRA adapter:

```bash
LORA_PATH=<checkpoint>/actor/lora_adapter \
bash scripts/start_vllm_lora.sh
```

2. Run task16 through OpenClaw + PinchBench grading.

3. Preserve:

- transcript JSONL
- synced workspace containing `triage_report.md`
- grader JSON
- training log
- tensorboard/event logs if present

## 6. Code Entry Points

Training command:

- `scripts/run_task16_rl.sh`
- `rl/train/run_reinforce_lora.sh`
- `rl/train/launch_main_ppo.py`

Rollout:

- `agent_loop/openclaw_agent_loop.py`
- `agent_loop/model_proxy.py`
- `agent_loop/trajectory.py`

Reward:

- `rewards/task16_event_reward.py`
- `rl/train/reward_manager.py`

Data:

- `scripts/build_task16_prompts.py`
- `pinchbench_tasks/task_16_email_triage.md`

Evaluation/debug:

- `scripts/run_task16_eval.sh`
- `scripts/analyze_task16_terminal_gate.py`
- `scripts/extract_training_metrics.py`

## 7. Current Status

This is a minimal, portable reproduction/debug repo. It is suitable for:

- setting up the veRL/OpenClaw/PinchBench task16 training path
- inspecting data and reward logic
- running short RL experiments
- collecting transcripts and validation curves

It is not a polished benchmark result repo and does not guarantee that every run
will improve over the base model.
