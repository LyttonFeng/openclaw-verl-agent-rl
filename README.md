# openclaw-verl-agent-rl

This repo reproduces task16 email triage RL training using:

- Qwen3-1.7B by default, Qwen3-4B supported by `VERL_MODEL`
- veRL REINFORCE++
- OpenClaw multi-turn agent loop
- PinchBench task16 grader
- turn-level process reward + terminal reward
- no `masked_whiten`

The repo is intentionally narrow: task16 RL only. It does not include agent team,
wolfpack, DPO, task18, slides, history, or exploration notes.

## Start Here

1. `README.md`
2. `docs/reproduction.md`
3. `scripts/run_task16_rl.sh`

Everything else is implementation or debugging support.

## Repository Map

- `configs/task16_rl.env.example`: training and evaluation environment template
- `data/task16_prompts/`: committed train/validation parquet and JSONL data
- `pinchbench_tasks/task_16_email_triage.md`: canonical task16 fixture
- `scripts/build_task16_prompts.py`: regenerate task16 prompt data
- `scripts/check_env.py`: verify Python, veRL/vLLM, judge, and OpenClaw reachability
- `scripts/check_data.py`: verify committed data files and row counts
- `scripts/test_task16.py`: local unit smoke for data, reward, and verifier logic
- `scripts/run_task16_rl.sh`: main training entrypoint
- `scripts/run_task16_eval.sh`: evaluation entrypoint/checklist
- `agent_loop/`: OpenClaw agent loop, model proxy, trajectory helpers
- `rewards/task16_event_reward.py`: task16 process/terminal reward
- `rl/train/`: veRL launcher and reward manager glue
- `patches/`: veRL/Ray compatibility patches used by the training script

## Required Environment

Training host:

- A100 80G recommended
- Python 3.12
- CUDA/vLLM-compatible environment
- veRL `0.7.1`
- vLLM `0.10.2`
- Transformers `4.57.1`
- Torch `2.8.0+cu128` verified on RunPod A100
- `pyarrow` for parquet

OpenClaw runtime:

- OpenClaw installed locally or on a reachable ECS host over SSH
- Reference OpenClaw CLI: `2026.4.5 (3e72c03)`
- Fresh per-episode workspace under `/tmp/pinchbench`
- Default/global skills disabled for task16 training

Grading:

- DashScope API key
- Judge model: `qwen-plus`

## Data

Committed data lives in `data/task16_prompts/`.

Default training/eval files:

- `train.parquet`: 91 rows
- `val.parquet`: 11 rows

Additional files:

- `train_small.parquet`: 32-row smaller ablation set
- `train_tiny.parquet`: 16-row smoke/debug set
- `train_canonical32_readexplicit.parquet`: 32 canonical read-explicit prompts
- `train_synth20.parquet`: 20 synthetic task16-style inbox instances
- `train_stage2_balanced.parquet`: 32 rows, 12 canonical focused + 20 synthetic
- `val_canonical5_readexplicit.parquet`: 5 canonical validation prompts
- `val_synth5.parquet`: 5 synthetic validation instances for diagnostics

The synthetic rows are not prompt-only variants. They carry
`extra_info.workspace_files`, so each episode gets its own 13-email inbox.

Regenerate data only when changing prompts or the canonical task:

```bash
python scripts/build_task16_prompts.py \
  --tasks-dir pinchbench_tasks \
  --output-dir data/task16_prompts
```

Verify data:

```bash
python scripts/check_data.py data/task16_prompts
python scripts/test_task16.py --data-dir data/task16_prompts
```

## Setup

Install dependencies:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Configure the environment:

```bash
cp configs/task16_rl.env.example ~/.pinchbench_env
chmod 600 ~/.pinchbench_env
```

Edit `~/.pinchbench_env` and set:

- `OPENCLAW_HOST`
- `OPENCLAW_USER`
- `OPENCLAW_PORT`
- `OPENCLAW_SSH_KEY`
- `DASHSCOPE_API_KEY`

Then load it:

```bash
set -a
source ~/.pinchbench_env
set +a
```

Check the environment:

```bash
python scripts/check_env.py
```

## Train

Dry-run the exact veRL command:

```bash
bash scripts/run_task16_rl.sh --dry-run
```

Run a one-step smoke:

```bash
TOTAL_TRAINING_STEPS=1 TEST_FREQ=999 SAVE_FREQ=999 bash scripts/run_task16_rl.sh
```

Run the short reproduction:

```bash
TOTAL_TRAINING_STEPS=32 TEST_FREQ=4 SAVE_FREQ=4 bash scripts/run_task16_rl.sh
```

Default training settings:

- `VERL_MODEL=Qwen/Qwen3-1.7B`
- `BATCH_SIZE=1`
- `LORA_RANK=32`
- `LR=2e-5`
- `ROLLOUT_TEMPERATURE=0.8`
- `ROLLOUT_TOP_P=0.9`
- `VAL_DO_SAMPLE=False`
- `MAX_PROMPT_LENGTH=20000`
- `MAX_RESPONSE_LENGTH=12000`
- `VLLM_MAX_MODEL_LEN=32768`

For Qwen3-4B:

```bash
VERL_MODEL=Qwen/Qwen3-4B \
RUN_VERSION=task16_qwen4b_debug \
TOTAL_TRAINING_STEPS=32 \
TEST_FREQ=4 \
SAVE_FREQ=4 \
bash scripts/run_task16_rl.sh
```

## Evaluate

Use `scripts/run_task16_eval.sh` as the evaluation checklist:

```bash
bash scripts/run_task16_eval.sh
```

For LoRA checkpoint evaluation:

1. Serve the base model plus LoRA adapter with `scripts/start_vllm_lora.sh`.
2. Run task16 through the same OpenClaw + PinchBench grader path.
3. Save transcripts and grader JSON under `logs/` or an external artifacts path.

## Debug Scripts

These are useful when training behaves strangely, but they are not the main
reproduction path:

- `scripts/check_train_infer_parity.py`: verifies task16 prompt extraction parity
- `scripts/check_ecs_harness_gate.py`: strict ECS/OpenClaw harness preflight
- `scripts/run_task16_step0_gate.sh`: veRL training-side validation-only smoke
- `scripts/analyze_task16_terminal_gate.py`: regrade task16 transcripts
- `scripts/extract_training_metrics.py`: extract validation/reward curves from logs

Known status: this repo is a debug-ready minimal reproduction. It is intended to
make the training and evaluation path portable first; it does not claim a
guaranteed improving checkpoint on every run.

## Reference Versions

- PinchBench benchmark: `1.2.1`
- veRL: `0.7.1`
- vLLM: `0.10.2`
- Transformers: `4.57.1`
- Torch: `2.8.0+cu128`
- OpenClaw CLI: `2026.4.5 (3e72c03)`
- Default model: `Qwen/Qwen3-1.7B`
- Judge: DashScope `qwen-plus`
