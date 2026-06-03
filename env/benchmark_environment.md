# Benchmark Environment

This file documents the environment needed to reproduce the isolated meeting-analysis Val5 benchmark.

Reference pod:

```text
ssh root@154.54.102.37 -p 15877 -i ~/.ssh/id_ed25519
```

## Runtime

Reference pod runtime:

- Python env: `/root/openclaw-venv`
- Python: 3.10.12
- OpenClaw: 2026.4.5 (3e72c03)
- GPU: NVIDIA A100-SXM4-80GB
- CUDA reported by `nvidia-smi`: 13.0
- Core Python packages are pinned in `env/requirements.txt`.

`DEEPSEEK_API_KEY` must be exported for the judge and for DeepSeek model baselines.

```bash
source /root/openclaw-venv/bin/activate
source ~/.pinchbench_env
openclaw --version
```

## Isolated OpenClaw Mode

Use:

```bash
bash scripts/run_val5_bench_isolated.sh
```

The wrapper creates private runtime state for each benchmark run:

- `OPENCLAW_HOME`
- `PINCHBENCH_OPENCLAW_HOME`
- `PINCHBENCH_RUN_ROOT`
- `PINCHBENCH_AGENT_SUFFIX`

This prevents stale agents, sessions, and workspaces from contaminating scores.

## Default Protocol

`scripts/run_val5_bench_isolated.sh` defaults:

```bash
RUNS=3
JUDGE_MODEL=deepseek-v4-pro
TASKS_DIR=data/eval/val5
PINCHBENCH_SKILL_DIR=data/eval
PINCHBENCH_OPENCLAW_CONTEXT_WINDOW=65536
PINCHBENCH_OPENCLAW_MAX_TOKENS=8192
```

The wrapper disables:

- upload
- parallel judge
- judge cache
- fail-fast

## DeepSeek API Baseline

Example:

```bash
RUN_ID=dsv4_pro_temp0 \
MODEL=deepseek-v4-pro \
BASE_URL=https://api.deepseek.com/v1 \
PINCHBENCH_MODEL_TEMPERATURE=0 \
OUTPUT_DIR=results/val5_isolated/dsv4_pro_temp0 \
bash scripts/run_val5_bench_isolated.sh
```

## Local vLLM Baseline

Reference Qwen3-4B vLLM process on the pod:

```bash
CUDA_VISIBLE_DEVICES=0 \
/root/openclaw-venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model /workspace/qwen_models/qwen3-4b-instruct-2507 \
  --served-model-name qwen3-4b-instruct-2507 \
  --host 127.0.0.1 \
  --port 8767 \
  --max-model-len 40960 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Then run benchmark:

```bash
RUN_ID=qwen3_4b_temp0 \
MODEL=qwen3-4b-instruct-2507 \
BASE_URL=http://127.0.0.1:8767/v1 \
PINCHBENCH_MODEL_API_KEY=dummy \
PINCHBENCH_MODEL_TEMPERATURE=0 \
OUTPUT_DIR=results/val5_isolated/qwen3_4b_temp0 \
bash scripts/run_val5_bench_isolated.sh
```

## Result Document

Baseline table is tracked in:

- `docs/isolated_val5_temp0_baseline_results.md`
