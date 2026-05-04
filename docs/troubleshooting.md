# Troubleshooting

## Data row count mismatch

Run:

```bash
python scripts/build_task16_prompts.py --tasks-dir pinchbench_tasks --output-dir data/task16_prompts
python scripts/check_data.py data/task16_prompts
```

Expected counts are 91 train rows, 20 synthetic-only rows, and 11 validation rows.

## veRL import or reward manager errors

Use veRL `0.7.1`. The training script relies on importlib reward manager loading:

```bash
python -c 'import verl; print(verl.__version__)'
```

## OpenClaw SSH preflight fails

Reference OpenClaw CLI version is `2026.4.5 (3e72c03)`.

Verify:

```bash
ssh -i "$OPENCLAW_SSH_KEY" -p "$OPENCLAW_PORT" "$OPENCLAW_USER@$OPENCLAW_HOST" \
  'command -v openclaw && openclaw --version'
```

If OpenClaw needs shell activation, set `OPENCLAW_REMOTE_ACTIVATE_CMD`.

## DashScope grading fails

Set one of:

```bash
export DASHSCOPE_API_KEY=<key>
export PINCHBENCH_GRADE_JUDGE_API_KEY=<key>
```

Default judge model is `qwen-plus`.

## OOM

Start by lowering:

```bash
export VLLM_GPU_MEM_UTIL=0.18
export VLLM_MAX_NUM_SEQS=4
export MAX_RESPONSE_LENGTH=8192
export PINCHBENCH_TASK16_MAX_TOKENS_PER_TURN=1536
```
