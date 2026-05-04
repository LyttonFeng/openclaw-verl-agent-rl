# Data

Task16 prompt parquet files are committed under `data/task16_prompts/`.

Current committed data:

- `data/task16_prompts/train.parquet`: 91 rows = 71 canonical wording rows + 20 synthetic task16-style inbox instances
- `data/task16_prompts/train_small.parquet`: 32 rows
- `data/task16_prompts/train_tiny.parquet`: 16 rows
- `data/task16_prompts/train_synth20.parquet`: 20 rows, synthetic inbox instances only
- `data/task16_prompts/train_stage2_balanced.parquet`: 32 rows = 12 canonical focused rows + 20 synthetic inbox instances
- `data/task16_prompts/val.parquet`: 11 rows
- `data/task16_prompts/val_synth5.parquet`: 5 held-out synthetic inbox instances

Regenerate them with:

```bash
python scripts/build_task16_prompts.py \
  --tasks-dir pinchbench_tasks \
  --output-dir data/task16_prompts
python scripts/check_data.py data/task16_prompts
```

Expected output:

- `train.parquet`: 91 rows
- `train_small.parquet`: 32 rows
- `train_tiny.parquet`: 16 rows
- `train_synth20.parquet`: 20 rows
- `train_stage2_balanced.parquet`: 32 rows
- `val.parquet`: 11 rows
- `val_synth5.parquet`: 5 rows

Synthetic rows include per-row `extra_info.workspace_files` and
`reward_rubric.expected_bindings`. The agent loop uses these workspace files to
seed a different inbox per episode.
