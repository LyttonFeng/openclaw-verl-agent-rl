# Data

This directory holds compiled training prompts. The split definition lives in
`rl/train/meeting_analysis_split.json` (23 train tasks / 5 test tasks across
4 real meeting transcripts).

To regenerate the parquet from task definitions:

```bash
python rl/train/build_meeting_analysis_prompts.py \
    --tasks-dir pinchbench_tasks/meeting_analysis \
    --split-file rl/train/meeting_analysis_split.json \
    --output-dir data/meeting_prompts
```

Inputs consumed by the build:

- `pinchbench_tasks/meeting_analysis/*.md` — 28 task definitions
- `assets/meetings/*.md` — 4 transcripts (NTIA spectrum advisory, GitLab PMM,
  Tampa City Council, NASA UAP hearing)
- `rl/train/meeting_analysis_split.json` — train/test split

Outputs:

- `meeting_prompts/train.parquet` — 23 train tasks
- `meeting_prompts/val.parquet` — 5 held-out test tasks

The compiled parquet is intentionally not committed — re-run the build whenever
the task definitions or split change.
