# Troubleshooting

## vLLM does not load the new LoRA adapter

vLLM must be started with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` and
`--enable-lora`. Without these the `/v1/load_lora_adapter` endpoint silently
no-ops. Verify after a hot-load:

```bash
curl -s http://127.0.0.1:8021/v1/models | python -c "import json,sys; print(json.load(sys.stdin))"
```

The new LoRA name should appear under `data[].id`.

## rope-scaling mismatch between train and bench

vLLM must run `--rope-scaling '{"type":"dynamic","factor":2.0}'` and
`--max-model-len 81920`. Training uses `--rope-scaling-factor 2.0` and
`--max-seq-length 81920`. Mismatched rope is the most common silent
reproducibility trap — the model receives different positional encodings at
train vs inference and produces nonsense scores.

## DeepSeek judge rate-limit / failures

The PRM scoring step calls DeepSeek per turn (4 parallel workers by default).
If you hit rate limits:

```bash
SCORE_MAX_WORKERS=2 \
    python -m agent_loop.roadmap_prm.scripts.score_trajectories \
    --graded-file ... --max-workers 2
```

Check the API key:

```bash
curl -s -H "Authorization: Bearer $DEEPSEEK_API_KEY" https://api.deepseek.com/v1/models | head
```

## Rollout hangs / slow

Concurrency must be ≥ 4. Default `NUM_WORKERS=4` is set in
`rl/train/run_meeting_grpo_prm_round.sh`. If a log line shows
`Concurrency: 1 worker(s)` that means a script was launched without the
parallel flag — kill and relaunch with `NUM_WORKERS=4`.

Also check vLLM GPU utilization (`nvidia-smi`). If GPU 1 is at 100% but
rollout is still slow, the bottleneck is judge calls — increase
`NUM_WORKERS` (also bumps DeepSeek QPS).

## OpenClaw `command not found`

OpenClaw must be installed on the same host as training (no SSH/ECS path is
used). Reference version `2026.4.5 (3e72c03)`:

```bash
which openclaw && openclaw --version
```

If missing: install via the same flow used for the source pod (rsync from
ECS or pip install from the OpenClaw repo).

## Workspace files missing during grading

Bench shares `/tmp/pinchbench/<NNNN>/agent_workspace` across all tasks in one
benchmark.py invocation. The benchmark snapshots each task's workspace
immediately after the rollout, before the next task starts. If grading sees
empty files, verify the snapshot logic in `scripts/benchmark.py` did not
fail silently (look for `Snapshotted workspace to ...` lines).

## Diagnostics says `output_not_written` but the file exists

Diagnostics primarily checks the in-trajectory `write` tool calls, then
falls back to filesystem read. If the workspace was already overwritten by
a later bench task, only the in-trajectory record is reliable. The
diagnostics module already handles this — but if it's reporting wrong, look
at the `task.workspace` field in `result.json` and confirm it points to the
correct NNNN dir.

## OOM during training

Reduce sequence cap or grad-accum:

```bash
MAX_SEQ_LEN=40960 GRAD_ACCUM=1 ROUND_NUM=1 \
    bash rl/train/run_meeting_grpo_prm_round.sh
```

Note: dropping `MAX_SEQ_LEN` below the longest training transcript will
truncate it. Council transcript is 206KB; with rope=2 it just fits at 80K.

## Single-run results don't match the experiment report

The experiment report uses **3-run mean**. Single-run scores have ~5-10pp
variance even on the same checkpoint. Always re-bench with `--runs 3`.
