# SETUP — committee-reward RL (qwen3.5-4b) follow-up guide

Read `docs/committee_rl_methodology.md` first for the why/architecture. This file is the
practical "how to make it run" — keys, model, path conventions.

## 1. API keys (`~/.pinchbench_env`)

The committee judge calls 3 hosted models. Keys live in `~/.pinchbench_env` (NOT in git — shared
privately). Format is `export NAME='value'` (single quotes), one per line:

```
export DEEPSEEK_API_KEY='...'     # deepseek-v4-flash / -pro (api.deepseek.com)
export DASHSCOPE_API_KEY='...'    # qwen3-max (dashscope compatible-mode)
export MINIMAX_API_KEY='...'      # MiniMax-M3 (api.minimaxi.chat)
```

`lib.py:load_key` reads this file; override the path with `PINCHBENCH_ENV=/path/to/env` if needed.
The same 3 keys are needed on whatever host runs the judge (we run it on the pod so the whole pipeline
is self-contained). Quick check (no values printed):
```
grep -oE 'export [A-Z_]+_API_KEY' ~/.pinchbench_env
python3 -c "import lib; print(lib.family_json('minimax',[{'role':'user','content':'JSON only {\"ok\":1}'}],max_tokens=50))"
```

## 2. Model download (qwen3.5-4b)

`Qwen3_5ForConditionalGeneration` (served as a text model). Download to a **LOCAL** path, e.g.
`/tmp/qwen3.5-4b`:
```
huggingface-cli download <qwen3.5-4b repo> --local-dir /tmp/qwen3.5-4b
```
**NEVER put the model (or any large ckpt) on MFS / network disk** — MFS silently corrupts large
downloads (we hit a "filename-copy-error" that was actually a corrupted weight, not a model bug).
Verify: `python3 -c "import json;print(json.load(open('/tmp/qwen3.5-4b/config.json'))['architectures'])"`.

vLLM is installed but **NOT used for the LoRA** (its online-LoRA is a no-op on qwen3.5). Serving is via
the transformers shim only.

## 3. Path conventions (what the scripts assume)

| What | Path |
|---|---|
| Repo (on pod) | `/workspace/openclaw-naive-meeting-analysis-github` (branch `naive_ppo_qwen35`) |
| Base model | `/tmp/qwen3.5-4b` |
| Judge lib (`lib.py`) | committed at `scripts/tf_agentic/lib.py`; point `JUDGE_LIB_DIR` at that dir (or copy to `/tmp/judge_lib`) |
| Shim (copy of tf_shim.py) | `/tmp/tf_shim.py`; logs `/tmp/shim_*.log` |
| Tasks file | `data/meeting_analysis_val3_slim_train/val3_plus6_train.json` |
| Per-round dir | `/tmp/nma_round1/<RUN_NAME>/` |
| ├ rollout transcripts | `<RUN>/rollouts/transcripts/<task>_resp<k>.jsonl` |
| ├ agent workspaces | `<RUN>/rollouts/workspaces/<task>_resp<k>/` (output file lives here for grading) |
| ├ flash-graded rollouts | `<RUN>/rollouts/graded_trajectories.jsonl` (has `response`,`automated_score`,`timed_out`) |
| ├ blended GRPO scores | `<RUN>/graded_blend.jsonl` (`score`=AUTO_W·auto+(1-AUTO_W)·committee) |
| ├ logprobs | `<RUN>/rollout_logprobs.jsonl` |
| └ trained adapter | `<RUN>/checkpoint/lora_adapter/` |
| base-ref (calibration) | a base-model graded file, e.g. `/tmp/nma_round1/val3plus6_w1e/rollouts/graded_trajectories.jsonl` |
| Eval output | `/tmp/eval_val3_{base,lora}/0001_transcripts/` |
| Adapter backups | `/workspace/saved_adapters/<name>/` (persistent; `/tmp` is ephemeral) |

Env knobs: `ROLLOUT_TIMEOUT_MULT` (rollout timeout = task.timeout_seconds × this, default 4.0),
`AUTO_W` (blend weight on automated, default 0.5), `INIT_LORA` (continue-train from an adapter),
`SHIM_DEFAULT_TEMP` (rollout 1.0 / eval 0), `PINCHBENCH_FORCE_LOCAL_OPENCLAW=1` + unset
`OPENCLAW_HOST`/`ECS_HOST` (use local OpenClaw, not remote ECS).

## 4. Run one on-policy round
```
# on the pod (self-contained nohup; RunPod ssh is unstable — never host the pipeline over ssh):
cp scripts/tf_agentic/tf_shim.py /tmp/tf_shim.py
cp scripts/tf_agentic/lib.py /tmp/judge_lib/lib.py   # or set JUDGE_LIB_DIR=scripts/tf_agentic
nohup bash scripts/tf_agentic/run_onpolicy.sh > /tmp/round.log 2>&1 &
# edit ADAPTER/TASKS/BASE_REF/AUTO_W at the top of run_onpolicy.sh first.
```
Then eval: `bash scripts/tf_agentic/eval_val3_adapter.sh lora <adapter>` → pull transcripts → grade with
`committee_judge.py` (pairwise) + `stable_rejudge.py` (automated/hybrid). **Always verify the live shim's
adapter** (`tr '\0' '\n' </proc/$(pgrep -f tf_shim)/environ | grep LORA_ADAPTER`) and that transcripts
are fresh before trusting eval numbers.
```
```
