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
| **veRL** | **0.8.0.dev0** (editable from source) | **NOT a pip release** — see install step below |
| vLLM | 0.10.2 | with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` for hot-load |
| Transformers | 4.57.1 | |
| PEFT | latest compatible | LoRA training |
| Torch | 2.8.0+cu128 | tested on A100 |
| OpenClaw CLI | `2026.4.5` (3e72c03) | runs the multi-turn agent locally |
| PinchBench | 1.2.1 | task definitions + grader (this repo embeds the subset needed) |
| GPU | 2 × A100-80GB | GPU 0 = training, GPU 1 = vLLM |

> **No ECS / external runtime needed.** OpenClaw runs locally on the same pod
> as training. SSH-to-OpenClaw mode (used by older task16 path) is not used here.

### Install Python deps

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Install veRL from source (required)

veRL is **not installed via pip**. Older releases (≤0.7.x) lack the
`agent_loop` integration this repo depends on. Use the dev branch:

```bash
git clone https://github.com/volcengine/verl.git ~/verl
cd ~/verl
# Reference state: 0.8.0.dev0 head as of 2026-04 — pin a commit if you need a
# fixed reference; the agent_loop API has been stable since 2026-03.
pip install -e .
```

Verify the install picked up the editable path:

```bash
python -c "import verl; print(verl.__version__, verl.__file__)"
# expected: 0.8.0.dev0 ~/verl/verl/__init__.py
```

### Install OpenClaw CLI

OpenClaw is the multi-turn agent runtime that drives each rollout. It is a
public **npm package**:

```bash
# Make sure node + npm are available (e.g. via nvm)
node --version && npm --version

# Install globally to local disk — DO NOT install under /workspace or any
# network-mounted FS; OpenClaw has hundreds of transitive deps and a
# network-FS install can take 50+ minutes vs. ~30s on local disk.
npm install -g openclaw@2026.4.5

# If your `node` binary lives in a non-standard location (e.g. nvm puts it
# under /workspace/nvm/...), the openclaw shebang `#!/usr/bin/env node`
# may not find it after `npm install -g`. Symlink node into PATH:
ln -sf "$(which node)" /usr/local/bin/node

# Verify
which openclaw
openclaw --version    # → OpenClaw 2026.4.5 (3e72c03)
```

The installed package is ~1.3 GB on disk (extensions + transitive deps);
the local install dir is `/usr/local/lib/node_modules/openclaw/` by
default. If `OPENCLAW_BIN` is set in your environment, the training
wrapper will respect it instead of looking on PATH.

### API keys

DeepSeek is the default judge for **both** terminal grading and PRM scoring:

```bash
cat > ~/.pinchbench_env <<'EOF'
export DEEPSEEK_API_KEY=sk-xxxxxxxxxxxx
EOF
chmod 600 ~/.pinchbench_env
```

DeepSeek is the only judge provider this repo configures by default. Other
OpenAI-compatible endpoints can be swapped in by overriding
`MEETING_JUDGE_BASE_URL` + `MEETING_JUDGE_MODEL` (and providing the matching
key as `DEEPSEEK_API_KEY` env var — that name is what the judge resolution
chain reads).

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

This runs the full pipeline (each step writes its own log + intermediate file):

1. **Rollouts** — 23 train tasks × 2 responses = 46 trajectories, 4 parallel workers via OpenClaw.
2. **Terminal grade** — automated check + DeepSeek LLM judge → `graded_trajectories.jsonl`.
3. **PRM scoring** — DSv4 judge with terminal-completion gate (skipped if `SKIP_PRM_SCORING=1`) → `graded_trajectories_prm.jsonl`.
4. **Variance filter + pos-only clip** — `select_grpo_samples.py` drops zero-variance groups; default `POS_ONLY_CLIP=1` clips -1 turn scores to 0 → `graded_trajectories_prm_pos_only.jsonl`.
5. **GRPO step** — 15 updates, batch=2, lr=2e-6, mode=`PRM_MODE`. Saves LoRA to `$BASE_DIR/round_1/checkpoint/lora_adapter`.
6. **Hot-load** — `POST /v1/load_lora_adapter` to vLLM with the new adapter (no restart).
7. **3-run bench** — `scripts/benchmark.py` against the 5 held-out test tasks.

### One round, terminal-only (baseline)

```bash
PRM_BETA=0 \
ROUND_NUM=1 \
EXPERIMENT=meeting_grpo_terminal_v1 \
bash rl/train/run_meeting_grpo_prm_round.sh
```

> **Note:** `PRM_BETA=0` makes the PRM scores have no effect on the GRPO
> advantage, so the update reduces to terminal-only. **PRM scoring itself
> still runs** (one DeepSeek call per assistant turn, ~$0.05 per round) —
> it's cheap and useful as a control signal in diagnosis. If you want to
> skip the PRM stage entirely (zero DeepSeek PRM calls), set
> `SKIP_PRM_SCORING=1` (and the wrapper will jump from grade → train,
> using all-zero per-turn scores).

### Continue from a previous LoRA

```bash
ROUND_NUM=2 \
PREV_LORA=/workspace/meeting_grpo_prm_v1/round_1/checkpoint/lora_adapter \
bash rl/train/run_meeting_grpo_prm_round.sh
```

### Key knobs

All overridable via environment variable.

| Variable | Default | Meaning |
|---|---|---|
| `ROUND_NUM` | required | round counter |
| `PREV_LORA` | empty | start from base if empty, else LoRA path |
| `EXPERIMENT` | `meeting_grpo_prm_v1` | output subdir name (under `BASE_DIR`) |
| `BASE_DIR` | auto-detect: `/workspace/$EXPERIMENT` if `/workspace` exists & writable, else `$HOME/grpo_runs/$EXPERIMENT` | run output root |
| `PRM_ALPHA` | `1.0` | terminal weight in advantage formula |
| `PRM_BETA` | `0.10` | PRM weight (0 = terminal-only ablation; PRM scoring still runs unless `SKIP_PRM_SCORING=1`) |
| `PRM_MODE` | `additive` | `additive` or `multiplicative` (formulas in [`algorithm.md`](algorithm.md)) |
| `SKIP_PRM_SCORING` | `0` | set to `1` to skip the DeepSeek PRM judge step entirely (synthesizes all-zero per-turn scores) |
| `POS_ONLY_CLIP` | `1` | clip negative PRM turn scores to 0 before training; set `0` to keep raw `{-1,0,+1}` |
| `VARIANCE_THRESHOLD` | `1e-8` | drop GRPO groups with terminal-score variance below this (no signal anyway) |
| `N_RESPONSES` | `2` | rollouts per prompt (GRPO group size) |
| `NUM_WORKERS` | `4` | parallel rollout workers |
| `MAX_SEQ_LEN` | `81920` | training sequence cap (must match vLLM `--max-model-len`) |
| `MEETING_JUDGE_PROVIDER` | `deepseek` | reserved for future providers; only `deepseek` is wired by default |
| `TASKS_DIR` | `pinchbench_tasks/meeting_analysis` | task `.md` lookup root |
| `VLLM_BASE_URL` | `http://127.0.0.1:8021/v1` | vLLM endpoint |
| `SERVED_MODEL` | `Qwen3-4B` | model id served by vLLM |

## 5. Bench / evaluate

After training, the wrapper hot-loads the new LoRA into vLLM and runs a 3-run
bench on the 5 test tasks. Result lands at:

```text
$BASE_DIR/bench_<round_tag>_v2_<timestamp>/result.json
```

(default `$BASE_DIR=/workspace/$EXPERIMENT`; override `BASE_DIR` to use any
writable directory.)

To run bench manually on a saved LoRA:

```bash
curl -X POST http://127.0.0.1:8021/v1/load_lora_adapter \
    -H "Content-Type: application/json" \
    -d '{"lora_name":"meeting-r1","lora_path":"<lora_path>"}'

# DEEPSEEK_API_KEY must be in env — the judge resolution chain
# (scripts/lib_grading.py:resolve_judge_backend_from_env) reads it directly.
# `--api-key dummy` below is for the vLLM endpoint, NOT the judge.
DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY \
python scripts/benchmark.py \
    --suite "task_meeting_advisory_stakeholders,task_meeting_council_votes,task_meeting_gov_speaker_summary,task_meeting_tech_action_items,task_meeting_sentiment_analysis" \
    --model "custom/meeting-r1" \
    --base-url "http://127.0.0.1:8021/v1" \
    --api-key "dummy" \
    --judge "openai/deepseek-chat" \
    --output-dir /workspace/bench_meeting_r1 \
    --runs 3
```

> **Common 401 trap.** Setting only `OPENAI_API_KEY=$DEEPSEEK_API_KEY` is
> NOT enough — the judge resolution chain in `lib_grading.py` does not read
> `OPENAI_API_KEY`. It checks `PINCHBENCH_GRADE_JUDGE_API_KEY` →
> `JUDGE_API_KEY` → `DEEPSEEK_API_KEY`. The simplest fix is to ensure
> `DEEPSEEK_API_KEY` is exported (as in `~/.pinchbench_env`).

## 6. Diagnose the run

After bench, run the diagnostics module to get per-task analysis (failure
modes, output budget allocation, automated check stability across runs):

```bash
python -m agent_loop.diagnostics analyze \
    --result-json $BASE_DIR/<bench_dir>/result.json \
    --transcripts-dirs results/0071_transcripts \
    --output $BASE_DIR/<bench_dir>/diagnosis.md
```

`results/<NNNN>_transcripts/` is automatically created by `scripts/benchmark.py`
during the bench step — `<NNNN>` is a zero-padded sequential job id (e.g.
`0071`), one folder per `benchmark.py` invocation, holding one
`<task_id>.jsonl` per evaluated task. List `ls results/` after a bench to find
the latest folder; pass it (or several, for multi-run merge) to
`--transcripts-dirs`.

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
