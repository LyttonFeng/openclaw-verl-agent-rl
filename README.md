# openclaw-verl-agent-rl

Offline GRPO training of Qwen3-4B on PinchBench `meeting_analysis` tasks,
with optional **Roadmap PRM** for per-turn process reward.

- **Async, off-policy.** vLLM serves rollouts; veRL-free GRPO step writes a
  new LoRA per round; LoRA hot-loaded back into vLLM for next round.
- **Two reward layers.** Terminal reward (automated check + LLM judge) +
  optional process reward (DSv4-flash judge against task-specific roadmaps,
  with terminal-completion gate that only supervises failed trajectories).
- **No ECS.** OpenClaw runs locally on the same pod as training.

## Start here

1. [`docs/algorithm.md`](docs/algorithm.md) — design (flow diagram + reward formula + Roadmap PRM)
2. [`docs/reproduction.md`](docs/reproduction.md) — end-to-end recipe (terminal-only and terminal+PRM)
3. [`docs/diagnostics.md`](docs/diagnostics.md) — trajectory analysis module
4. [`docs/experiment_report.md`](docs/experiment_report.md) — full result history with ablations

## Reference results

3-run mean on 5 held-out test tasks, judge = `deepseek-chat`:

| Config | Overall | Δ vs baseline | Notes |
|---|---|---|---|
| Baseline (rope=2, no LoRA) | 50.6% | — | apples-to-apples baseline |
| Terminal-only, R5 LoRA | 55.0% | +4.4pp | ~5 rounds to converge |
| **Terminal + Roadmap PRM, R1 LoRA** | **57.24%** | **+6.6pp** | converges in 1 round |

## Repo layout

```
agent_loop/                              OpenClaw multi-turn agent + analysis
├── openclaw_agent_loop.py / model_proxy.py / trajectory.py
├── diagnostics/                         layered trajectory analyzer (plugin-based)
└── roadmap_prm/                         Roadmap PRM judge stack
    ├── judge.py                         per-turn + terminal-completion gate
    ├── schema.py / trajectory.py / calibrate.py
    ├── roadmaps/                        46 calibrated yaml roadmaps
    └── scripts/score_trajectories.py    attaches PRM scores to a graded JSONL

rewards/meeting_reward.py                terminal reward (automated + LLM judge)

rl/
├── train/
│   ├── train_meeting_grpo_step.py       single GRPO step (additive / multiplicative PRM)
│   ├── generate_meeting_rollouts.py     parallel rollout collection + grading
│   ├── select_grpo_samples.py           variance filter + per-task selection
│   ├── build_meeting_analysis_prompts.py
│   ├── meeting_analysis_split.json      23 train / 5 test
│   └── run_meeting_grpo_prm_round.sh    end-to-end one-round wrapper
└── *_patch.py                           veRL / transformers / vLLM patches

scripts/benchmark.py                     PinchBench grader entrypoint

assets/meetings/                         4 real meeting transcripts
pinchbench_tasks/meeting_analysis/       28 task definitions
```

## Required versions

| Component | Version |
|---|---|
| Python | 3.12 |
| **veRL** | **0.8.0.dev0** — install **editable from source**, not pip release (≤0.7.x lacks the agent_loop API). See [`docs/reproduction.md`](docs/reproduction.md) §1. |
| vLLM | 0.10.2 (with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`) |
| Transformers | 4.57.1 |
| Torch | 2.8.0+cu128 |
| OpenClaw CLI | 2026.4.5 (3e72c03) — installed locally, not via SSH/ECS |
| PinchBench | 1.2.1 (subset embedded — `pinchbench_tasks/meeting_analysis/` + `assets/meetings/`) |
| GPU | 2 × A100-80GB (GPU 0 = train, GPU 1 = vLLM) |

## Quick start

```bash
# 0a. Python deps
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 0b. veRL — editable install from source (NOT a pip release)
git clone https://github.com/volcengine/verl.git ~/verl
pip install -e ~/verl

# 0c. OpenClaw CLI — public npm package. Install to LOCAL disk, NOT NFS:
npm install -g openclaw@2026.4.5      # ~30s on local disk; do NOT install under /workspace/
# If node lives outside /usr/local/bin (e.g. nvm), symlink it so the shebang resolves:
[ -x /usr/local/bin/node ] || ln -sf "$(which node)" /usr/local/bin/node
openclaw --version    # → 2026.4.5 (3e72c03)

# 1. DeepSeek API key (judge for both terminal grading and PRM scoring)
echo 'export DEEPSEEK_API_KEY=sk-xxxxxxxxxxxx' > ~/.pinchbench_env
chmod 600 ~/.pinchbench_env

# 2. start vLLM on GPU 1 (background) — see docs/reproduction.md §3 for full args
CUDA_VISIBLE_DEVICES=1 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B --served-model-name Qwen3-4B \
    --port 8021 --max-model-len 81920 \
    --rope-scaling '{"type":"dynamic","factor":2.0}' \
    --enable-lora --max-lora-rank 16 \
    --enable-auto-tool-choice --tool-call-parser hermes --reasoning-parser qwen3 \
    --gpu-memory-utilization 0.85 --dtype bfloat16 --trust-remote-code &
# wait until vLLM responds on /v1/models before proceeding (~60-90s on cold load)

# 3. build training prompts (one-time per checkout, not committed)
python rl/train/build_meeting_analysis_prompts.py \
    --tasks-dir pinchbench_tasks/meeting_analysis \
    --split-file rl/train/meeting_analysis_split.json \
    --output-dir data/meeting_prompts

# 4a. one round, terminal + PRM (recommended).
# BASE_DIR auto-resolves to /workspace/$EXPERIMENT on pods, else $HOME/grpo_runs/$EXPERIMENT.
ROUND_NUM=1 bash rl/train/run_meeting_grpo_prm_round.sh

# 4b. or terminal-only ablation (PRM weight zero; PRM judge still scores —
#     set SKIP_PRM_SCORING=1 to skip the DeepSeek PRM calls entirely)
PRM_BETA=0 SKIP_PRM_SCORING=1 \
ROUND_NUM=1 EXPERIMENT=meeting_grpo_terminal_v1 \
bash rl/train/run_meeting_grpo_prm_round.sh
```

## Status

Reproduces a single round end-to-end on the 4 transcripts × 5 held-out tasks
suite. Continued continuing past R2 with the default recipe regresses on the
test set (reward hacking); see [`experiment_report.md`](docs/experiment_report.md) §15.
