# openclaw-verl-agent-rl

在 PinchBench `meeting_analysis` 任务上对 Qwen3-4B 进行离线 GRPO 训练，
可选 **Roadmap PRM** 提供 per-turn 过程奖励。

> **关于 repo 名字的说明**：repo 叫 `openclaw-verl-agent-rl` 是历史命名遗留，
> 但**当前 GRPO 训练器不使用 veRL**。训练是 `rl/train/train_meeting_grpo_step.py`
> 里一个自包含的 PyTorch + transformers + peft 循环（~250 行）。
> veRL 源码只留给遗留 PPO 脚本（`rl/train/launch_main_ppo.py`、
> `run_reinforce_lora.sh` 等）使用，**这些脚本不是本文档的复现路径**。
> 详见 [`docs/algorithm.md`](docs/algorithm.md) §"训练器实现"。

- **异步、off-policy。** vLLM 负责 rollout；veRL-free 的 GRPO step 每轮写出一个
  新的 LoRA；LoRA 热加载回 vLLM 用于下一轮。
- **两层奖励。** Terminal reward（自动检查 + LLM judge）+
  可选 process reward（DSv4-flash judge 对照任务专属 roadmap，
  带有 terminal-completion gate，只对失败轨迹进行监督）。
- **不依赖 ECS。** OpenClaw 在与训练相同的 pod 上本地运行。

## 从这里开始

1. [`docs/algorithm.md`](docs/algorithm.md) — 设计（流程图 + 奖励公式 + Roadmap PRM）
2. [`docs/reproduction.md`](docs/reproduction.md) — 端到端复现流程（terminal-only 与 terminal+PRM）
3. [`docs/diagnostics.md`](docs/diagnostics.md) — 轨迹分析模块
4. [`docs/experiment_report.md`](docs/experiment_report.md) — 包含消融实验的完整结果历史

## 参考结果

在 5 个 held-out 测试任务上的 3-run 平均，judge = `deepseek-chat`：

| 配置 | 总分 | Δ vs baseline | 备注 |
|---|---|---|---|
| Baseline (rope=2, no LoRA) | 50.6% | — | apples-to-apples baseline |
| Terminal-only, R5 LoRA | 55.0% | +4.4pp | ~5 轮收敛 |
| **Terminal + Roadmap PRM, R1 LoRA** | **57.24%** | **+6.6pp** | 1 轮即收敛 |

## 仓库结构

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

## 所需版本

| 组件 | 版本 |
|---|---|
| Python | 3.12 |
| **veRL** | **可选** — 当前 GRPO 训练器（`train_meeting_grpo_step.py`）不 import veRL。只有想运行遗留 PPO 脚本（`launch_main_ppo.py` / `run_reinforce_lora.sh`）才需要，那些脚本不是用来跑下面 SOTA 结果的。跳过 veRL 安装，§"Quick start" 的 0c-4 步骤照常工作。 |
| vLLM | 0.10.2 (with `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True`) |
| Transformers | 4.57.1 |
| Torch | 2.8.0+cu128 |
| OpenClaw CLI | 2026.4.5 (3e72c03) — 本地安装，不通过 SSH/ECS |
| PinchBench | 1.2.1 (内嵌子集 — `pinchbench_tasks/meeting_analysis/` + `assets/meetings/`) |
| GPU | 2 × A100-80GB (GPU 0 = train, GPU 1 = vLLM) |

## Quick start

```bash
# 0a. Python deps
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 0b. veRL — OPTIONAL, can skip. Current GRPO trainer doesn't import veRL.
# Only install if you also want the legacy PPO scripts to work.
# git clone https://github.com/volcengine/verl.git ~/verl && pip install -e ~/verl

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

## 状态

可在 4 个 transcripts × 5 个 held-out 任务的 suite 上端到端复现单轮训练结果。
使用默认配方继续训练超过 R2 会在测试集上回退（reward hacking）；
详见 [`experiment_report.md`](docs/experiment_report.md) §15。
