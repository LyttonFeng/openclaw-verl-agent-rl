# openclaw-verl-agent-rl

在 PinchBench `meeting_analysis` 任务上对 Qwen3-4B 进行离线 GRPO 训练，
可选 **Roadmap PRM** 提供 per-turn 过程奖励。

> **关于 repo 名字的说明**：repo 叫 `openclaw-verl-agent-rl` 是历史命名遗留，
> 但**当前 GRPO 训练器不使用 veRL**。训练是 `rl/train/train_meeting_grpo_step.py`
> 里一个自包含的 PyTorch + transformers + peft 循环（~250 行）。
> `rl/legacy/` 下的 verl_*_patch.py 是早期 veRL-based PPO 路径的遗留 monkeypatch，
> **当前 SOTA 训练完全不依赖**。详见 [`docs/algorithm.md`](docs/algorithm.md) §"训练器实现"。

> ⚠️ **必须用 fp32 训练 + max_seq_len 64K**（2026-05-09 发现的两个独立问题）：
>
> **问题 1：bf16 + 长 context NaN**。在 **transformers 4.57 + peft 0.19** 下，
> bf16 训练会**数值溢出 → loss=NaN → optimizer 把 NaN 写入 LoRA 参数
> → 整个 LoRA 全部 NaN → bench 全 0%**。验证：bf16 R1 训练后 LoRA 全部 layers
> (lora_A/lora_B) 都是 `tensor([nan, ...])`，bench 第一 task 第一 run 即 0%。
> 修复：model load 用 `torch_dtype=torch.float32`，去掉
> `torch.amp.autocast("cuda", dtype=bfloat16)` wrapper。
>
> **问题 2：max_seq_len 必须匹配 rope 设计上限**。Qwen3-4B native context 是
> 32K，rope_scaling factor=2.0 把它扩到 **64K = 32K × 2**（这是 rope=2 的
> 设计上限）。之前用 `max_seq_len=81920` (80K) **超出 rope=2 设计范围**，
> attention bias 在 64K-80K 区间数值不稳，是 bf16 NaN 的根因之一。
> 修复：训练用 `--max-seq-length 65536` (64K) 跟 rope=2 对齐。
> 长 trajectory 会被 truncate 到 64K，但实测 23 个训练 task 的 trajectory
> 都在 64K 以内（除少数 419KB raw text 的极端例子）。
>
> **fp32 + 64K 双修复**：
> - 训练慢约 2x（每 sample ~25s），但 100% 稳定
> - 单测 R1 fp32（80K 即 fp32 阶段验证）= **46.16%**（baseline 44.68% → +1.48pp）
> - 同代码在 2026-05 老 SOTA pod (旧 transformers/peft) 用 bf16 80K 不出 NaN，
>   说明根因在「新版本 transformers/peft 数值实现 + 超 rope 设计上限的 context」组合。
>
> 加上环境变量防长 context OOM：`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`

- **异步、off-policy。** vLLM 负责 rollout；veRL-free 的 GRPO step 每轮写出一个
  新的 LoRA；LoRA 热加载回 vLLM 用于下一轮。
- **两层奖励。** Terminal reward（自动检查 + LLM judge）+
  可选 process reward（DSv4-flash judge 对照任务专属 roadmap，
  带有 terminal-completion gate，只对失败轨迹进行监督）。
- **不依赖 ECS。** OpenClaw 在与训练相同的 pod 上本地运行。

## 从这里开始

1. [`docs/algorithm.md`](docs/algorithm.md) — 设计（流程图 + 奖励公式 + Roadmap PRM + **race-to-bottom 防御 + PPO 三件套**）
2. [`docs/reproduction.md`](docs/reproduction.md) — 端到端复现流程（含**进阶: filter + PPO** 和 **PRM ablation**）
3. [`docs/diagnostics.md`](docs/diagnostics.md) — 轨迹分析 + **race-to-bottom 诊断方法论**
4. [`docs/experiment_report.md`](docs/experiment_report.md) — 完整结果历史（含 R3 退化诊断 + 6 轮 chain + PRM ablation）
5. [`experiments/clean_chain_filter_ppo/`](experiments/clean_chain_filter_ppo/) — 6 轮 clean chain artifacts（每轮 bench/quality/training_meta）
6. [`experiments/r1_prm_ablation/`](experiments/r1_prm_ablation/) — PRM ablation 三方对比 (no PRM / naive PRM / PRM+fixes)

## 参考结果（2026-05-10 复现 pod）

5 个 held-out 测试任务，3-run 平均，judge = `deepseek-chat`：

| 配置 | 总分 | Δ vs base | 关键设计 |
|---|---:|---:|---|
| **base Qwen3-4B**（无 LoRA） | **44.68%** | — | apples-to-apples baseline |
| 老 R1 (vanilla GRPO，无过滤) | 46.20% | +1.52pp | terminal-only, 退化在 R3 出现 |
| R2 (vanilla GRPO，平台) | 46.40% | +1.72pp | 同上，已饱和 |
| R3 v2 (vanilla GRPO + 质量过滤) | 46.20% | +1.52pp | 止住退化（R3 v1 跌到 43.30%） |
| **R3 v3 (filter + PPO + KL)** | **47.50%** | **+2.82pp** | 首次破 R2 |
| **R4' (clean chain 第 4 轮)** | **47.80%** | **+3.12pp** | 6 轮 chain 峰值 |
| **R1' + PRM (with reward gate + per-turn loss)** | **47.80%** | **+3.12pp** | 单轮持平 R4'，PRM ablation v2 |
| R1' + PRM (naive，无 fixes) | 45.69% | +1.01pp | 失败实验：PRM 拖累强项 task |

> **历史参考（不可比，仅作对照）**：原 SOTA pod（旧 transformers/peft + bf16）实验
> 报告 R5 terminal-only 55.0%、R1 + PRM 57.24% — 那是不同 baseline (50.6%) 下的
> 数字。本表是 2026-05 重建 pod（新 transformers 4.57 + fp32 必需）的真实复现，
> baseline 重新跑 = 44.68%。详见 [`docs/experiment_report.md`](docs/experiment_report.md)。

## 关键发现（2026-05-10）

**1. Vanilla GRPO 在 R3 会退化**（46.4% → 43.3%, -3.1pp）。  
诊断：N=2 group 内若两条 rollout 都质量低，judge 噪声让 lazy 答案拿正 advantage，
GRPO 把它当好榜样训练（race-to-bottom）。

**2. 解决 race-to-bottom：质量过滤**（`rl/train/apply_quality_filter.py`）。  
只对 positive-advantage 样本做质量审查，三道保守过滤（group max ≥ 0.4、total
output ≥ 500 字符、≥ 1 次成功 tool call）。R3 退化止住到 R2 持平（46.2%）。

**3. 突破 R2 平台：PPO 三件套**（importance ratio + clip + KL k3 estimator）。  
我们之前的 vanilla PG **严格说不算 RL**——缺 ratio 和 clip。加上之后单轮 +1.1pp，
6 轮 chain 峰值 47.80% (+3.10pp vs base)。  
实现亮点：用 saved P_old 替代常驻 ref model，**不需要加载第二份模型**。

**4. PRM 在新 baseline 下需要两个修复才有正向**：  
朴素加 PRM 反而 -1.2pp（强项 task 被啰嗦化偏置拖累）。加上：
- **Reward gate**（score ≥ 0.5 trajectory 跳过 PRM）
- **Per-turn loss weighting**（消除"长 turn 多放大 PRM"偏置）

之后 PRM = 47.80%（+0.9pp vs no-PRM）。

详见 [`experiments/r1_prm_ablation/README.md`](experiments/r1_prm_ablation/README.md)。

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
| **veRL** | **不需要** | 当前 SOTA 训练器（`train_meeting_grpo_step.py`）完全独立实现，不 import `verl`。`rl/legacy/` 下的 verl-based monkeypatch 也无需安装 veRL 才能查看（仅作历史参考）。**跳过 veRL 安装**完全不影响本文档的复现路径。 |
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
# 注：vLLM --max-model-len 81920 是 inference 缓存上限（rope=2 设计支持 80K）；
# 训练侧 --max-seq-length 用 65536 (64K)，因为 fp32 + 80K 训练会 OOM。两者是不同概念。
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

# 4a. ⚠️ 老路径（vanilla GRPO，无 filter 无 PPO）— **仅作 baseline 对照**
#     注意：R3 会 race-to-bottom 退化 -3.1pp。要复现 SOTA 47.80% 用 4b。
ROUND_NUM=1 bash rl/train/run_meeting_grpo_prm_round.sh

# 4b. ⭐ 推荐路径：[质量过滤 + PPO + KL] setting（峰值 47.80%）
#     单轮命令逐条见 docs/reproduction.md §"进阶: filter + PPO"
#     一键 6 轮自动化（含 OOM 重试 + 断点续跑 + vLLM lifecycle）：
#         bash experiments/clean_chain_filter_ppo/chain_script.sh

# 4c. 进阶 ++：再加 PRM (with reward-gate + per-turn-loss)
#     单轮命令见 docs/reproduction.md §"进阶 ++: PRM"
#     一键脚本：bash experiments/r1_prm_ablation/v2_script.sh
```

> ⚠️ **训练时千万不要改回 bf16**：transformers 4.57 + peft 0.19 + 64K context
> 在 bf16 下会数值溢出 → loss=NaN → 整个 LoRA 全 NaN → bench 全 0%。
> 所有当前脚本默认 fp32，详细原理见本文件开头的 fp32 警告框。

## 状态

- ✅ **基础：vanilla GRPO** 单轮收敛但 R3 会退化（reward hacking / race-to-bottom）
- ✅ **解决退化：质量过滤** 把 R3 v1 的 -3.1pp 退化救回到持平 R2
- ✅ **突破平台：PPO 三件套** (importance ratio + clip + KL) 单轮 +1.1pp
- ✅ **6 轮 clean chain 验证**：base 44.68% → R4' 47.80% (+3.12pp)
- ✅ **PRM ablation**：朴素加 PRM 退化 -1.2pp；加 reward gate + per-turn loss 后 +0.9pp

详见 [`docs/experiment_report.md`](docs/experiment_report.md)。
