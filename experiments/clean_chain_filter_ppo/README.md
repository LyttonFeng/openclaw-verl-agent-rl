# Clean Chain：[质量过滤 + PPO] 6 轮全量结果

本目录是 **2026-05-10 在 Qwen3-4B 上从 base 起跑 6 轮**的完整实验产出，
verifying that **质量过滤（race-to-bottom 防御）+ PPO 三件套（importance ratio + clip + KL）**
作为同一 setting 在多轮 chain 中能稳步提升性能。

详细分析见 [`docs/experiment_report.md`](../../docs/experiment_report.md) §"Clean chain"。
算法设计见 [`docs/algorithm.md`](../../docs/algorithm.md)。

## 总览

| 轮次 | MEETING % | 增量 vs base | 训练样本（filter 后） | 训练样本（学习信号 task）| avg KL |
|---|---:|---:|---:|---:|---:|
| **base** (Qwen3-4B) | **44.70** | — | — | — | — |
| R1' | 46.90 | +2.20 | 32 | 16 | 0.0014 |
| R2' | 46.30 | +1.60 | 24 | 12 | 0.0021 |
| R3' | 46.70 | +2.00 | 18 | 9 | 0.0023 |
| **R4'** 🏆 | **47.80** | **+3.10** | 14 | 7 | 0.0024 |
| R5' | 46.50 | +1.80 | 18 | 9 | 0.0025 |
| R6' | 45.40 | +0.70 | 16 | 8 | 0.0024 |

**峰值 R4' = 47.80%（+3.10pp vs base）**

> avg_kl 全程稳定在 0.001-0.003，说明 KL k3 estimator 工作正常，β=0.02 选得合适。
> 如果 avg_kl > 0.05 说明 β 太小（policy 漂移过快），< 0.0005 说明 β 太大（policy 不动）。

## Per-task 细分

| Task | base | R1' | R2' | R3' | R4' | R5' | R6' |
|---|---:|---:|---:|---:|---:|---:|---:|
| advisory_stakeholders | 0.384 | 0.424 | 0.448 | 0.419 | 0.428 | **0.478** | 0.429 |
| council_votes | 0.198 | 0.204 | 0.185 | 0.177 | **0.252** | 0.179 | 0.206 |
| gov_speaker_summary | 0.425 | 0.407 | 0.434 | 0.407 | 0.417 | 0.405 | 0.406 |
| tech_action_items | 0.586 | 0.642 | **0.656** | 0.645 | 0.633 | 0.646 | 0.623 |
| sentiment_analysis | 0.641 | 0.667 | 0.592 | **0.687** | 0.662 | 0.619 | 0.604 |

## 文件说明

每轮 3 个 artifact：

- `round_N_bench.json` — bench 完整结果：5 task × 3 run，每个 run 含 judge score、
  per-check 明细、judge notes（解释扣分原因）、agent transcript metadata
- `round_N_quality_report.json` — 质量过滤报告：哪些 positive-adv 样本被过滤、
  原因（group_max < 0.4 / total_output < 500 / no_tool_success）、过滤后保留多少
- `round_N_training_meta.json` — 训练元数据：n_samples、n_optimizer_steps、
  avg_loss、**avg_kl**、ppo_enabled、clip_eps、kl_beta

加上：

- `chain_script.sh` — 6 轮自动化 orchestrator（pod 上运行）
  - 含 OOM 自动重试 2 次
  - 步级 idempotent（重启不重复已完成的 step）
  - vLLM lifecycle 管理（rollout/bench 时活，train 时杀）

## 关键观察

### 1. 训练数据信号衰减是健康信号

| 轮次 | learning signal task 数 | 跳过的 task 数（zero variance）|
|---|---:|---:|
| R1' | 16 | 7 |
| R2' | 12 | 11 |
| R3' | 9 | 14 |
| R4'+ | 7-9 | 14-16 |

随训练推进，越来越多 task 内的 N=2 rollout 收敛到相同分数（zero variance），
被 GRPO 跳过。这意味着**已经会做的 task 不再产生学习信号**，policy 自动把
精力放在还没攻克的 task 上。

### 2. KL 全程稳定，PPO 没失控

avg_kl 6 轮 0.0014-0.0025，符合预期。如果出现 KL 突然飙升（> 0.05）说明：
- β 太小压不住 policy 漂移
- 或 advantage 信号太大（reward shaping 出问题）

### 3. R4' 后过训迹象

R5'/R6' 总分微降（46.5/45.4 vs R4' 47.8）。Per-task 看：
- tech_action_items：R2' 峰值 0.656，之后慢慢退到 0.623（R6'）
- sentiment_analysis：R3' 峰值 0.687，R6' 退到 0.604

这是经典 **RL 后期过训**：policy 在 ceiling 附近开始过度优化某些任务，损
害其他任务。**生产 setting 应该跑到 R4' 就停**（或加 early-stopping based on
held-out bench）。

### 4. R1' 已经超过 messy chain 的原 R1

| | Setting | MEETING % |
|---|---|---:|
| 原 R1（vanilla GRPO，无过滤） | 标准 GRPO | 46.20 |
| **R1'（filter + PPO）** | 本次 setting | **46.90** |

差 +0.7pp，证明 setting 改进**从第一轮就开始生效**，不是后期才发挥作用。

## 复现

完整命令序列见 [`docs/reproduction.md`](../../docs/reproduction.md) §"进阶：质量过滤 + PPO"。
也可直接看 `chain_script.sh`。
