# veRL 移植 POC（实验分支）

> ⚠️ **本目录仅在 `experiment/verl-port` 分支**，不会合并到 main。是探索性工程，
> 目标"看 veRL 能不能跑起来 + 学习 framework"，**不追求复现 47.80% SOTA**。

## 背景

主分支当前最佳 setting = `train_meeting_grpo_step.py`（自定义 250 行）+ filter + PPO，
单轮 47.80%。这套 setting 是为我们 agentic OpenClaw rollout + DeepSeek judge 量身写的。

`openclaw-verl-agent-rl` 的命名暗示原本想用 veRL，但实际从未集成。本分支检验：
**用业界标准框架 veRL 跑当前算法是什么体验？**

## POC 范围（一晚的工作量）

✅ **目标**：veRL 在本 pod (Qwen3-4B + 2x A100) 跑通官方 GRPO 例子（GSM8K）  
✅ **目标**：理解 veRL 的 data / rollout / reward / training 抽象  
✅ **目标**：识别"用 veRL 跑我们 meeting agentic 任务"的具体集成点和工作量  

❌ **非目标**：复现 47.80% SOTA  
❌ **非目标**：完整 OpenClaw + DeepSeek judge 集成  
❌ **非目标**：custom feature（quality filter / per-turn loss / reward gate）的 veRL 版

## 文件结构

```
docs/verl_port/
├── README.md            ← 本文件
├── 01_setup.md          ← 环境 + 数据准备
├── 02_first_run.md      ← 跑通 GSM8K GRPO 的过程 + 错误 + 修复
├── 03_findings.md       ← 关键发现（API、配置量、性能、集成点）
└── 04_meeting_gap.md    ← 把我们 meeting 任务接入需要做什么（gap 分析）
```

## 当前进度

跟踪在 `02_first_run.md`。
