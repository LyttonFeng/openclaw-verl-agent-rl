# 算法

本仓库针对 PinchBench `meeting_analysis` 任务族在 Qwen3-4B 上运行 **离线 GRPO**，
可选 **Roadmap PRM** 提供 per-turn 过程奖励。

它有意保持简单：

- **异步、off-policy。** Rollout（vLLM 服务进程）和训练（一步式 GRPO updater）
  是解耦的。一轮 = 收集 rollouts → 评分 → PRM 打分 → 训练 → 把 adapter 热加载回 vLLM。
- **无 critic、无 GAE、无 KL。** GRPO 使用 group-relative advantage；
  PRM 启用时仅对 per-token advantage 进行 re-weight。
- **两层奖励** （可同时启用）：
  1. **Terminal reward** — 对最终产物做自动检查 + LLM judge。
  2. **Process reward (PRM)** — DSv4-flash 针对 task 专属 roadmap 评判每个
     rollout turn，给出每个 turn +1 / 0 / -1。

## 训练器实现：极简、**非 veRL**

尽管仓库名字带 verl，GRPO 训练器（`rl/train/train_meeting_grpo_step.py`）
是一个**自包含的 ~250 行 PyTorch + transformers + peft 循环**。它只
import `torch`、`transformers`、`peft` — **任何地方都没有 import `verl`**。
原始 SOTA pod（`experiment_report.md` 数据来源）和新的复现 pod 跑的都是
完全相同的这个极简 trainer。

trainer 端到端做的事情：
1. 加载 graded JSONL，按 `task_id` 分组，对每条记录计算
   `(score - mean) / max(std, 1.0)` advantage（Dr.GRPO 风格 — `max(..., 1.0)`
   clamp 实际上禁用了 std normalization）。
2. 过滤 |advantage| < 0.01 的记录，跳过近零更新。
3. 对每条保留的记录：tokenize transcript、构建 assistant-only mask、构建
   per-token advantage tensor（additive：`α·adv + β·prm_turn`；
   multiplicative：`adv > 0` 时 `adv · (1 + β·prm_turn)`，否则 `adv`）。
4. 在 bf16 autocast 下做单次 forward 通过 Qwen3-4B body + lm_head，
   用 fused `cross_entropy` 得 token log-probs，policy-gradient loss
   `-(adv * logp).mean()`。
5. Backward、`clip_grad_norm_(1.0)`、每 `grad_accum` 个样本 `optimizer.step()`。
6. 保存 LoRA adapter，退出。

为什么**不**用 veRL：
- 一次性的离线更新；veRL 的 online actor/critic/rollout-worker 架构
  对单次 30-trajectory 的 LoRA 梯度步骤来说过度。
- 奖励依赖 `deepseek-chat` LLM judge（每条轨迹约 50s），无法内联到
  veRL 的 per-step reward callback（它期望亚秒级 reward）— 我们先收集并
  评分整个 batch，再训练。
- PRM 提供一个 per-turn token-level 的 advantage reshape
  （`(1 + β·prm_turn)` / `α·adv + β·prm_turn`），与 veRL 的标准 advantage
  pipeline 不太匹配；把它 hack 进 veRL 的 `compute_advantages` 比当前 250
  行代码更具侵入性。

遗留的 `rl/train/launch_main_ppo.py`、`run_reinforce_lora.sh`、
`run_verl.sh`、`run_verl_outcome.sh`、`verl_*_patch.py` 和
`patch_verl_core_algos_no_whiten.py` 都是**没有产出报告中 SOTA 的并行实验** —
仅供参考。

## 端到端流程

```mermaid
flowchart LR
    subgraph "Inference (vLLM, GPU 1)"
        VLLM[vLLM server<br/>Qwen3-4B + rope=2 80K<br/>+ optional LoRA]
    end

    subgraph "Rollout (CPU/GPU 1, parallel)"
        ROLL[generate_meeting_rollouts.py<br/>OpenClaw multi-turn agents<br/>N workers]
        TERM[meeting_reward.py<br/>automated + LLM-judge]
    end

    subgraph "PRM scoring (optional)"
        PRM[score_trajectories.py<br/>DSv4-flash judge + roadmap<br/>terminal-completion gate]
    end

    subgraph "Training (GPU 0)"
        SEL[select_grpo_samples.py<br/>variance filter]
        TRAIN[train_meeting_grpo_step.py<br/>GRPO step + LoRA save]
    end

    VLLM -->|sample N responses| ROLL
    ROLL --> TERM
    TERM -->|graded.jsonl| PRM
    PRM -->|graded_prm.jsonl| SEL
    TERM -.->|graded.jsonl<br/>(terminal-only path)| SEL
    SEL --> TRAIN
    TRAIN -->|LoRA adapter| VLLM
```

虚线路径完全跳过 PRM 阶段 — 那就是 terminal-only 训练。

## 奖励公式

对 response `i` 中的每个 token-span `k`，GRPO advantage 在每个 prompt 的
`N` 个 response 组内做 group-relative 计算：

```
group_mean_i = mean(score_j for j in group)
terminal_adv_i = (score_i - group_mean_i) / max(std_group, 1.0)
```

然后 per-token advantage 取决于 `prm_mode`：

```
# Terminal-only (β = 0)
per_token_adv[k] = α · terminal_adv_i

# Additive (default, β = 0.10)
per_token_adv[k] = α · terminal_adv_i + β · prm_turn_score[k]

# Multiplicative (β = 1.0; only amplifies positive advantages)
if terminal_adv_i > 0:
    per_token_adv[k] = terminal_adv_i · (1 + β · prm_turn_score[k])
else:
    per_token_adv[k] = terminal_adv_i   # failures keep pure terminal gradient
```

`prm_turn_score[k] ∈ {-1, 0, +1}` 是 roadmap judge 对产生 token `k` 的
assistant turn 给出的分数。

**Pos-only clip**（推荐，默认开启）：将 -1 视为 0 — 只鼓励进步，
从不惩罚。原因：per-turn judge 在难任务上有不可忽略的假阴性率
（judge 看到的是部分轨迹，不一定能判断一个 turn 是错路还是只是慢的探索步骤）。
让 -1 直接从 advantage 中扣减会惩罚 judge 误分类的探索步骤，伤害恰恰是
PRM 最该帮助的那些任务。经验上（`experiment_report.md` §"Judge-gate
ablation"），pos-only clip 是我们配置中单一最大的设计杠杆
（在其他条件不变的情况下，相对 no-clip 变体 +3.1pp）。

## Roadmap PRM 设计

PRM 有三个特征：

1. **Roadmap 来自成功轨迹** — 每个任务的"要达成的关键 milestone"都是从真实
   专家运行中提取的（`agent_loop/roadmap_prm/roadmaps/` 下经过校准的 yaml
   文件），或者可以由更强的 teacher model 提供。不需要每个任务的人工撰写
   rubric。
2. **Per-turn 评判。** DSv4-flash 一次拿 roadmap + 一个 rollout turn，
   决定它推进了哪个 milestone（如果有），返回 +1/0/-1。
3. **Terminal-completion gate**（"只帮迷路的轨迹"）。在 per-turn 评分前，
   judge 先决定该轨迹是否已经完成 terminal 目标
   （`mostly_done: yes/no`）。如果是 → `prm_turn_scores = [0]*n_turns`，
   训练回退到纯 terminal gradient。如果否 → 运行 per-turn judge。
   **成功的轨迹永远不会接收 PRM 增强**，这是 PRM 训练中最常见的
   reward-hacking 来源。

代码：`agent_loop/roadmap_prm/judge.py:judge_terminal_completion`、
`judge.py:judge_trajectory`。

## 为什么离线 + 异步

对于 agentic 任务（多轮、tool use、真实 workspace I/O），rollout 成本支配
wall-clock 时间。把 rollout 与 gradient step 耦合是浪费的。所以：

- vLLM 跨轮持续运行；每次训练步骤后热加载新的 LoRA
  （`/v1/load_lora_adapter` API）。无需重启。
- Rollout 和 PRM scoring 在并行 worker 中（默认 4 个）针对同一个 vLLM
  endpoint 运行。
- 训练读取已收集的 JSONL — 在整个 rollout batch 上做一次向量化的 GRPO step。
  没有 multi-step PPO loop。

我们配置下一轮的端到端时间大约：

| 阶段 | 时间 (Qwen3-4B, 30 records, single A100) |
|---|---|
| Rollout (4 workers, 23 tasks × 2 responses) | ~25 min |
| PRM scoring (DSv4-flash, 4 workers) | ~5 min |
| Variance filter + pos-only clip | <1 min |
| GRPO step (15 updates, batch=2) | ~10 min |
| Hot-load + 3-run bench | ~25 min |
| **总计** | ~65 min |
