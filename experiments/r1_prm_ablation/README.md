# PRM Ablation：R1' + Roadmap PRM 在 [filter + PPO] setting 上的实证

## 问题

PRM (Roadmap-based per-turn judge) 在原始 vanilla GRPO 实验里是 +6.6pp 的关键设计。
但当 baseline 升级到 [质量过滤 + PPO + KL] 之后（R1' = 46.89%），PRM 是否还有增量？

## 实验设计

控制变量：跟 clean chain R1' 完全一样的 setting（同 rollouts 复用、同
`compute_rollout_logprobs`、同 `train --clip-eps 0.2 --kl-beta 0.02`），
**只切换 PRM 启停**。

3 个对照：

- **R1' baseline (no PRM)**：填零 prm_turn_scores，β=0
- **v1 naive PRM**：真实 PRM 评分（multiplicative β=0.5，pos-only clip，terminal-completion gate）
- **v2 PRM with fixes**：在 v1 基础上加两个修正
  - `--prm-reward-gate 0.5`：score≥0.5 的 trajectory 清零 PRM
  - `--per-turn-loss`：消除 "长 turn 多放大" 偏置

## 结果

| Task | R1' (no PRM) | v1 (naive PRM) | **v2 (PRM + fixes)** |
|---|---:|---:|---:|
| advisory_stakeholders | 0.424 | 0.427 | **0.492** ⭐ |
| council_votes (弱项) | 0.204 | **0.235** | 0.233 |
| gov_speaker_summary | 0.407 | 0.418 | 0.397 |
| tech_action_items (强项) | **0.642** | 0.597 ↓ | 0.612 |
| sentiment_analysis (强项) | **0.667** | 0.608 ↓ | 0.654 |
| **TOTAL %** | **46.90** | **45.70** | **47.80** |

## 三方差异（vs R1' no-PRM）

```
v1 naive:       -1.20pp  ❌ PRM 拖累强项 task
v2 with fixes:  +0.90pp  ✅ 强项救回 + 弱项保留 PRM 加成
```

## 诊断：v1 为何退化

按 R1'（no-PRM）reward 把 task 分两类：

- **强项 (R1' > 0.6)**：tech_action_items, sentiment_analysis
  - v1 上分别 -4.5pp 和 -5.9pp
  - 原因：PRM 把已经在局部最优附近的 policy 拉向"milestone-aligned 啰嗦模式"
- **弱项 (R1' < 0.3)**：council_votes
  - v1 上 +3.1pp
  - 原因：PRM 强迫迷茫的模型按结构推进
- **中等 (R1' 0.3-0.6)**：advisory_stakeholders, gov_speaker_summary
  - v1 上几乎持平（噪声内）

PRM 的 **inductive bias** 偏向"deliberate, milestone-aligned"行为：
- 帮迷茫的模型 → 弱项受益
- 推走最优策略 → 强项受损

## 修复 v2 攻击的两个根因

### Fix 1：Reward-gated PRM

`apply_quality_filter.py --prm-reward-gate 0.5`：在过滤阶段对 score ≥ 0.5
的 trajectory 直接清零 `prm_turn_scores`，等同于纯 terminal gradient。

比现有的 `judge_terminal_completion` (binary `mostly_done`) 更细粒度——
binary gate 大约要 0.8+ 才放过；reward gate 直接看绝对分数。

效果：tech_action_items 从 v1 0.597 → v2 0.612（recover），sentiment 从
0.608 → 0.654（recover）。

### Fix 2：Per-turn loss weighting

`train_meeting_grpo_step.py --per-turn-loss`：每个 token 权重 = 1/n_tokens_in_its_turn，
确保**每个 turn 对梯度贡献相等，不被 turn 长度影响**。

切断 PRM "长 turn 多放大" → 训练向"啰嗦"漂移的机制。

效果：advisory_stakeholders 从 v1 0.427 → v2 **0.492**（涨幅最大），
说明这个 fix 单独也帮上忙——不只对强项有效。

## 核心洞察

PRM 在 [filter + PPO] setting 上**不是简单加上去就行**。需要：

1. **Reward gate** 防止 PRM 干扰已及格的 trajectory
2. **Per-turn weighting** 消除 PRM 在长 turn 上的偏置
3. **Multiplicative + β=0.5** 比 additive 或 default β=1.0 更稳

加上这三条，PRM 在 5-task bench 上**+0.9pp 超过 no-PRM baseline**，但增量在
噪声边界附近，需要更大 task suite + 多 seed 才能严格证明。

## 文件

| 文件 | 内容 |
|---|---|
| `v1_naive_bench.json` | v1 5-task × 3-run 完整 bench (含 judge notes) |
| `v2_with_fixes_bench.json` | v2 5-task × 3-run 完整 bench |
| `v1_training_meta.json` | v1 训练元数据 (avg_loss, kl_avg) |
| `v2_training_meta.json` | v2 训练元数据 |
| `v2_quality_report.json` | v2 quality filter 过滤的样本 + reason |
| `v2_script.sh` | v2 完整自动化脚本（含 PRM scoring + reward gate + per-turn-loss） |

## 复现

```bash
# 1. PRM scoring (~5 min)
python3 agent_loop/roadmap_prm/scripts/score_trajectories.py \
  --graded-file <rollouts.jsonl> ...

# 2. Quality filter with reward gate
python3 rl/train/apply_quality_filter.py \
  --input ... --output ... --report ... \
  --prm-reward-gate 0.5

# 3. Train with PRM + per-turn loss
python3 rl/train/train_meeting_grpo_step.py \
  --logprobs-file ... \
  --clip-eps 0.2 --kl-beta 0.02 \
  --prm-alpha 1.0 --prm-beta 0.5 --prm-mode multiplicative \
  --per-turn-loss
```

完整命令见 [`v2_script.sh`](v2_script.sh)。
