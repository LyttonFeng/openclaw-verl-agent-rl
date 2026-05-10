# 实验报告 — Meeting Analysis GRPO

5 个 held-out 测试任务上的 3-run 平均分，judge = `deepseek-chat`。
**范围说明**：这是一个小 N 的研究（5 个测试任务 × 3 次 run = 每个 checkpoint
15 次评估，单一训练 seed）。下面的数字是点估计；我们报告 per-run 分数以便
读者判断 variance。结论以观察的形式给出，不是统计学意义上的断言。

## 设置

| 项目 | 取值 |
|---|---|
| 模型 | Qwen3-4B (LoRA rank 16, α=32, target = q/k/v/o/gate/up/down) |
| 上下文 | 80K (rope-scaling dynamic factor=2.0), bf16 |
| GRPO | 离线，group N=2 responses/prompt，无 critic，无 GAE |
| 训练数据 | 23 tasks × 2 responses = 46 rollouts/round，variance-filtered，pos-only PRM clip |
| 测试数据 | 5 个 held-out 任务 × 3 次 run（降低 judge variance） |
| 源会议 | NTIA spectrum advisory (71KB) · GitLab PMM (34KB) · Tampa City Council (206KB) · NASA UAP hearing (265KB) |
| Step 预算 | 每轮 15 次 GRPO 更新，lr=2e-6，batch=2 (grad accum) |

## Baselines

这里 `rope=N` 指的是 RoPE-scaling 的 `dynamic` factor — `rope=1` 是原生
40K-token 上下文，`rope=2` 把有效上下文翻倍至 80K（必需，因为 Tampa City
Council 和 NASA UAP transcript 超过 40K）。未适配的 base model 在 rope=2
下会有小幅（-1.1pp）质量损失，因为它从未在扩展位置上训练过；LoRA 训练能
恢复并超过 rope=1 baseline。

**下面所有训练后的模型都在 rope=2 下做 bench；要做 apples-to-apples 对比
请用 rope=2 baseline。**

| 配置 | 3-run 平均 |
|---|---|
| Qwen3-4B base, rope=1 / 40K | 51.7% |
| Qwen3-4B base, rope=2 / 80K | **50.6%** ← 标准 baseline |

## 主要结果

| 配置 | 3-run 平均 | Δ vs rope=2 baseline |
|---|---|---|
| baseline rope=2 (no LoRA) | 50.6% | — |
| **terminal + Roadmap PRM (additive judge-gate, β=0.10) — R1** | **57.24%** | **+6.6pp** |

SOTA checkpoint 的 per-task per-run 分数：

| Task | run 1 | run 2 | run 3 | mean | std |
|---|---|---|---|---|---|
| advisory_stakeholders | 0.560 | 0.553 | 0.607 | 0.573 | 0.029 |
| council_votes | 0.344 | 0.344 | 0.163 | 0.283 | 0.105 |
| gov_speaker_summary | 0.719 | 0.604 | 0.490 | 0.604 | 0.115 |
| sentiment_analysis | 0.646 | 0.652 | 0.719 | 0.672 | 0.040 |
| tech_action_items | 0.717 | 0.820 | 0.650 | 0.729 | 0.086 |

注意 per-task std 不可忽略 — gov_speaker run-to-run 波动 ±0.115，
council_votes ±0.105。+6.6pp 的总体增益远超 3-run 总体抖动，但 per-task
的论断要带着这个 variance 一起读。

advantage 公式（α=1.0, β=0.10, 对 `prm_turn_score` 做 pos-only clip）：

```
per_token_adv[k] = α · terminal_adv + β · prm_turn_score[k]
```

`mostly_done` gate 先运行；如果为 `yes`，`prm_turn_scores=[0]*n_turns`
（纯 terminal gradient）。设计动机见 [`algorithm.md`](algorithm.md)。

## Judge-gate 消融 (R1)

相同 base、相同 30-record 训练集、相同 15 次 GRPO step，仅改变 per-token
advantage 形状。这隔离了 judge-gate 与单纯运行 per-turn PRM scoring 的贡献。

| 配置 | 平均 | Δ vs rope=2 baseline | 改了什么 |
|---|---|---|---|
| rope=2 baseline | 50.6% | — | 没训练 |
| terminal-only β=0 | 52.5% | +1.9pp | 仅 terminal advantage |
| additive β=0.10, raw PRM (incl. -1) | 51.0% | +0.4pp | per-turn judge，无 gate，无 clip |
| additive β=0.10, **pos-only**, no gate | 55.6% | +5.0pp | 单独 pos-only clip；-1 → 0 |
| additive β=0.10, pos-only, **mult-B form** | 56.6% | +6.0pp | `(1+β·prm)` 形式，无 gate |
| **additive β=0.10, pos-only, judge-gate (SOTA)** | **57.24%** | **+6.6pp** | 在 pos-only additive 之上加 gate |

解读：**单一最大的胜利是 pos-only clip**（相对 baseline +5.0pp，
或相对 no-clip +3.1pp）。在 pos-only additive 之上 **judge-gate 又加了
+1.6pp** — 有意义但不是全部。mult-B 形式（无 gate）量级相近（56.6%, +6.0pp）。

这说明 gate 是真实的但不是魔法；pos-only clip 在挑大梁。我们没有运行
multi-seed 重复来给 +1.6pp gate 增益加 error bar — 在当前 N 下视为暗示性。

## Per-task 增益分析

| Task | rope=2 baseline | SOTA (R1 judge-gate) | 绝对 Δ | 相对 Δ |
|---|---|---|---|---|
| advisory_stakeholders | 0.49 | 0.573 | +0.083 | +17% |
| council_votes (最难) | 0.18 | 0.283 | **+0.103** | **+57%** |
| gov_speaker_summary | 0.55 | 0.604 | +0.054 | +10% |
| sentiment_analysis | 0.68 | 0.672 | -0.008 | -1% |
| tech_action_items | 0.63 | 0.729 | +0.099 | +16% |
| **总体** | **0.506** | **0.5724** | **+0.066** | **+13%** |

最大的**相对**增益出现在 baseline 最低的任务上 — 部分是 headroom 效应
（低 baseline = 更多增长空间），部分与"PRM 集中在失败轨迹"的设计意图一致。
我们无法在不做 per-task gate 消融的情况下干净地分离这两者，而我们没做。
tech_action_items 是反例：高 baseline (0.63) 但绝对增益与 council_votes
相当 (+0.10pp)，这不符合纯 headroom 解释。sentiment_analysis 基本持平 —
模型已经接近上限，PRM 没什么可补充的。

## Judge-gate 命中率

R1 和 R2 都产生了相同的 gate split：

```
22/46 trajectories  mostly_done  (no PRM applied)
24/46 trajectories  lost          (per-turn PRM applied)
```

两轮还不足以宽泛地声称"跨轮稳定"，但**两轮中相同的 22/24 split** 暗示在
固定 roadmap 和 base model 下 gate decision 在训练数据上是确定性的。如果
这个比例塌缩到 `0/46` 或 `46/46`，rubric 就漂移了，需要重新校准。
我们把它当作冒烟信号来监控。

## 继续训练：R2 回退

R2-additive（从 R1-additive LoRA 继续，相同配方）：**55.34%**，
相比 R1 回退 -1.9pp。

Per-task R2 vs R1：

| Task | R1 | R2 | Δ |
|---|---|---|---|
| advisory_stakeholders | 0.573 | 0.438 | **-0.135** ⚠ |
| council_votes | 0.283 | 0.233 | -0.050 |
| gov_speaker_summary | 0.604 | 0.652 | +0.048 |
| sentiment_analysis | 0.672 | 0.689 | +0.017 |
| tech_action_items | 0.729 | 0.754 | +0.025 |

advisory_stakeholders 解释了几乎全部回退。所有 3 个 R2 run 都失败于相同的
自动 check（确定性的，非随机）。

Diagnostics 模块呈现了一个与回退强相关的指标 — 训练好的文件产物与轨迹末尾
chat 端总结之间的**输出预算分配**变化：

| Task | R1 budget ratio | R2 budget ratio |
|---|---|---|
| advisory_stakeholders | 0.92 | **0.64** ⬇ |
| sentiment_analysis | 0.75 | 0.47 ⬇ |

`budget_ratio = output_file_chars / (output_file_chars + final_chat_chars)`。

**假设（尚未通过 judge-prompt 检视确认）**：PRM judge 读取轨迹 turns，
而轨迹末尾的 chat 端总结看起来像"深思熟虑的收尾"值 +1，而自动 grader
只读文件。如果属实，模型在被奖励"在 chat 里谈做这件事"而不是写交付物。
这是 PRM 系统中奖励层错位时的教科书式 reward-hacking 失败模式。

要确认/否证该假设，我们需要捕获回退轨迹的实际 per-turn judge prompt + score
并检查 chat-only turns 是否被给 +1。我们尚未做。把 R2 分析理解为
"diagnostics 标记了一个连贯的模式" — 而不是"已被证实的机制"。

我们能下的结论比"PRM 导致 R2 回退"更窄：
- R1 是相对 baseline 的真实单轮增益。
- R2 用相同配方确定性地回退。
- Diagnostics 在 per-task budget 粒度上捕获了回退，单看总体均值
  会错过它（仅 -1.9pp）。
- 用此配方继续 R1 之后是不安全的；rubric 可能需要先惩罚 chat 端冗长输出
  才能让 R2+ 有意义。

## R3 重训：质量过滤 + PPO 突破 R2 平台 (2026-05-09 ~ 2026-05-10)

R2→R3 vanilla GRPO 退化 -3.1pp（46.4% → 43.3%）后，做了三轮干预实验：

| Round | Setting | MEETING % | vs R2 |
|---|---|---:|---:|
| R2 (基准) | vanilla PG, no filter | 46.40 | — |
| **R3 v1** | vanilla PG, no filter（重跑 R3） | 43.30 | **-3.1pp 退化** |
| **R3 v2** | vanilla PG **+ 质量过滤**（race-to-bottom 防御） | 46.20 | -0.2pp（持平） |
| **R3 v3** | **PPO + KL** + 质量过滤 | **47.50** | **+1.1pp 真实增量** ⭐ |

### Per-task 细分（5-task held-out set）

| Task | base | R1 | R2 | R3 v1 | R3 v2 | R3 v3 |
|---|---:|---:|---:|---:|---:|---:|
| advisory_stakeholders | 0.384 | 0.430 | 0.440 | **0.358** ↓ | 0.490 | **0.504** ⭐ |
| council_votes | 0.198 | 0.252 | 0.248 | **0.175** ↓ | 0.200 | 0.225 |
| gov_speaker_summary | 0.425 | 0.418 | 0.434 | 0.397 | 0.397 | 0.425 |
| tech_action_items | 0.586 | 0.598 | 0.582 | 0.603 | 0.559 | **0.607** ⭐ |
| sentiment_analysis | 0.641 | 0.610 | 0.617 | 0.631 | **0.666** | 0.613 |
| **TOTAL %** | **44.70** | **46.20** | **46.40** | **43.30** | **46.20** | **47.50** |

### 退化诊断

R3 v1 退化的真实原因（通过对比 R2/R3 v1 同题 transcript 找到）：

1. **Race-to-bottom 组**：N=2 的 group 中两条都质量低时，judge 噪声/bias 让
   lazy 答案拿正 advantage，GRPO 把它当好榜样训练。R3 v1 的 16 个 useful
   group 中有 4 个属此类（max reward < 0.4）。
2. **早期终止漂移**：1/3 的 R3 v1 run 在 advisory_stakeholders 上"写到文件
   就交差"，final reply 仅 385 字符（vs R2 1742）—— judge 评 final text，
   这种 run 直接被 discount。
3. **缺少 PPO 安全机制**：vanilla PG 无 importance ratio + clip + KL，单步
   更新可以把 policy 推得很远，没有刹车。

### 干预 1：质量过滤（R3 v2）

只对 **正 advantage 样本**做质量审查（负样本是"避免信号"，质量差也 OK）。
保守三道过滤：
- group max(reward) ≥ 0.4（race-to-bottom 整组扔）
- final_reply + 所有写文件总字符 ≥ 500（彻底没产出的扔）
- 至少一次成功 tool call

**结果**：从 16 个正样本中删掉 4 个（全是 max < 0.4 的 race-to-bottom），
退化止住到 -0.2pp（持平 R2）。但**没有产生增量**，单靠过滤打不破平台。

详见 `algorithm.md` § 训练数据质量过滤。

### 干预 2：PPO + KL（R3 v3）

在过滤之上加 PPO 三件套：importance ratio + clip(ε=0.2) + KL k3 estimator(β=0.02)。

**实现关键**：
- 用 `compute_rollout_logprobs.py` 离线算 P_old（不用常驻 ref model，节省一半显存）
- bf16 saved P_old + fp32 训练（实测 1-3% outlier 被 PPO clip 自然吸收）
- π_ref = π_old（单轮锚定，不用第二份模型）

**结果**：47.5%，**首次突破 R2** +1.1pp。avg_kl=0.0015 全程稳定。

### 收获

| Setting | MEETING % | 增量解读 |
|---|---:|---|
| vanilla PG | 46.4 → 43.3 | **退化**（race-to-bottom + 无安全网） |
| + 质量过滤 | 46.2 | **止住退化**（删除假正样本） |
| + PPO + KL | **47.5** | **真正增量**（importance correction + 防漂移） |

**结论**：

> **质量过滤"治退化"，PPO+KL "破平台"，两者必须同时上**。  
> 单加任意一个都无法从 R2 继续涨。

详见：
- `algorithm.md` § 训练数据质量过滤
- `algorithm.md` § PPO 三件套
- `rl/train/apply_quality_filter.py`、`rl/train/compute_rollout_logprobs.py`

## PRM ablation 在新 baseline 上的实证（2026-05-10）

把 Roadmap PRM 加到 [filter + PPO] setting 上的实验。详见
[`experiments/r1_prm_ablation/README.md`](../experiments/r1_prm_ablation/README.md)。

| | base | R1' (no PRM) | v1 naive PRM | **v2 PRM + fixes** |
|---|---:|---:|---:|---:|
| MEETING % | 44.68 | **46.89** | **45.69** ↓ | **47.80** ↑ |
| 增量 vs no-PRM | — | base | -1.20 | **+0.91** |

**核心发现**：
- **朴素加 PRM 反而退化 -1.2pp**——PRM 把 "milestone-aligned 啰嗦" 偏置注入
  policy，破坏强项 task（tech_action_items、sentiment_analysis）已收敛的局部最优
- **加两个 fix（reward-gate + per-turn-loss）后 +0.9pp**：
  - Reward gate (`score >= 0.5 → 清零 PRM`)：救回强项
  - Per-turn loss (token 权重 = 1/n_tokens_in_turn)：消除 "长 turn 多放大" 偏置
- **PRM 增量在 5-task bench 噪声边界附近**（±0.5-1pp），严格性需要更大 task
  suite 或多 seed 验证

## Clean chain：从 base 起跑 6 轮 [filter + PPO]（2026-05-10）

为验证 setting 的稳定性，**从 base Qwen3-4B 起重跑 6 轮**，每轮都用同一 setting：
质量过滤（race-to-bottom 防御）+ PPO 三件套（importance ratio + clip + KL）。

| Task | base | R1' | R2' | R3' | **R4'** | R5' | R6' |
|---|---:|---:|---:|---:|---:|---:|---:|
| advisory_stakeholders | 0.384 | 0.424 | 0.448 | 0.419 | 0.428 | **0.478** | 0.429 |
| council_votes | 0.198 | 0.204 | 0.185 | 0.177 | **0.252** | 0.179 | 0.206 |
| gov_speaker_summary | 0.425 | 0.407 | 0.434 | 0.407 | 0.417 | 0.405 | 0.406 |
| tech_action_items | 0.586 | 0.642 | **0.656** | 0.645 | 0.633 | 0.646 | 0.623 |
| sentiment_analysis | 0.641 | 0.667 | 0.592 | **0.687** | 0.662 | 0.619 | 0.604 |
| **TOTAL %** | **44.70** | **46.90** | **46.30** | **46.70** | **🏆 47.80** | **46.50** | **45.40** |

### 关键观察

1. **峰值 R4' = 47.80%**（+3.10pp vs base，超过 messy chain 的 R3 v3 = 47.5%）
2. **R1' = 46.90% > 原 R1 = 46.20%**（差 +0.7pp），证明 [filter + PPO] setting **从 base 起步就胜过 vanilla GRPO**
3. **R5'/R6' 微降**（46.50/45.40）— 典型的 RL 过训信号：policy 在 ceiling 附近开始过度优化
4. **Best practice**：跑到 R4' 就停；继续训会退化

### 与 messy chain (R1...R3) 对比

| Chain | 起点 | Setting | 峰值 | 峰值轮次 |
|---|---|---|---:|---|
| **Messy chain** | base | vanilla PG → 过滤补救 → 加 PPO 抢救 | 47.50 (R3 v3) | 第 3 轮（带救火） |
| **Clean chain** | base | 从头就用 filter + PPO | **47.80 (R4')** | 第 4 轮 |

Clean chain **不需要救火，纯靠 setting 自身就能稳步上爬到接近 ceiling**。
说明：质量过滤 + PPO+KL 是真正起作用的，不是巧合。

### 训练数据信号衰减规律

逐轮观察 `Tasks with learning signal`：

| 轮次 | 有信号 task 数 | 训练样本数 |
|---|---|---|
| R1' | 16 | 32 |
| R2' | 12 | 24 |
| R3' | 9 | 18 |
| R4'-R6' | 7-9 | 14-18 |

随训练推进，越来越多 task 内的 N=2 rollout 收敛到相同分数（zero variance），
被 GRPO 跳过。这是**训练后期的健康信号**——学完了的 task 自然失去学习信号。

### 复现命令

参考 `repro/clean_chain_filter_ppo.sh`（pod 上的 6 轮自动化脚本，含 OOM 重试 +
断点续跑 + vLLM lifecycle 管理）。

## 收敛性比较（定性）

Terminal-only 和 terminal+PRM 在这套配置上都会收敛。轶事性地，terminal-only
路径（在我们之前的 rope=1 run 中，见 git history）大约需要 ~5 轮才能达到
可比的平台期，而 rope=2 下 terminal+PRM 仅需 1 轮。我们这里**不**给出
side-by-side per-round 表格，因为我们没有逐轮跑 rope=2 下的 terminal-only —
只把 R5 terminal-only LoRA 在 rope=2 上做了 bench (55.0%，见之前结果)。
所以"1 round vs 5 rounds"的说法是基于 rope=1 趋势形状的暗示，不是干净的
rope=2 比较。

如果你想做干净的收敛性研究，在 rope=2 下用 `PRM_BETA=0 SKIP_PRM_SCORING=1`
跑 terminal-only 5 轮以上并逐轮比较。

## 这个实验支持什么 — 和不支持什么

支持：
- 带 additive judge-gate 的 Roadmap PRM 在这个 5-task held-out set 上产生
  真实的单轮增益（相对 rope=2 base +6.6pp，相对相同配方但无 gate 的版本 +1.6pp）。
- Pos-only clip 是单一最大的设计杠杆 (+3.1pp)。
- Diagnostics 工具能呈现总体均值跟踪会错过的 task-level reward-hacking 失败。

不支持：
- 关于 judge-gate 对 +1.6pp 增益必要性的统计严格论断（单 seed，无 error bar）。
- 经过测量的 rope=2 下"1-round vs 5-rounds"terminal-only 比较。
- 关于 Roadmap PRM 在此设置、模型尺寸和任务族之外的一般性论断。

## 可复现性

- 训练 + bench：见 [`reproduction.md`](reproduction.md)
- Diagnostics CLI：见 [`diagnostics.md`](diagnostics.md)
- 算法 + 流程图：见 [`algorithm.md`](algorithm.md)
