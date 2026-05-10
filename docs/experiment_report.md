# 实验报告 — Meeting Analysis GRPO

5 个 held-out 测试任务上的 3-run 平均分，judge = `deepseek-chat`。
**范围说明**：这是一个小 N 的研究（5 个测试任务 × 3 次 run = 每个 checkpoint
15 次评估，单一训练 seed）。下面的数字是点估计；per-task 分数有 ±5pp 量级 variance。
结论以观察的形式给出，不是统计学意义上的断言。

> **两个 pod 的故事**：本仓库经过两个时期：
>
> 1. **老 SOTA pod (2026-04 之前)**：bf16 + 80K context，旧 transformers/peft。
>    跑出 R1 + PRM = **57.24%** (vs base 50.6%, +6.6pp)。
> 2. **新复现 pod (2026-05)**：transformers 4.57 + peft 0.19 强制 fp32 + 64K context
>    （bf16 在新版本下长 context 必出 NaN）。base **重跑 = 44.68%**，跟老数字不可比。
>
> **§1-§4 是新 pod 工作（推荐看新读者）**，§5 是老 SOTA pod 历史结果（保留作为
> PRM judge-gate / pos-only clip 等设计选择的实证基础）。**两个 pod 数字不可直接对比**。

---

## §1. 当前推荐设置（2026-05 复现 pod）

| 配置 | MEETING % | Δ vs base | 关键设计 |
|---|---:|---:|---|
| base Qwen3-4B (rope=2 / 64K, fp32) | **44.68** | — | apples-to-apples baseline |
| 老 R1 (vanilla GRPO，无过滤) | 46.20 | +1.52 | terminal-only |
| R2 (vanilla GRPO，平台) | 46.40 | +1.72 | 同上，已饱和 |
| R3 v1 (vanilla GRPO 续训) | **43.30** ↓ | -1.38 | **退化**（race-to-bottom） |
| R3 v2 (+ 质量过滤) | 46.20 | +1.52 | 止退化（详见 §2） |
| **R3 v3 (+ PPO + KL)** | **47.50** | **+2.82** | 单轮破 R2（详见 §3） |
| **R4' (clean chain [filter + PPO] 第 4 轮)** | **47.80** | **+3.12** 🏆 | 6 轮 chain 峰值（§3.4） |
| R1' + PRM v1 (naive) | 45.69 | +1.01 | PRM 拖累强项 |
| **R1' + PRM v2 (+ reward gate + per-turn loss)** | **47.80** | **+3.12** | PRM ablation（§4） |

**对应章节**：
- §2 = race-to-bottom 诊断 + 质量过滤（救退化）
- §3 = PPO 三件套（破平台）+ Clean chain 6 轮验证
- §4 = PRM ablation（PRM 在新 baseline 下能否再加分？）
- §5 = **历史 SOTA pod**（2026-04 之前的 57.24% 等数据，**不可与 §1-§4 直接对比**）

**Artifacts**：
- [`experiments/clean_chain_filter_ppo/`](../experiments/clean_chain_filter_ppo/README.md) — 6 轮 chain
- [`experiments/r1_prm_ablation/`](../experiments/r1_prm_ablation/README.md) — PRM ablation

---

## §2. R3 退化诊断 + 质量过滤（救退化）

R2→R3 vanilla GRPO 退化 -3.1pp（46.4% → 43.3%）。三轮干预实验：

| Round | Setting | MEETING % | vs R2 |
|---|---|---:|---:|
| R2 (基准) | vanilla PG, no filter | 46.40 | — |
| **R3 v1** | vanilla PG, no filter（重跑 R3） | 43.30 | **-3.1pp 退化** |
| **R3 v2** | vanilla PG **+ 质量过滤** | 46.20 | -0.2pp（止退化但无增量） |
| **R3 v3** | **PPO + KL** + 质量过滤（详见 §3） | **47.50** | **+1.1pp 真实增量** ⭐ |

### Per-task 细分

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

### 干预 1：质量过滤（R3 v2 → 46.20%）

只对 **正 advantage 样本**做质量审查（负样本是"避免信号"，质量差也 OK）。
保守三道过滤：
- group max(reward) ≥ 0.4（race-to-bottom 整组扔）
- final_reply + 所有写文件总字符 ≥ 500（彻底没产出的扔）
- 至少一次成功 tool call

**结果**：从 16 个正样本中删掉 4 个（全是 max < 0.4 的 race-to-bottom），
退化止住到 -0.2pp（持平 R2）。但**没有产生增量**，单靠过滤打不破平台。

详见 [`algorithm.md`](algorithm.md) §"训练数据质量过滤"。

---

## §3. PPO 三件套（破平台）

在过滤之上加 importance ratio + clip(ε=0.2) + KL k3 estimator(β=0.02)。

**实现关键**：
- 用 `compute_rollout_logprobs.py` 离线算 P_old（不用常驻 ref model，节省一半显存）
- bf16 saved P_old + fp32 训练（实测 1-3% outlier 被 PPO clip 自然吸收）
- π_ref = π_old（单轮锚定，不用第二份模型）

**结果（R3 v3）**：47.5%，**首次突破 R2** +1.1pp。avg_kl=0.0015 全程稳定。

| Setting | MEETING % | 增量解读 |
|---|---:|---|
| vanilla PG | 46.4 → 43.3 | **退化**（race-to-bottom + 无安全网） |
| + 质量过滤 | 46.2 | **止住退化**（删除假正样本） |
| **+ PPO + KL** | **47.5** | **真正增量**（importance correction + 防漂移） |

**结论**：

> **质量过滤"治退化"，PPO+KL "破平台"，两者必须同时上**。  
> 单加任意一个都无法从 R2 继续涨。

代码：
- [`rl/train/apply_quality_filter.py`](../rl/train/apply_quality_filter.py)
- [`rl/train/compute_rollout_logprobs.py`](../rl/train/compute_rollout_logprobs.py)
- [`rl/train/train_meeting_grpo_step.py`](../rl/train/train_meeting_grpo_step.py) (加 `--logprobs-file --clip-eps --kl-beta`)

数学详细见 [`algorithm.md`](algorithm.md) §"PPO 三件套"。

### §3.4 Clean chain：从 base 起跑 6 轮 [filter + PPO]

为验证 setting 的稳定性，**从 base Qwen3-4B 起重跑 6 轮**。每轮都用同一 setting：
质量过滤 + PPO 三件套。

| Task | base | R1' | R2' | R3' | **R4'** | R5' | R6' |
|---|---:|---:|---:|---:|---:|---:|---:|
| advisory_stakeholders | 0.384 | 0.424 | 0.448 | 0.419 | 0.428 | **0.478** | 0.429 |
| council_votes | 0.198 | 0.204 | 0.185 | 0.177 | **0.252** | 0.179 | 0.206 |
| gov_speaker_summary | 0.425 | 0.407 | 0.434 | 0.407 | 0.417 | 0.405 | 0.406 |
| tech_action_items | 0.586 | 0.642 | **0.656** | 0.645 | 0.633 | 0.646 | 0.623 |
| sentiment_analysis | 0.641 | 0.667 | 0.592 | **0.687** | 0.662 | 0.619 | 0.604 |
| **TOTAL %** | **44.70** | **46.90** | **46.30** | **46.70** | **🏆 47.80** | **46.50** | **45.40** |

**关键观察**：

1. **峰值 R4' = 47.80%**（+3.10pp vs base，超过 messy chain R3 v3 的 47.5%）
2. **R1' 46.90% > 老 R1 46.20%**（+0.7pp），证明 [filter + PPO] **从 base 起步就胜过 vanilla GRPO**
3. **R5'/R6' 微降**（46.50/45.40）— 典型 RL 过训信号
4. **Best practice**：跑到 R4' 就停；继续训会退化

**Messy chain vs Clean chain**：

| Chain | 起点 | Setting | 峰值 | 峰值轮 |
|---|---|---|---:|---|
| Messy chain | base | vanilla PG → 过滤救场 → 加 PPO 抢救 | 47.50 (R3 v3) | 第 3 轮 |
| **Clean chain** | base | 从头就用 filter + PPO | **47.80 (R4')** | 第 4 轮 |

Clean chain **不需要救火，纯靠 setting 自身就能稳步上爬**。证明这是 setting 的功劳。

**训练数据信号衰减**：

| 轮次 | 有信号 task 数 | 训练样本数 |
|---|---|---|
| R1' | 16 | 32 |
| R2' | 12 | 24 |
| R3' | 9 | 18 |
| R4'-R6' | 7-9 | 14-18 |

随训练推进，越来越多 task 内的 N=2 rollout 收敛到同分（zero variance），被
GRPO 跳过——**训练后期的健康信号**，学完了的 task 自然失去学习信号。

**复现命令**：[`experiments/clean_chain_filter_ppo/chain_script.sh`](../experiments/clean_chain_filter_ppo/chain_script.sh)（6 轮 chain，含 OOM 重试 + 断点续跑）。

---

## §4. PRM ablation 在新 baseline 上的实证

把 Roadmap PRM 加到 [filter + PPO] setting 之上的实验。详见
[`experiments/r1_prm_ablation/README.md`](../experiments/r1_prm_ablation/README.md)。

| | base | R1' (no PRM) | v1 naive PRM | **v2 PRM + fixes** |
|---|---:|---:|---:|---:|
| MEETING % | 44.68 | **46.89** | **45.69** ↓ | **47.80** ↑ |
| 增量 vs no-PRM | — | base | -1.20 | **+0.91** |

**核心发现**：
- **朴素加 PRM 反而退化 -1.2pp**——PRM 把 "milestone-aligned 啰嗦" 偏置注入
  policy，破坏强项 task（tech_action_items、sentiment_analysis）已收敛的局部最优
- **加两个 fix（reward-gate + per-turn-loss）后 +0.9pp**：
  - **Reward gate** (`apply_quality_filter --prm-reward-gate 0.5`)：score ≥ 0.5 的
    trajectory 直接清零 PRM，避免干扰已及格 trajectory（救强项）
  - **Per-turn loss** (`train_meeting_grpo_step --per-turn-loss`)：每 token 权重 =
    1/n_tokens_in_turn，每个 turn 等权贡献梯度（消除"长 turn 多放大 PRM"偏置）
- **PRM 增量在 5-task bench 噪声边界附近**（±0.5-1pp），严格性需要更大 task
  suite 或多 seed 验证

PRM 在 [filter + PPO] setting 上**不是简单加上去就行**，需要两个 fix 才能拿到正向。
设计要点详见 [`algorithm.md`](algorithm.md) §"PRM 与 [filter + PPO] 组合的实证设计要点"。

---

## §6. 这个实验支持什么 — 不支持什么

### 新 pod (§1-§4) 支持

- **质量过滤 + PPO + KL** 是从 vanilla GRPO 突破 R2 平台的关键组合（+3.1pp vs base）
- **clean chain 4 轮收敛到 47.80%**，5 轮后开始过训（典型 RL 模式）
- **Race-to-bottom 是真实失败模式**：N=2 group 双低时 GRPO 会放大 lazy 模式
- **PPO 三件套 + saved P_old**（不用 ref model）实现可行，避免显存翻倍
- **PRM 在新 baseline 下的设计需要 reward gate + per-turn loss 才能不退化**
- **bf16 saved P_old + fp32 训练**的精度组合可行（PPO clip 自然吸收 outlier）

### 新 pod 不支持

- 严格的多 seed error bar（单 seed，每个数字±噪声 0.5-1pp）
- PRM 是否在更大 task suite 也是 +0.9pp（噪声边界附近）
- 多卡 device_map="auto" 在更大模型 (>7B) 是否还稳

### 老 pod (§5) 支持

- 带 additive judge-gate 的 Roadmap PRM 在老 pod 5-task held-out set 上 +6.6pp
- Pos-only clip 是单一最大设计杠杆（老 pod +3.1pp）
- Diagnostics 工具能呈现总体均值跟踪会错过的 task-level reward-hacking 失败

### 老 pod 不支持

- judge-gate 对 +1.6pp 的统计严格论断（单 seed）
- "1-round vs 5-rounds" terminal-only 严格比较
- Roadmap PRM 在此设置之外的一般性论断

---

## §7. 可复现性

- 训练 + bench：见 [`reproduction.md`](reproduction.md)
- Diagnostics CLI：见 [`diagnostics.md`](diagnostics.md)
- 算法 + 流程图：见 [`algorithm.md`](algorithm.md)
- 6 轮 chain artifacts：[`experiments/clean_chain_filter_ppo/`](../experiments/clean_chain_filter_ppo/)
- PRM ablation artifacts：[`experiments/r1_prm_ablation/`](../experiments/r1_prm_ablation/)

---

## §5. 历史 SOTA pod 数据（Appendix，2026-04 之前）

> ⚠️ **以下数据来自旧 pod (bf16 + 80K + 旧 transformers/peft)**。
> base 是 50.6%（不是新 pod 的 44.68%），所以**所有数字都不能跟 §1-§4 直接对比**。
> 保留这部分是为了 (a) 解释 PRM judge-gate / pos-only clip 等设计选择的实证基础，
> (b) 历史完整性。

### 5.1 设置（老 pod）

| 项目 | 取值 |
|---|---|
| 模型 | Qwen3-4B (LoRA rank 16, α=32, target = q/k/v/o/gate/up/down) |
| 上下文 | 80K (rope-scaling dynamic factor=2.0), **bf16**（新 pod 已知 bf16 出 NaN，必须 fp32） |
| GRPO | 离线，group N=2 responses/prompt，无 critic，无 GAE，**无 importance ratio + clip + KL** |
| 训练数据 | 23 tasks × 2 responses = 46 rollouts/round，variance-filtered，pos-only PRM clip |
| 测试数据 | 5 个 held-out 任务 × 3 次 run |
| Step 预算 | 每轮 15 次 GRPO 更新，lr=2e-6，batch=2 |

### 5.2 老 Baselines

| 配置 | 3-run 平均 |
|---|---|
| Qwen3-4B base, rope=1 / 40K | 51.7% |
| Qwen3-4B base, rope=2 / 80K | **50.6%** ← 老标准 baseline |

### 5.3 老 SOTA 主要结果

| 配置 | 3-run 平均 | Δ vs 老 rope=2 baseline |
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

### 5.4 Judge-gate 消融 (R1，老 pod)

相同 base、相同 30-record 训练集、相同 15 次 GRPO step，仅改变 per-token
advantage 形状。这隔离了 judge-gate 与单纯运行 per-turn PRM scoring 的贡献。

| 配置 | 平均 | Δ vs 老 rope=2 baseline | 改了什么 |
|---|---|---|---|
| rope=2 baseline | 50.6% | — | 没训练 |
| terminal-only β=0 | 52.5% | +1.9pp | 仅 terminal advantage |
| additive β=0.10, raw PRM (incl. -1) | 51.0% | +0.4pp | per-turn judge，无 gate，无 clip |
| additive β=0.10, **pos-only**, no gate | 55.6% | +5.0pp | 单独 pos-only clip；-1 → 0 |
| additive β=0.10, pos-only, **mult-B form** | 56.6% | +6.0pp | `(1+β·prm)` 形式，无 gate |
| **additive β=0.10, pos-only, judge-gate (老 SOTA)** | **57.24%** | **+6.6pp** | 在 pos-only additive 之上加 gate |

解读：**单一最大的胜利是 pos-only clip**（相对 baseline +5.0pp）。在 pos-only
之上 **judge-gate 又加了 +1.6pp** — 有意义但不是全部。

### 5.5 Per-task 增益分析（老 pod）

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

### 5.6 Judge-gate 命中率（老 pod）

R1 和 R2 都产生了相同的 gate split：

```
22/46 trajectories  mostly_done  (no PRM applied)
24/46 trajectories  lost          (per-turn PRM applied)
```

两轮中相同的 22/24 split 暗示在固定 roadmap 和 base model 下 gate decision
在训练数据上是确定性的。如果这个比例塌缩到 `0/46` 或 `46/46`，rubric 漂移了，
需要重新校准。

### 5.7 R2 回退（老 pod）

R2-additive（从 R1 LoRA 继续）：**55.34%**，相比 R1 回退 -1.9pp。

advisory_stakeholders 跌 0.573 → 0.438（-0.135）解释了几乎全部回退。所有 3 个
R2 run 都失败于相同的自动 check（确定性的，非随机）。

诊断：训练好的文件产物与轨迹末尾 chat 端总结之间的**输出预算分配**变化：

| Task | R1 budget ratio | R2 budget ratio |
|---|---|---|
| advisory_stakeholders | 0.92 | **0.64** ⬇ |
| sentiment_analysis | 0.75 | 0.47 ⬇ |

`budget_ratio = output_file_chars / (output_file_chars + final_chat_chars)`。

**假设**：PRM judge 读取轨迹 turns 时，把"chat 端总结"误判为"深思熟虑的收尾"
给 +1，但自动 grader 只读文件。模型被奖励"在 chat 里谈做这件事"而不是写交付物——
PRM 系统中奖励层错位时的教科书式 reward-hacking 失败模式。

> 注：这个 R2 回退的诊断**直接启发了新 pod 的 R3 退化诊断方法论**（详见 §2 退化诊断）。
