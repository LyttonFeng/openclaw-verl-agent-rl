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
4. **fp32 forward**（重要）通过 Qwen3-4B body + lm_head，用 fused `cross_entropy`
   得 token log-probs，policy-gradient loss `-(adv * logp).mean()`。
5. Backward、`clip_grad_norm_(1.0)`、每 `grad_accum` 个样本 `optimizer.step()`。
6. 保存 LoRA adapter，退出。

**⚠ 关于 fp32（2026-05-09 发现）**：早期版本用 bf16 autocast，但在
**transformers 4.57 + peft 0.19 + 80K context** 组合下会数值溢出 → loss=NaN
→ optimizer 把 NaN 写入 LoRA 参数 → 整个 LoRA 全部 NaN。详见
[`reproduction.md`](reproduction.md) §"训练 dtype" 复现 trace 与修复细节。
现在 trainer 强制 fp32（`torch_dtype=torch.float32`、去掉 autocast wrapper），
慢约 2x 但 100% 稳定。

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

### PRM 与 [filter + PPO] 组合的实证设计要点 (2026-05-10 ablation)

把 PRM 直接叠加到 [filter + PPO] setting 之上 **会退化 -1.2pp**（R1+PRM 朴素配置 = 45.7%，
vs R1' no-PRM = 46.9%）。诊断发现两个根因：

1. **PRM 在已及格 trajectory 上是负向扰动**。tech_action_items（R1' 0.642）和
   sentiment_analysis（R1' 0.667）这种"中等偏强"task，原模型已在局部最优附近，PRM
   引入的 milestone-aligned 偏置反而把它从局部最优拉走。binary `mostly_done` gate
   太保守（要 ~0.8 才放过），中段反应不过来。
2. **PRM token-level 应用导致"长 turn 多放大"偏置**。multiplicative 公式给一个
   PRM=+1 turn 的每个 token 都乘 1.5；1000-token 长 turn 比 50-token 短 turn 拿到
   20× 更多梯度推力。**直接训练出"多写字"的副作用**。

修正这两条之后 R1+PRM = **47.3%**（**+0.4pp vs R1' no-PRM, +1.6pp vs naive PRM**）：

#### Fix 1：Reward-gated PRM（绝对阈值替代 binary gate）

`apply_quality_filter.py --prm-reward-gate 0.5`：score >= 0.5 的 trajectory 直接把
`prm_turn_scores` 清零，等同于纯 terminal gradient。比 `judge_terminal_completion`
的 binary 判断更直接、更细粒度。

#### Fix 2：Per-turn loss weighting（消除长 turn 偏置）

`train_meeting_grpo_step.py --per-turn-loss`：每个 token 权重 = 1/n_tokens_in_its_turn，
确保**每个 turn 对梯度贡献相等，不被 turn 长度影响**。这切断 PRM 推策略向"啰嗦"
偏移的机制。

#### Per-task 实证（PRM ablation, 5 task × 3 run）

| Task | base | R1' | R1+PRM v1 (naive) | R1+PRM v2 (with fixes) |
|---|---:|---:|---:|---:|
| advisory_stakeholders | 0.384 | 0.424 | 0.427 | **0.443** |
| council_votes (弱项) | 0.198 | 0.204 | **0.235** | 0.221 |
| gov_speaker_summary | 0.425 | 0.407 | 0.418 | 0.397 |
| tech_action_items (强项) | 0.586 | 0.642 | **0.597** ↓ | 0.639 (修复) |
| sentiment_analysis (强项) | 0.641 | 0.667 | **0.608** ↓ | 0.665 (修复) |
| **TOTAL %** | 44.68 | **46.89** | **45.69** | **47.29** |

#### 关键观察

- **PRM 系统性帮弱项、伤强项**：在没有 fix 的 v1 里，R1' < 0.3 的弱项 council_votes
  从 0.204 → 0.235（+15%）；R1' > 0.6 的强项 tech_action_items 0.642 → 0.597（-7%）、
  sentiment 0.667 → 0.608（-9%）。**reward gate 直击这一点**。
- **Per-turn loss 几乎不动 v1 没有 fix 的强项退化**（强项数据本身需要 reward gate
  把 PRM 关掉），但**配合 reward gate 之后保护了 advisory_stakeholders 不被
  PRM 副作用反噬**（v2 0.443 > v1 0.427）。
- **PRM β=0.5 in multiplicative**（不是 docs 默认的 1.0，也不是 additive 的 0.10）
  在 5-task bench 噪声下信号最稳：β 太小看不到效果，太大破坏 filter+PPO 稳定性。
- **整体增量 +0.4pp 偏小**，逼近 5-task bench 的 ±1pp 噪声边界。要做严格性证明
  需要更大 task suite 或多 seed 验证。

## 训练数据质量过滤（Race-to-bottom 防御）

在 N=2 GRPO + judge-based reward 下，会出现 **race-to-bottom** 退化模式：当一个 group 内的两条 rollout 都质量低（judge 噪声/bias 让 lazy 答案偶然拿高分），GRPO 仍然会按 group-relative 给"两个都差但稍微不那么差"的那条正 advantage。模型把 lazy 当成"好榜样"学，慢慢漂向偷懒模式。

**实证**：R3 round 用 vanilla GRPO 训练后 bench 从 R2 的 46.4% 退化到 43.3%（-3.1pp）。诊断发现 16 个 useful group 里有 4 个是 race-to-bottom（max reward < 0.4）。R3 v2 加上下面的过滤后恢复到 46.2%（≈ R2，止住退化），advisory_stakeholders 单任务从 0.358 → 0.490。

**核心规则**：**只对 positive-advantage 样本做质量过滤**。负样本是"避免信号"，质量差反而是好的对比例，全部保留。只有正样本（模型要 imitate 的）需要质量审查。

**保守的三道过滤**（实测，避免误伤）：

1. `group_max(reward) >= 0.4` — 整组都低分（race-to-bottom）则整组扔掉
2. `final_reply_chars + sum(written_file_chars) >= 500` — 总产出过少
3. `n_successful_tool_calls >= 1` — 没调工具的扔掉

**不要用的过滤**（实测会过度过滤）：

- ❌ `[[xxx]]` glitch token 检查 — Qwen3 chat template 残留，R2/R3 都有，**不是质量信号**
- ❌ 单看 final reply 字符数 — 内容常在 write 出来的 markdown 文件里
- ❌ 单看 written file 字符数 — 部分任务回复短文件长，部分相反

**插入位置**：`select_grpo_samples.py` 之后、训练之前。参考 `rl/train/apply_quality_filter.py`。

## PPO 三件套：importance ratio + clip + KL

### 为什么 vanilla policy gradient 不够

最初 trainer 的 loss 只有：

```
loss = -advantage × log P_θ(token)
```

这是 **REINFORCE 加 group baseline**，**严格说不是 PPO/GRPO**——缺少：
1. Importance ratio（rollout 和 update 不共策略时的修正）
2. PPO clip（限制单步更新幅度）
3. KL penalty（防止漂离参考策略）

实证：R3 round 用 vanilla loss 就从 R2 的 46.4% 退化到 43.3%（-3.1pp）。
单加 race-to-bottom 过滤恢复到 46.2%（持平 R2，无增量）；进一步加上 PPO+KL
后 R3 v3 = 47.5%（**+1.1pp 真实增量**）。结论：**只有过滤不够，必须配 PPO**。

### 数学形式（最终 loss）

```
log_ratio_t  = log P_θ(t) - log P_old(t)         # per trainable token
ratio_t      = exp(log_ratio_t.clamp(-20, 20))   # 数值保护

surr1_t      = ratio_t · A_t
surr2_t      = clip(ratio_t, 1-ε, 1+ε) · A_t
pg_loss      = -mean_t[ min(surr1_t, surr2_t) ]   # PPO clipped surrogate

kl_t         = exp(-log_ratio_t) + log_ratio_t - 1   # k3 estimator (unbiased)
kl           = mean_t[ kl_t ]

loss         = pg_loss + β · kl
```

**关键设计**：

- **min** 操作对正负 advantage 都对：A>0 时截断"过度提升"，A<0 时截断"过度远离"
- **k3 estimator** `exp(-log_ratio) + log_ratio - 1` 永远非负，方差远小于 k1
- **ε = 0.2**（PPO 标准值）；**β = 0.02**（实测 KL 落在 0.001-0.003 健康区间）

### 实现：避开 reference model

标准 PPO 需要常驻 π_ref forward 算 KL，**显存翻倍**。我们的简化：

> 让 **π_ref = π_old**（rollout 时的策略），把 P_old 一次性算好存盘，
> 训练时仅用 saved log_probs，**不需要加载第二份模型**。

代价：每轮的"KL 锚"会随 chain 漂移（R1' 锚 base，R2' 锚 R1'，...）。
单轮训练里两者效果一样，多轮 chain 时长程"绝对漂移"无 KL 约束。
为消除这个限制需把 ref 改成固定 anchor（如 base Qwen3-4B），见
`rl/train/compute_rollout_logprobs.py` 设计预留。

### bf16 saved P_old 的合理性

`compute_rollout_logprobs.py` 默认 bf16 而非 fp32：
- vLLM 实际 rollout 就是 bf16，bf16 重算反而更接近真实 P_old
- fp32 + 17k+ tokens 序列在单卡 / 双卡都会 OOM（无 gradient checkpointing）
- bf16 cuts attention memory in half，让 SDPA 走 flash backend

实测：fp32 训练时计算的 log_p_θ vs bf16 saved P_old 在 step 0 时：
- **median ratio = 1.0000**（绝大多数 token 完全不受影响）
- **mean ratio ≈ 1.001**（无系统 bias）
- **1-3% 的 token ratio 在 [0.8, 1.2] 之外**（多数是 log_p ~ -20 的低概率 token）

这 1-3% outlier 会被 PPO clip 自然吸收，不污染梯度方向。clip 本就是为了
处理 ratio 漂移设计的，bf16 引入的精度噪声远小于多步训练后的 policy 漂移。

### 流程

```
[现有] graded_trajectories_prm_valid.jsonl
       ↓ (apply_quality_filter.py)
       graded_trajectories_quality_filtered.jsonl
       ↓ (compute_rollout_logprobs.py with PREV adapter)
       rollout_logprobs.jsonl   ← P_old per token
       ↓ (train_meeting_grpo_step.py with PPO + KL)
       新 LoRA
```

CLI：训练时加 `--logprobs-file <path> --clip-eps 0.2 --kl-beta 0.02`。
不传 `--logprobs-file` 自动退回 vanilla PG（向后兼容）。

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
