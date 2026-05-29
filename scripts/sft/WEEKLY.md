# 周报 — Qwen3-4B + PinchBench RL/SFT 实验

**Week**: 2026-05-17 ~ 2026-05-22
**Branch**: `jiuwenclaw-agent-loop-impl` | commits: `5af2e64..9cd0155`

## 核心结论

**这周最大产出是定位并修复了 OC + vLLM 的一个 production-level 关键 bug, 推翻了 v32-v41 的"RL 微提升"结论, 并锁定真实瓶颈在 32K context 而非模型能力, 为后续 multi-agent 路线奠定基础。**

## 三个实质 deliverable

### 1. OC + vLLM hermes parser bug 发现 + 修复 (`PATCH-B`)
- **问题**: OC 在多 turn tool 调用 turn 2+ 把 `tools` 字段置空 (Anthropic 兼容代码), vLLM hermes parser 因此不解析模型输出的 `<tool_call>` 文本, 导致所有 Qwen3 多 turn agentic bench 被压低 ~4pp
- **修复**: 25 行 JS patch, 在 OC 响应处理结束时扫描 text 提取 `<tool_call>` 转结构化 (`scripts/sft/apply_oc_hermes_patch.sh`)
- **影响**: 所有用 vLLM + Qwen3 + hermes 的人 (我们 + Codex team agent + 任何下游)

### 2. 推翻 v32-v41 "RL 微提升" 结论
- 历史 v38_ckpt9 = 0.517 (大于 base 0.474, 看起来 RL 涨 +4pp)
- 修 bug 后对照 (3 run × 5 task, 共 30 sample):
  - **base Qwen3-4B: 0.510 ± 0.019**
  - **v38_ckpt9 RL: 0.464 ± 0.017**
  - **Δ = -4.6pp (RL 实际比 base 低, 统计显著)**
- 即历史"涨"完全是 bug 把 base 压低制造的假象
- **未来所有 RL 实验必须 用 patched OC, 否则 reward signal 不可信**

### 3. 锁定瓶颈: context 长度, 不是模型 weight
- PinchBench transcript 120K-250K chars, Qwen3-4B 原生 32K context (YaRN 64K 仍紧)
- 实测 council_votes / gov_speaker_summary 等长任务在 turn 2+ 触发 `400 max_tokens too large` 错误 (cumulative input 超 32K)
- 单 agent 模型物理上**无法 rollout 出 teacher 的"chunked read → synthesize"轨迹**
- 这就是为什么 single-agent RL 探索不出强策略 (action space 里没有可行的长轨迹)

## 失败的尝试 (有 finding 价值)

| 实验 | 结果 | 学到什么 |
|---|---|---|
| SFT v1 (28 records, 2 epoch) | 0.489 ≈ base | 训练步数太少 |
| SFT v2 (42 records, 15 epoch, lr 5e-5) | 0.354 (-14pp) | overfit + 长上下文样本被砍 |
| **SFT overfit (9 records, 50 epoch, lr 1e-4)** | **0.148 (-34pp)** | 极致 overfit 触发 32-turn micro-read 死循环, 模型从 base 退化 |

→ 验证: **小规模 + 高 epoch overfit 会破坏 base 能力**, 这是 distillation 反例案例。

## 下一步方向 (已 align Codex)

放弃 single-agent RL fine-tune, 转向 multi-agent team:
- **Policy agent** (DSv4 Flash 或 Qwen3) 全局 plan + delegate
- **Worker agents** (Qwen3) 每个处理一个 chunk, 输出 short summary
- **Final agent** (Qwen3) 综合 summaries 写最终答案

Codex 已实验单 task: `task_meeting_tech_action_items` = **0.773** (vs single-agent base 0.692, RL 0.626, +8pp)。等 Qwen3-as-policy 对照结果, 决定是否需要训 policy LoRA。

## 时间分布 (5 天)

| 任务 | 时间 |
|---|---|
| DSv4 Flash teacher 收集 (val_5 + train_23 共 8 runs × 5h) | 1d |
| SFT pipeline 实现 (7 阶段, 8 个脚本) | 1d |
| SFT 训练 + bench v1/v2/v3/v8/v9 | 1.5d |
| OC bug 调试 + PATCH-B + 验证 | 1d |
| Report + commit + docs | 0.5d |

## 落盘 artifacts (git tracked, repo 内可复用)

```
scripts/sft/
├── REPORT.md                       <- 完整研究报告
├── oc_hermes_patch.md              <- bug RCA
├── apply_oc_hermes_patch.sh        <- 一键应用 patch
├── 1_extract_oc_transcript.py      <- OC JSONL → 统一 messages
├── 1b_merge_reads.py               <- 合并同文件连续 read (msg 砍 40%)
├── 1c_normalize_paths.py           <- 工作区路径随机化
├── 2_filter_quality.py             <- score + top-K + min_msgs 过滤
├── 3_to_chatml.py                  <- OpenAI ChatML + <think> 格式
├── 3b_truncate_tools.py            <- 长 tool result 头尾截断
├── 4_validate.py                   <- tool_call_id 配对 + token 预算
├── train_qwen3_lora.py             <- Qwen3-4B LoRA SFT (assistant-only mask)
├── bench_base.sh / bench_sft_lora.sh
├── run_v1/v2/v3/v9.sh              <- 6 个版本训练 launcher
└── (commit: dccd9e7, 9cd0155)
```
