# jiuwenclaw + veRL async RL 实验诊断（2026-05-13 → 05-15）

整个长会话从 jiuwenclaw runtime bench smoke 演化到完整 verl-async + jiuwenclaw
GRPO 训练实验，共跑了 26 个版本（v32 → v57）。本文记录最终诊断：**为什么这条
路 RL 不收敛**，以及下次重试时应该改什么。

---

## TL;DR

| 配置 | 状态 | 关键现象 |
|---|---|---|
| OpenClaw + verl sync GRPO | ✅ 47.8% bench (`main`) | 已跑通 |
| OpenClaw + hand-crafted off-policy GRPO | ✅ 47.8% bench (`main`) | 已跑通 |
| **jiuwenclaw + verl async GRPO** | ❌ **训练 reward 退化** | 1 step LoRA 后 80% trajectory timeout，critic 从 0.166 → 0.047 |

**根本结论**：jiuwenclaw 作为 RL runtime 的问题不是工程修不完，是**信噪比太低**导致
GRPO 单步 gradient 就把 tool-call 能力打坏。下次重试需要从**算法层**（不是工程层）
入手 —— 加大有效 batch、降 lr、加 KL clip。

---

## §1 实验链路（最终能跑通的全栈）

历经 v32 → v57，最终一条能跑出 step + ckpt 的链路是 v57：

```
launch_meeting_jiuwen_async.sh
  ↓
pre-flight: kill stale + verify ports + GPU clean
  ↓
mock_trajectory_gateway.py (:9000)  ← 收 PerTurnSample batch
  ↓
veRL FullyAsyncRollouter + Trainer + MessageQueue
  ↓ (Ray actor 拓扑，3 进程独立 GPU)
2x jiuwenclaw stack (start_jw_headless.sh)
  ├── sitecustomize.py → inject_rl_online_rail.py (meta_path hook)
  ├── monkey-patch JiuWenClawDeepAdapter._build_agent_rails
  │   appends 同事的 RLOnlineRail
  └── MEMORY_ENABLED=false + wipe memory.db/daily_memory
  ↓
jiuwenclaw chat.send → vLLM (Qwen3-4B port 8000)
  ↓ RLOnlineRail captures (prompt_ids, completion_token_ids, logprobs) per turn
  ↓
TrajectoryUploader → POST mock_trajectory_gateway
  ↓
/tmp/jw_rail_v1/{traj_id}.jsonl  (PerTurnSample format)
  ↓
JiuwenClawAgentLoop._load_rail_v1_samples(session_id)
  + _build_response_from_rail_samples
  + left-truncate prompt_ids to max_prompt_length
  + response_logprobs = None (避免 veRL batch torch.cat 崩)
  ↓
AgentLoopOutput → MessageQueue → Trainer
  ↓
RolloutFilter (drop response_mask=0) →
ZeroMaskFix (force valid token on empty rows) →
QualityFilter (race-to-bottom group max<0.05 zero advantage)
  ↓
GRPO loss + PPO clip + KL → backward → optim
  ↓
CheckpointEngine NCCL bucket → vLLM hot-swap LoRA
  ↓
ckpt save (every param_version)
```

**14+ 个独立 patch 累计**才让这条链路活下来。完整 patch 清单见
`verl_patches/README.md` + 本文 §6。

---

## §2 v57 实测数据（终态）

v57 配置（最稳的一组）：

```
USE_RL_ONLINE_RAIL=1 MEMORY_ENABLED=false
JW_N_STACKS=2 ROLLOUT_N=2
BATCH_SIZE=4 REQUIRE_BATCHES=2     # require 8 samples/step (v55 用 16)
MAX_PROMPT_LENGTH=25000 (left-truncate 39k jiuwenclaw prompts)
MAX_RESPONSE_LENGTH=6000
VLLM_MAX_MODEL_LEN=65536
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=40000  # OOM fix
SAVE_FREQ=1 TRIGGER_PARAM_SYNC_STEP=2  # ckpt every 2 steps
LR=2e-6 LORA_RANK=32 LORA_ALPHA=64
```

跑了 5 个 trainer step，落了 3 个 ckpt：

| Step | 时刻 | critic_mean | critic_max | actor/loss | grad_norm | entropy | race_to_bottom |
|---|---|---|---|---|---|---|---|
| 1 | 19:08 | **0.166** | 0.65 | -0.0044 | 0.117 | 0.081 | 37.5% |
| 4 | 20:57 | **0.047** | 0.535 | -0.0021 | 0.037 | 0.151 | 0% |

**critic mean 掉 71%**。

trajectory-side 同步崩坏：

| 时间窗 | empty timeout 率 |
|---|---|
| step 1 收集期（base 模型 rollout） | ~0% |
| **step 4 batch 准备期（param_v1 应用后）** | **8/16 = 50%** |
| step 5 batch 准备期（param_v2 应用后） | 25 个 empty 在 84 min 内（~80%） |

每个 empty 都是 `status=timeout, history_len=0 or 1` —— **jiuwenclaw 在 600s
AGENT_TIMEOUT 内只产了 0-1 个 history event**，完全卡死。

---

## §3 根因诊断

### 3.1 信噪比击穿

Step 1 trainer 实际有效 signal trajectory:

```
8 samples 进 trainer
  - 2 ZeroMaskFix 兜底 (1 token 强 mask=1, reward=0 placeholder)
  - 3 race-to-bottom group dropped (max reward < 0.05)
  = ~3 条真有效 signal
```

3 条 trajectory 算 GRPO advantage，对 LoRA rank=32 × 6 target modules 做一次
gradient update —— **这个 update size 在 batch=3 的样本量下噪声极大**。

### 3.2 LoRA 打坏 tool-call 层

LoRA target_modules = `[q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj,
down_proj]` —— 这些是 attention + FFN 的核心层。GRPO 的 gradient 即使方向对、幅度小，
作用在这些层上也会**改变模型的 tool-use 决策路径**。

模型从某次成功 trajectory 学到的 "应该这样调 tool" 被推广到其他场景，但
- 推广不准 → tool call JSON 格式偏离 hermes parser 期望
- jiuwenclaw 卡在 streaming tool parser → 等 model 输出完整 JSON → 等不到 →
  600s AGENT_TIMEOUT 触发
- → trajectory 空 → 下一 batch signal 更弱 → 死循环

### 3.3 训推一致已经做到极致

之前 v32-v51 我们怀疑过：
- ❌ history.json 反解错（fix: rail-v1 直接抓 vLLM token_ids）
- ❌ memory.db daily_memory drift（fix: MEMORY_ENABLED=false + wipe）
- ❌ prompt 不一致（fix: rail-v1 prompt_ids + left-truncate）
- ❌ multi-turn mask 错（fix: rail-v1 step_index + tool gap mask=0）

这些**都 fix 了**，v57 的训练数据已经是史上最干净的一组。但模型仍然崩 ——
说明剩下的崩溃**不是工程问题，是 RL 算法层信噪比问题**。

### 3.4 为啥 OpenClaw 同 setting 不崩

OpenClaw GRPO 用同样 lr / LoRA rank / batch 配置训出 47.8%。区别：

| | OpenClaw | jiuwenclaw |
|---|---|---|
| 单 trajectory wall time | 2-3 min | 5-15 min |
| Prompt 长度 | ~2-5k | 17k-40k+ |
| 训练 batch 实际有效 sample | 8/8 (无 empty) | 3/8 (大量 race-to-bottom) |
| Tool-call 格式弹性 | CLI subprocess，解析容错高 | hermes streaming parser，JSON 严格 |
| LoRA 影响半径 | 局部（CLI 解析容错） | **放大**（一点偏离就 timeout）|

OpenClaw 每个 batch 有 8 个完整 signal；jiuwenclaw 有 3 个。**3 vs 8 在 LoRA rank=32
设置下，前者必崩**。

---

## §4 为什么 v32-v51 看似在训但没出 ckpt

回顾 v32-v51 那条线，从来没看到过 v57 这种 "训了 1 step 就崩" 的 pattern。原因：

**v32-v51 的 trajectory 大部分是假的**。20s ws.recv timeout bug 让 jiuwenclaw 冷
启动时（25-40s 加载 SOUL/IDENTITY/memory）就被掐死，agent_loop 收到 status=timeout
+ history_len=0 → emit EOS placeholder → 99% 是 EOS。trainer 看到的 batch 全是空
placeholder，ZeroMaskFix 救命，所以 update 几乎没动 LoRA → 不会出现 "1 step 就崩"。

v57 修了 20s timeout（→ 120s + 45s），加上 RLOnlineRail 拿真 trajectory data，
**第一次让 trainer 拿到真信号** → **第一次看见 GRPO 真的在工作** → 也第一次**看见
它把模型干坏**。

讽刺：v32-v51 看起来"在跑"但模型不动；v57 跑通了反而崩。

---

## §5 下次重试的方向

### P0: 不动 RL 算法的"工程优化路线"（**已证明不通**）

- 改 RolloutFilter / ZeroMaskFix 阈值 → 杯水车薪
- 加 stack 并发 → 不解决信号质量
- 关 memory drift → 已做了，没解决

**不要再走这条路。**

### P1: RL 算法层调整（**真正应该试**）

| 改动 | 期望效果 | 风险 |
|---|---|---|
| **batch ×2-4**（REQUIRE_BATCHES=8 → 32 samples/step） | 噪声 ×0.5，信号稳定 | 单 step 慢 4 倍（jiuwenclaw 已经很慢） |
| **lr 降 10x**（2e-6 → 2e-7） | 单 step LoRA 更新更小，模型不被打坏 | 收敛慢，可能需要 100+ step |
| **LoRA rank 减半**（32 → 16） | LoRA 容量小，update 影响范围小 | 长期收敛可能性也小 |
| **KL clip 加严**（kl_loss_coef 0.01 → 0.1） | 强约束 policy 不远离 base | 学得慢 |
| **从 OpenClaw R4' 47.8% LoRA 起步** | 已有 baseline LoRA，不从 base 直接 RL | 但 OpenClaw 训出的 LoRA 在 jiuwenclaw runtime 上只有 27.7%（cross-runtime drift），起点更差 |

### P2: 改 runtime 设计（**根本解但慢**）

jiuwenclaw 黑盒 + hermes streaming parser 对 LoRA 偏移容错差。如果同事愿意改 jiuwenclaw：
- 把 tool parser 切成 **非 streaming**（一次性吐完整 JSON 再解析，容忍格式抖动）
- 提供 **graceful timeout**（trajectory 失败但保留已有 history events 给训练）

但短期内不现实，依赖外部团队。

---

## §6 沉淀 patches 清单（已 commit, ready for next time）

本仓库 `jiuwenclaw-agent-loop-impl` 分支：

| 文件 | 作用 |
|---|---|
| `experiments/verl_port_poc/jiuwenclaw_agent_loop.py` | rail-v1 数据源 + truncate + response_logprobs=None |
| `experiments/verl_port_poc/launch_meeting_jiuwen_async.sh` | 硬化 pre-flight + rail wiring + drift fix env |
| `experiments/verl_port_poc/start_jw_headless.sh` | MEMORY_ENABLED=false + wipe + inject hook |
| `experiments/verl_port_poc/inject_rl_online_rail.py` | DeepAdapter monkey patch (sys.meta_path) |
| `experiments/verl_port_poc/sitecustomize.py` | 让 jiuwenclaw 多进程子进程都 import patch |
| `experiments/verl_port_poc/run_jw_app_with_rl_rail.py` | wrapper exec jiuwenclaw.app via runpy |
| `experiments/verl_port_poc/mock_trajectory_gateway.py` | aiohttp gateway 接 PerTurnSample → JSONL |
| `experiments/verl_port_poc/verl_patches/*.patch` | veRL 7-file unified diff (MessageQueue, FSDP2, RolloutFilter, ZeroMaskFix, QualityFilter, FSDP2 LoRA ckpt) |
| `experiments/verl_port_poc/tests/test_jiuwenclaw_agent_loop.py` | 16/16 unit tests pass |

下次接手的人不用从零开始 —— 上面所有"工程不通"的 fix 都在了，可以直接进入算法
调参（§5 P1）。

---

## §7 时间花费

- 总会话长度: 4 天 (2026-05-12 → 05-15)
- 实验版本数: v32 → v57 (26 versions)
- 真有出 ckpt 的版本: v57 only (1/26)
- v57 跑到 step 5 用了: 4h
- 总 GPU 时间消耗（估）: ~30 GPU·hours
- **训练结果**: 失败 (critic 退化 71%)

---

## §8 结论

**作为白盒/黑盒对比的实验数据，已经够了**。结合 main 分支已有的：

- OpenClaw + GRPO 47.8% (能训)
- jiuwenclaw + GRPO 训练 reward 退化 (不能训)

足以写报告 "jiuwenclaw 当前架构不适合 RL 训练" 这个结论。

如果还要继续，**请先调 RL 算法层**（§5 P1），别再调工程。

如果调到能跑通，下游真正能解决的是 §5 P2 —— 让 jiuwenclaw 改 streaming tool
parser 容错。否则训得再好，模型一动 jiuwenclaw 就 timeout。
