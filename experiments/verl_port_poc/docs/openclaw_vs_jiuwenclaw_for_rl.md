# OpenClaw vs JiuwenClaw —— 作为 Claw Runtime for RL Training 的对比

写给 jiuwenclaw 团队同事。基于本仓库两条已落地分支的实测对比：

- `experiment/verl-port` —— OpenClaw runtime + veRL GRPO（**已跑通**，main baseline 47.8% mean on 5 meeting test tasks）
- `jiuwenclaw-agent-loop-impl` —— JiuwenClaw runtime + veRL async GRPO（**未跑通到出 LoRA**，本次夜跑 v32→v50 一路 patch 但 critic score 仍在退化）

目标：让同事看清楚 JiuwenClaw 用于 RL 训练时差在哪里，需要在 runtime 侧改什么才能让它真正能训。

---

## 0. 核心结论（必读）

> **JiuwenClaw 当前架构违反 RL 训练的 stationary MDP 假设。在 runtime 提供严格 fresh-session 模式之前，所有训练侧 patch 都是 work around，无法解决根本问题。**

### 为什么这是 RL 训练的根本冲突

RL 算法（Q-learning / Policy Gradient / GRPO …）的收敛性证明全部建立在 **stationary MDP** 假设之上：同一个 state + action，环境反馈的 next state 和 reward 必须是同一个分布。否则：

- π(a|s) 的梯度方向今天对、明天错
- Reward signal 噪声爆炸（不是 policy 变了，是 env 变了）
- IID 假设破裂 → 梯度估计有 bias

### JiuwenClaw 实际的 env 行为

```
trajectory 1: state s → action a → reward r1, env 写 MEMORY.md + 学新 skill_A + 累积 daily_memory
trajectory 2: state s → action a → reward r2 (≠ r1!), env 又写新东西
trajectory 999: 同一个 task，env 已经完全不是原来的东西
```

**而且漂移方向跟训练目标耦合（更糟）**：

- 模型成功 → jiuwenclaw 把这次 trace promote 成 skill → 下次模型不用学就调 skill → reward=1，但**policy 没学到东西，是 env 替它做了**
- 模型失败 → jiuwenclaw 写"失败原因"进 memory → 下次启动看到 → bias 行为 → **你训练的不是 policy，是"被 jiuwenclaw 提示后的 policy"**

### 结果

1. **训练 reward ≠ bench reward**（env state 不同），训推一致从根上不成立
2. **多 stack 并发更糟** —— 每个 stack 独立漂移，policy 收到的是 N 个不同 env 的混合梯度
3. **OpenClaw 能训通是因为它每条 trajectory 起 subprocess、结束销毁，env 严格确定**

### Runtime 侧需要做的（必须）

**严格 fresh-session 模式**：启动时跳过所有持久化 state 加载，结束时不写任何 state。具体：

- 不加载/不写 `memory.db`、`daily_memory/*.md`、`MEMORY.md`
- 不自学 skill（不 promote trace 进 `skills/`）
- 不读 `extensions/` 累积状态
- Identity / SOUL / AGENT / TOOLS.md 可以读（这些是静态的），但加载要 lazy / cache

**在这之前，训练侧的所有 patch（RolloutFilter / ZeroMaskFix / QualityFilter / workspace 清理）都只是 work around**，能让训练不崩，但无法让信号干净。

---

## 1. 核心区别：白盒 vs 黑盒

用户的判断是对的。整理成一张表：

| 维度 | OpenClaw（白盒） | JiuwenClaw（黑盒） |
|---|---|---|
| **执行方式** | CLI subprocess（每条 trajectory 一个进程） | WebSocket 长连接 + Ray actor + IM pipeline |
| **System prompt** | 直接在 prompt 里可见，可定制 | 由 jiuwenclaw 内部 `AGENT.md / IDENTITY.md / SOUL.md / TOOLS.md` 拼接，训练侧不可见 |
| **Tool call 格式** | 模型直接输出 OpenAI-style JSON tool call，stdout 可解析 | 经过 E2A wire codec / hermes tool parser / IM channel routing 多层封装 |
| **Env 反馈** | tool 返回结果原样回灌 prompt | 走 channel manager → IM outbound pipeline，模型看到的是 jiuwenclaw 处理过的版本 |
| **Trajectory 拿法** | stdout 直接读完整 conversation | 必须事后从 `history.json` 反解，且 history 已经被 jiuwenclaw 加工过（token-level chunking、ack message 等） |
| **Reward function 训推一致** | 直接 grep stdout，训和 bench 用同一字符串 | 训练 reward 看 `history.json` + workspace 文件；bench 看 jiuwenclaw 自己的 grading API。两条路径，容易 drift |
| **Workspace 状态** | 进程结束就清理，纯函数 | class-level lock + persistent SQLite memory.db + daily_memory.md 累积污染 |
| **失败模式可观测性** | 进程 exit code + stderr 即诊断 | 要看 WS log + agent_server log + tool executor log + jiuwenclaw 内部 trace 才能拼出真因 |

---

## 2. 实测数字对比

### Trajectory wall time（per task，Qwen3-4B，5 meeting tasks）

| Runtime | 每 trajectory wall time | 主要开销 |
|---|---|---|
| OpenClaw | ~2–3 min | 模型生成 + tool exec |
| JiuwenClaw | ~5–10 min（**3× 慢**） | 模型生成 + WS protocol ceremony |

实测 `read_file` 读 5KB 文本花 **12.2 秒**（jiuwenclaw `method_exec_time_ms=12226`）。这不是 bug，是企业级 agent framework 的开销 —— 但 RL 训练每 step 需要 N×batch 条 trajectory，3× 慢直接放大成训练吞吐瓶颈。

### Token efficiency（trajectory 长度）

用户的判断也对：

- OpenClaw trajectory ≈ N 轮 (user/assistant/tool) JSON，**几千–一万 token**
- JiuwenClaw trajectory **token 数显著更长**，原因：
  - Token-level chunking：每个模型 token 一个 chunk message，history 里有大量 metadata（protocol_version / response_kind / correlation_id / seq）
  - IDENTITY / AGENT / SOUL / TOOLS.md 每 session 加载一遍进 prompt
  - daily_memory.md 累积历史（v35 patch 之前 5.9MB → 启动读 12 秒）
  - ack message + channel routing message 也占位

后果：
1. **vLLM context overflow**：`VLLM_MAX_MODEL_LEN=65536` 都频繁触发 "input has 75668 tokens" 错误（v49 日志 5 次）
2. **`response_mask` 计算复杂**：jiuwenclaw 把 assistant token 和 framework token 混在一起，我们必须 reverse-engineer 拆出哪些是模型真生成的（见 `jiuwenclaw_agent_loop.py` 的 mask 重建逻辑）
3. **训练吞吐**：相同 batch 实际 GPU compute 远多于 OpenClaw

---

## 3. 在 RL 训练中触发的具体问题（按解决顺序）

### 3.1 Trajectory 高 timeout 率（v32 首次夜跑 70%）

**根因**：jiuwenclaw 默认开启 memory 系统，每条 trajectory 启动时加载 daily_memory.md（5.9MB），单是读这个就 12 秒；累积下去整个 session 90s 才走完几个 turn → 超 AGENT_TIMEOUT。

**patch**: `jiuwenclaw_agent_loop.py` chat.send params 加 `enable_memory: False` + 清空 daily_memory 文件夹。

**效果**：timeout 率 70% → 19%。

**runtime 侧应该做的事**：
- 提供 stateless 模式开关，RL 训练时跳过 memory 加载
- 或允许 per-session 隔离 memory（不复用全局 daily_memory）

### 3.2 Trajectory 完全没工具调用（全 `<think>`）→ reward=0

**根因**：训练 dataset 的 `extra_info` 只有 task_id/category 没有 `workspace_files` 字段。jiuwenclaw 启动时 workspace 是空的，模型 `read_file("transcript.md")` 找不到 → 999 turns 全在 `<think>` 推理 → 0 tool call → reward=0。

**patch**: `jiuwenclaw_agent_loop.py:_load_task_workspace_files(task_id)` 从 task YAML frontmatter 解析 workspace_files，把 assets 拷进 workspace。

**runtime 侧应该做的事**：
- jiuwenclaw 提供 `setup_workspace(files: List[FilePair])` 显式 API
- 或者 chat.send 时支持 attached files 参数（绕开 workspace 概念）

### 3.3 Empty response trajectory 污染 batch

**根因**：jiuwenclaw timeout / EOS placeholder 的 trajectory 是 `response_ids=[EOS]`, `response_mask=[0,...,0]`，被无条件推进 MQ，trainer 凑齐 batch 里 8/8 都是这种 → 0 梯度训练 step → critic score 退化（v50 实测）。

**patch（双层防御）**：
1. **Rollouter 侧**（`fully_async_rollouter.py:_process_single_sample_streaming`）—— `response_mask.sum() == 0` 时直接 return，不推 MQ
2. **Trainer 侧**（`ray_trainer.py:_fit_compute_advantage`）—— ZeroMaskFix 兜底，强制 mask 第 1 位 = 1，reward = 0，避免 NaN

**runtime 侧应该做的事**：
- 让 timeout / 异常 trajectory **返回明确的 status code**（成功 / agent_timeout / parser_error / ws_disconnect），不要假装"完成"但返回空 response
- 训练侧能根据 status 决定丢弃 vs 重试 vs 计入失败 reward

### 3.4 Race-to-bottom group → 全 batch advantage=0

**根因**：GRPO group size = 2，jiuwenclaw 不稳定导致大量 group 两条 trajectory 都失败（reward 都 ≈ 0），组内 advantage normalization 后 = 0，又是 0 梯度 step。

**patch**: `ray_trainer.py:_fit_compute_advantage` 加 race-to-bottom filter（group_max < 0.05 丢弃整个 group），含 50% drop 上限 fallback 避免空 batch。

**runtime 侧应该做的事**：
- 提高 trajectory success rate 是最根本的（见 3.1–3.3）
- 当前 patch 是治标

### 3.5 训练栈 vs Bench 栈姿势不一致（v51 新发现）

训练用 `start_jw_headless.sh`（headless，wipe data dir，无 identity 文件，`MAX_ITERATIONS=8`），bench 用 `start_jw_pod.sh`（complete stack，保留 identity，无 MAX_ITERATIONS cap）。两份脚本各自演化，**根本不是同一个 env** —— 同一个 LoRA 在两个 stack 下行为不同。

**症状**:
- 训练侧 jiuwenclaw log 频繁报 `File not found: SOUL.md / IDENTITY.md / HEARTBEAT.md / USER.md`（被 wipe 了）
- bench 侧这些文件齐全（pinchbench 那侧 setup 完整）
- 模型 persona prompt 不同 → 同样 LoRA bench 表现可能比训练 critic 看到的更好或更差

**修复**:
- 选 A：bench 改用 `start_jw_headless.sh` 同样 wipe + 同样 MAX_ITERATIONS，需改 pinchbench harness 启动逻辑
- 选 B：训练侧 seed identity 文件，但这样又要保证 bench 也 seed
- **最干净的还是 P0 #1 严格 fresh-session 模式** —— jiuwenclaw 自己提供"启动时 reset 到 canonical state"开关，训推都用它

### 3.6 全 token stream + parser 抖动

实测 v49 vLLM 报：

```
hermes_tool_parser.py:417 AssertionError
Error trying to handle streaming tool call
```

发生在 jiuwenclaw 生成长 trajectory 中间 tool call 解析时。OpenClaw 走批量返回，没这个问题。

**runtime 侧应该做的事**：
- 检查 jiuwenclaw 跟 vLLM streaming tool parser 的配合
- 或允许切换非 streaming 模式

---

## 4. 关键脚本 / 文件位置（跨两个分支）

### OpenClaw 路径（`experiment/verl-port` 分支，已跑通）

```
experiments/verl_port_poc/
├── launch_meeting_openclaw_lora.sh        # GRPO + OpenClaw 主 launcher（main baseline）
├── launch_meeting_cispo_lora.sh           # CISPO 变种
├── launch_meeting_vanilla_lora_v2.sh      # baseline 对照
└── run_bench_step{8,16}.sh                # bench LoRA on 5 test tasks

# 外部依赖（pod 上）
/workspace/verl_port/openclaw_integration/rl/
├── agent_loop/config.yaml                 # openclaw_agent loop 注册
├── agent_loop/openclaw_agent.py           # AgentLoopBase 子类
└── train/reward_manager.py                # compute_score 入口
```

OpenClaw 的 reward 直接读 CLI stdout，**训练 reward function 和 bench reward function 用同一字符串**，零 drift 风险。

### JiuwenClaw 路径（`jiuwenclaw-agent-loop-impl` 分支，未完成）

```
experiments/verl_port_poc/
├── jiuwenclaw_agent_loop.py              # ★ 核心：AgentLoopBase 子类
│   ├── _load_task_workspace_files()      # YAML → workspace 拷贝（3.2 fix）
│   ├── _build_workspace()                # 多 stack 隔离 workspace
│   ├── _pick_stack()                     # per-stack lock + round-robin
│   ├── chat.send(..., enable_memory=False)  # 3.1 fix
│   └── extra_fields = {                  # 训推一致 payload
│       "session_id", "task_id",
│       "status", "timed_out",
│       "history_path",  workspace_path", "transcript",
│       "stack_idx"
│     }
│
├── start_jw_headless.sh                   # 起单个 jiuwenclaw WS stack（无 vLLM，复用 veRL hybrid engine）
├── start_jw_pod.sh                        # pod 多 stack 启动 helper
├── jiuwen_lora_sync.py                    # LoRA hot-swap into jiuwenclaw（bench 用）
├── restart_jw_stack_with_lora.sh          # bench-time stack restart
├── launch_meeting_jiuwen_path_a.sh        # Path A: jiuwenclaw + veRL HTTP vLLM (smoke 用)
├── launch_meeting_jiuwen_async.sh         # ★ Path B: fully async GRPO (本次夜跑用的, v32→v50)
├── launch_meeting_jiuwen_lora.sh          # bench LoRA via jiuwenclaw runtime
├── run_jw_bench.sh                        # bench wrapper
├── smoke_jiuwenclaw_agent_loop.py         # 单元 smoke
└── tests/
    ├── test_jiuwenclaw_agent_loop.py     # mask 重建 / extra_fields 单测
    ├── test_jiuwen_lora_sync.py
    └── fixtures/jiuwen_history_sentiment.json   # 真实 history.json 样本

# 本地 reward function（与 bench 用同一份）
rewards/meeting_reward.py                  # 读 workspace_path + transcript
                                           # 训和 bench 同一字符串 = 训推一致基础

# 外部依赖（pod 上，同事接手后需自己装）
/root/jiuwen_work/jiuwenclaw/              # jiuwenclaw runtime
/root/jiuwen_work/pinchbench/              # bench harness（同事 fork，含 WS adapter）
/root/.jiuwenclaw_{0,1}/                   # 每 stack 独立 JIUWENCLAW_DATA_DIR
```

### 4.1 veRL 侧必须打的 patch（pod `/root/verl/`，重启不要丢）

⚠️ **这些 patch 没在本仓库里**，因为是直接改 veRL 源码。同事接手时按下面四段贴回去：

#### Patch ① `verl/experimental/fully_async_policy/message_queue.py:26`

不打 → MessageQueue Ray actor 永远 PENDING，整个训练卡住启动。

```python
# 改前
@ray.remote(num_cpus=2, max_concurrency=20)
# 改后
@ray.remote(num_cpus=0, max_concurrency=20)
class MessageQueue:
```

#### Patch ② `verl/experimental/fully_async_policy/detach_utils.py:~162`

不打 → STANDALONE vLLM 模式 `global_steps=None`，`abs(None - None)` TypeError 崩。

```python
# 原本：直接 zip 相减
# param_version_diff = [abs(a - b) for a, b in zip(param_version_end, param_version_start)]

# 改后：先把 None sanitize 为 0
_vs = [0 if v is None else v for v in param_version_start]
_ve = [0 if v is None else v for v in param_version_end]
param_version_diff = [abs(a - b) for a, b in zip(_ve, _vs, strict=False)]

# 同处：trajectory_param_versions 也要 None→0
trajectory_param_versions = [0 if v is None else v for v in final_batch.non_tensor_batch["max_global_steps"]]
```

#### Patch ③ `verl/experimental/fully_async_policy/fully_async_rollouter.py` —— `_process_single_sample_streaming`

§3.3 fix：rollouter 侧 drop empty trajectory，**不让空样本进 MQ 污染 trainer batch**。

```python
async def _process_single_sample_streaming(self, rollout_sample: RolloutSample):
    """Process a single sample streamingly"""
    ret = await self.async_rollout_manager.generate_sequences_single(rollout_sample.full_batch)
    rollout_sample.full_batch = ret

    # === 训推一致 fix: drop all-empty trajectories ===
    # jiuwenclaw timeout placeholders have response_mask=0. Without this,
    # trainer pulls them as valid samples → ZeroMaskFix forces mask=1
    # reward=0 → zero gradient + wasted step counter.
    _rmask = ret.batch.get("response_mask") if hasattr(ret, "batch") else None
    if _rmask is not None:
        try:
            _valid = int(_rmask.sum().item())
            if _valid == 0:
                print(f"[RolloutFilter] dropping empty trajectory sample_id={rollout_sample.sample_id} (response_mask all zero)", flush=True)
                self.dropped_stale_samples += 1
                self.processed_sample_count += 1
                return
        except Exception as _e:
            print(f"[RolloutFilter] WARN check failed: {_e}", flush=True)
    # === end fix ===

    rollout_sample.full_batch.non_tensor_batch["uid"] = np.array(
        [f"uid_{rollout_sample.sample_id}"] * len(rollout_sample.full_batch), dtype=object
    )
    rollout_sample.rollout_status = await self.get_statistics()
    success = await self.message_queue_client.put_sample(
        sample=ray.cloudpickle.dumps(rollout_sample),
    )
    # ... 原代码继续
```

也需要在 `__init__` 里初始化 `self.dropped_stale_samples = 0`（如果还没有的话）。

#### Patch ④ `verl/experimental/separation/ray_trainer.py` —— `_fit_compute_advantage`

两段连续 fix，加在 `compute_advantage` 调用**之前**：

**Part A — ZeroMaskFix（兜底，防 NaN）**：

```python
# Empty EOS placeholders (jiuwenclaw timeout) have response_mask all 0.
# log_prob compute on these div-by-zero → NaN → poisons entire batch loss.
try:
    _resp_mask = batch.batch.get("response_mask")
    if _resp_mask is not None:
        _valid_per_row = _resp_mask.sum(dim=-1)
        _zero_rows = (_valid_per_row == 0).sum().item()
        if _zero_rows > 0:
            _zero_idx = (_valid_per_row == 0).nonzero(as_tuple=True)[0]
            for _i in _zero_idx.tolist():
                batch.batch["response_mask"][_i, 0] = 1
                batch.batch["token_level_rewards"][_i, 0] = 0.0
                if "token_level_scores" in batch.batch:
                    batch.batch["token_level_scores"][_i, 0] = 0.0
            print(f"[ZeroMaskFix] forced 1 valid token on {_zero_rows} all-empty rows to prevent NaN", flush=True)
            metrics["filter/zero_mask_rows"] = _zero_rows
except Exception as _e:
    print(f"[ZeroMaskFix] WARN: {_e}", flush=True)
```

**Part B — GRPO race-to-bottom filter**（§3.4 fix）：

```python
# If group_max(reward) < 0.05, zero out token_level_rewards for that group
# so GRPO advantage = 0. Prevents imitating garbage samples that "won" within
# a low-reward group. Fallback: if >50% groups would drop, skip filter to avoid
# noop training step.
try:
    _n_repeat = int(self.config.actor_rollout_ref.rollout.n)
    _traj_rewards = batch.batch["token_level_rewards"].sum(dim=-1).cpu().numpy()
    _num_groups = len(_traj_rewards) // _n_repeat
    _dropped = 0
    _to_zero = []
    for _g in range(_num_groups):
        _lo = _g * _n_repeat
        _hi = _lo + _n_repeat
        _group_max = _traj_rewards[_lo:_hi].max()
        if _group_max < 0.05:
            _to_zero.append((_lo, _hi))
            _dropped += 1
    if _dropped / max(_num_groups, 1) > 0.5:
        print(f"[QualityFilter] SKIPPED: {_dropped}/{_num_groups} too many to drop (>50%), preserving signal", flush=True)
        _dropped = 0
    else:
        for _lo, _hi in _to_zero:
            batch.batch["token_level_rewards"][_lo:_hi] = 0
        print(f"[QualityFilter] race-to-bottom: {_dropped}/{_num_groups} groups dropped (max_reward<0.05)", flush=True)
    metrics["filter/race_to_bottom_groups"] = _dropped
    metrics["filter/total_groups"] = _num_groups
    metrics["filter/race_to_bottom_ratio"] = _dropped / max(_num_groups, 1)
except Exception as _e:
    print(f"[QualityFilter] WARN: filter failed: {_e}", flush=True)
```

### 4.2 Patch 验证 checklist

跑起来后看 log 应该出现：
- `[RolloutFilter] dropping empty trajectory ...`（rollouter 侧主防御生效）
- `[ZeroMaskFix] forced 1 valid token on N all-empty rows`（兜底，应该越来越少）
- `[QualityFilter] race-to-bottom: X/Y groups dropped`（GRPO 信号过滤）
- 没有 `param_version_diff` TypeError、没有 MessageQueue PENDING 死锁

### Data schema：dataset → agent_loop → reward 链路

```
train.parquet row:
  prompt: <user prompt>
  extra_info:
    task_id         ★ 必须
    task_name
    category
    grading_type
    timeout
    repeat_idx
    workspace_files  ← ★ 目前 dataset 没有，agent_loop 内部 fallback 解析（3.2 fix）
                       ★ 建议下次重生成 parquet 时正式写入

↓ veRL 调 jiuwenclaw_agent_loop.run()

JiuwenClawAgentLoop output (AgentLoopOutput):
  prompt_ids, response_ids, response_mask     # 训练用
  extra_fields:                               # reward function 用
    session_id
    task_id
    status                                    # success / agent_timeout / parser_error
    timed_out: bool
    history_path: str                         # jiuwenclaw 的 history.json 位置
    workspace_path: str                       # 训推一致：reward 读这个目录
    transcript: list                          # 已加工的 conversation
    stack_idx: int

↓ verl reward_manager 合并 extra_fields → extra_info

rewards/meeting_reward.py:compute_score(
    data_source, solution_str, ground_truth,
    extra_info={ ..., workspace_path, transcript, ... }
) → float
    └── 读 workspace_path 下文件 + transcript
        └── 自动检查 (lib_tasks.grade_func) + LLM judge (deepseek)
```

---

## 5. 给 jiuwenclaw 团队同事的建议（按优先级）

### P0 —— 让 trajectory 可靠

1. **严格 fresh-session 模式（最关键，见 §0）**
   - 启动时**不读** `memory.db` / `daily_memory/*.md` / `MEMORY.md`
   - 结束时**不写**任何持久化 state（不 promote skill、不更新 memory）
   - 当前 `enable_memory: False` 只关了 chat 召回，没关 SQLite 写入 / skill 自学 / daily_memory 累积
   - **没有这条，RL 训练在 jiuwenclaw 上永远训不干净**（违反 stationary MDP）

2. **明确的 trajectory status code**
   - 当前 timeout / parser error 都返回空 response，训练侧靠 mask 全 0 来辨认
   - 建议在 WS response 加 `terminal_status` 字段（success / agent_timeout / parser_error / context_overflow / ws_disconnect）

3. **Workspace setup API**
   - `setup_workspace(session_id, files=[(src, dst), ...])` 显式 setup
   - 或 chat.send 支持 attachments 参数

### P1 —— Token / wall time 效率

4. **可选关闭 token-level chunking**
   - RL 训练只关心最终 trajectory，不需要 streaming chunk
   - 用 batch 模式可大幅缩短 history.json，也避开 vLLM hermes parser 异常

5. **Identity 文件 lazy load**
   - 每 session 12 秒的启动开销主要是 identity / memory；RL 训练 setup phase 一次加载，trajectory 启动不重读

### P2 —— 可观测性

6. **暴露内部 timing**
   - 每条 trajectory 返回 `timing_breakdown: {model_gen_ms, tool_exec_ms, framework_overhead_ms}`
   - 让训练侧能定位瓶颈

7. **统一 log 入口**
   - 当前要拼 WS log + agent_server log + tool executor log
   - 至少提供 per-session aggregated log endpoint

### 长期 —— 训推一致的 reward 路径

8. **JiuwenClaw 提供 grading API**，让训练侧能直接调用同一函数（当前 bench 走 pinchbench harness，训练走 reward function 自己读 workspace），架构上 drift 风险长期存在

---

## 6. Summary：为什么悲观是合理的，但路也不是死的

OpenClaw 作为 RL runtime 的胜出不是因为它"更好"，而是它**没有为生产环境优化的负担**：

- 白盒可观测 → debug 容易
- Stateless → 没有跨 session 状态污染
- CLI subprocess → 失败隔离干净
- Trajectory 短 → vLLM context / response_mask 都简单

JiuwenClaw 反过来是为"长 session、有记忆、多通道分发"优化的，**这些恰好是 RL 训练不需要的**。

短期内训得通的最小集，需要 runtime 侧做的最关键的事是 **P0 三项（stateless 模式 / status code / workspace API）**。其余 patch 可以继续放在训练侧，但 P0 不做的话每条 trajectory 都在跟 framework ceremony 较劲，永远训不快也训不稳。

本仓库当前所有 patch 在 `jiuwenclaw-agent-loop-impl` 分支，veRL 侧 patch 在 pod `/root/verl/`，给同事接手时可作 baseline 继续往前推。
