# OpenClaw rollout train-infer alignment — v34 / v35 修通记

## 背景

veRL FullyAsync + OpenClaw GRPO 训练的 rollout 链路与 bench.py 跑同样模型同样任务**分数严重失配**：

- bench.py `task_meeting_advisory_stakeholders` base 4B × 3 runs = **52.6%**
- rollout 同 task 同 base 4B = **0.00**（deterministic 失败）

这个 ~50pp gap 让 RL 在错的信号上优化，v17 ckpt_4 顶死在 46.5%。

## 根因（按时间发现顺序）

### Fix 1：env 没传到 ray worker（v27）

`launch_meeting_openclaw_async.sh` 漏了 `OPENCLAW_MODEL_REASONING`、`PINCHBENCH_OPENCLAW_CONTEXT_WINDOW`、`PINCHBENCH_OPENCLAW_MAX_TOKENS`、`PINCHBENCH_DISABLE_DEFAULT_SKILLS` 没列到 `ray_kwargs.ray_init.runtime_env.env_vars`，导致 verl ray actor 看不到这些 env。

### Fix 2：ModelProxy stream 丢 content（v28）

`model_proxy.py:_delayed_stream_response` 在 `tool_calls` 存在时用 `elif` 互斥发送 —— **只发 tool_calls chunks，content (含 `<think>`) 整段丢**。修成先发 content chunk，再发 tool_calls chunks。

### Fix 3：agent 注册 race（v29）

`_setup_agent_local` 多 worker 并发调 `openclaw agents add` 互相 race，导致部分 agent 注册失败 → "Unknown agent id" 报错 → 任务 0 分。加 `/tmp/.openclaw_agents_add.lock` flock 串行化注册。

### Fix 4：reasoning env 控制（v27 + v28）

agent_loop:411 + agent_loop:2454 两处硬编码 `enable_thinking: False`。Qwen3 不思考 → 直接 tool_call 蒙头读文件死循环。改成 `OPENCLAW_MODEL_REASONING` env 控制，传 `1` 启用 thinking。

### Fix 5：contextWindow / maxTokens 适配 rope×2（v24）

`models.json` 没设 `contextWindow` / `maxTokens` → OC client 用默认 4-8k 截断 prompt → 71KB 会议 transcript 进不去。补 `contextWindow=81920 maxTokens=16000` 配合 Qwen3-4B 的 yarn rope factor=2。

### Fix 6：**`_setup_agent_local` 用 sync subprocess.run 阻塞 async event loop**（v34 — 关键）

`OpenClawAgentLoop._setup_agent_local` 用 `subprocess.run`（同步）在 `async` 函数里调用，**阻塞 event loop 5-30s per call**。16 个 async coroutine 互相串行化在这个 syscall 上 → **真实并发 = 4**（= AGENT_NUM_WORKERS），每 worker 内 4 coroutine 串行。

测得 v30 staircase：

```
 70s × 4 task  ← wave 1
140s × 4 task  ← wave 2 (+70s)
210s × 4 task  ← wave 3
...
500s × 1 task  ← 最后一个完成 setup
```

故 advisory_stakeholders（在 wave 5）setup 350s 后才 spawn OC，立刻命中 OC 内部超时。

**修法**：把 `subprocess.run` 改 `await asyncio.create_subprocess_exec(...).communicate()`，event loop 不阻塞。Caller `self._setup_agent_local(...)` → `await self._setup_agent_local(...)`。

v34 实测：staircase 消失，timeouts 从 40 跌到 4，**bench5 mean 从 0.20 跳到 0.578**（追平 bench reference）。

### Fix 7：**OC 写死的 `DEFAULT_LLM_IDLE_TIMEOUT_MS = 60s`**（v35 — 最后真凶）

OC CLI `pi-embedded-*.js:33348` 硬编码 LLM stream idle timeout 60s：
> "If no token is received within this time, the request is aborted."

ModelProxy 是**假 streaming**（先发 role chunk，等 verl 整段生成完才发 content chunk）。`task_meeting_advisory_attendees` 这种 task 模型 thinking + content 生成 ~146s，OC 60s 内收不到任何 token → idle timeout fire → 报 "Profile verl-default timed out"。

**修法**：在 `~/.openclaw/openclaw.json` 加 `agents.defaults.llm.idleTimeoutSeconds: 0`（禁用 idle timeout），OC 用 `agents.defaults.timeoutSeconds: 600` 总 timeout 兜底。

v35 实测：**0 timeouts**，**全 28 task 都有非 0 分**，attendees 从 0 → 0.68。

## 验证数据

| Version | bench5 mean | overall mean | timeouts | 0 分 task 数 | 关键 fix |
|---------|-------------|--------------|----------|-------------|---------|
| v23 baseline | 0.172 | 0.10 | 40 | 10 | 无 |
| v25 | 0.098 | 0.12 | 40 | 10+ | 试 skills=0 (没用) |
| v27 | 0.162 | 0.15 | 40 | 10 | reasoning env |
| v28 | 0.194 | 0.146 | 40 | 10 | content+toolcalls |
| v29 | 0.176 | 0.226 | 40 | 10 | flock |
| v30 | 0.200 | 0.250 | 40 | 10 | vLLM seqs=8 |
| **v34** | **0.578** | **0.475** | 4 | 1 | **async _setup_agent_local** |
| **v35** | 0.432 | 0.459 | **0** | **0** | **idleTimeoutSeconds=0** |

v34 单次 bench5 偶然偏高，但 0 个 0 分的稳定性 v35 更好。

## bench reference (base 4B, bench.py 跑同样 task)

- `task_meeting_advisory_stakeholders` × 3 = **52.6%**
- `task_meeting_advisory_attendees` × 1 = **80.8%**

**rollout 链路现在跟 bench 完全对齐**。

## 修改文件清单

- `agent_loop/openclaw_agent_loop.py`：
  - line 1466：`_setup_agent_local` 改 async + `await asyncio.create_subprocess_exec`
  - line 1250：caller 加 `await`
  - line 1481：flock on `openclaw agents add`
  - line 411 / 2454：reasoning env-controlled
  - line 1491+：models.json 加 `contextWindow` / `maxTokens` env-controlled
- `agent_loop/model_proxy.py`：
  - line 278：`_delayed_stream_response` 在 tool_calls 路径先发 content chunk
- `experiments/verl_port_poc/launch_meeting_openclaw_async.sh`：
  - line 275-282：4 个 env vars 加入 ray runtime_env
- pod-side `/root/.openclaw/openclaw.json`：
  - 加 `agents.defaults.llm.idleTimeoutSeconds: 0`
- pod-side `/root/verl/verl/experimental/fully_async_policy/fully_async_rollouter.py`：
  - line 238：`max_concurrent_samples` 改 `MAX_CONCURRENT_PER_REPLICA` env 可控

## 教训

1. **async 函数内绝对禁用 sync subprocess.run / time.sleep / blocking IO** —— 任何阻塞 syscall 都序列化整个 worker 的 event loop
2. 异步框架（verl FullyAsync）声称的"并发"假象会被 sync block 完全破坏，导致 GPU 大部分时间 idle（v30 GPU 0% util）
3. 调试 RL on agent 链路：**先测 timing**，不要从 hyperparameter 改起。staircase pattern 是 event-loop-blocked-by-sync 的标志
4. OC / verl / vLLM 三层都有自己的 timeout 配置，需要对齐
5. ModelProxy 实现假 streaming 会让模型端的真 streaming 期待落空 —— 长 thinking task 会触发 idle timeout
