# openclaw_async v2 forensic (2026-05-15)

跑 `launch_meeting_openclaw_async.sh` v2 (`fff03ad..998e830`)，跑了 20 分钟，0 trainer step。
**Generation 链路工作，OpenClaw 卡在 turn 0 → turn 1 转换上**。

## 现象

| 指标 | 计数 |
|---|---|
| OpenClaw subprocess started | 90+ |
| ModelProxy `Got proxy request turn=0` | 94 |
| `Calling server_manager.generate` | 34 |
| `Generate done` | 34 |
| `Turn 0: tool_calls=True, finish=tool_calls, content_len=0` | 100+ |
| `Got proxy request turn=1` | **0** |
| Reward computed | 80 (turns=0, terminal=0.0) |
| total_generated_samples in MQ | 0 |
| **Bench score可比性** | 完全 0 |

## Generation 工作正常证据

- agent_loop 收到 OpenClaw 的 HTTP request (94 次)
- agent_loop 调 `self.server_manager.generate(...)` (34 次)
- 每次 generate 返回 36-50 token（少数 466、677）
- **`tool_calls=True, finish=tool_calls, content_len=0`** —— vLLM 真的 emit 了 tool_call (`<tool_call>{...}</tool_call>`)，因为 hermes parser 把 tool_call 抽到结构化字段，content 留空

## OpenClaw 不前进的猜测

每个 trajectory 在 turn 0 拿到 tool_call 后死，**没看到 `Got proxy request turn=1`**。OpenClaw subprocess 大概率：
- 执行 tool 失败（文件路径、SSH loopback 配置）
- 或 ModelProxy 发回的 OpenAI 格式跟 OpenClaw 期望对不上
- 或 OpenClaw 内部超时（虽然我们已经 patch 了 `agents.defaults.timeoutSeconds=600`）

## 已 patch 修复的东西

1. **Hydra output to MFS EIO**: 加 `hydra.run.dir=$LOG_DIR/hydra_$TS` 重定向到 overlay
2. **OpenClaw internal timeout**: 改 `/root/.openclaw/openclaw.json` 加 `agents.defaults.timeoutSeconds=600`（从默认 ~60s）
3. **Stale wrapper PIDs**: launcher pre-flight 增加 `launch_meeting_jiuwen_async` 模式

## 修不了但已确诊

- vLLM standalone async 链路工作
- veRL server_manager.generate 工作
- tool_call_parser=hermes 工作 (能产 tool_calls 字段)
- judge `localhost:9090` warnings 是 PRM check noise，不致命（terminal score 来自 DeepSeek，但**因为 OpenClaw 没真完成任何任务，terminal=0 是合理输出**）

## 下次 v3 优先 debug 项

1. **看 OpenClaw subprocess stderr 完整内容**（不是只看 timeout 那几行），找 tool exec 失败的 stack
2. **dump ModelProxy 发回的 HTTP response**（加日志 print `response_text`, `tool_calls` JSON）
3. **测试单条 OpenClaw subprocess 手工跑**，绕开 veRL，看 base model 跟 OpenClaw 工作流是否本身正常
4. **对比 sync openclaw_lora.sh** 跑同 task 时 turn 0 → turn 1 的 ModelProxy 日志

如果 sync 也是 only turn 0，那是 OpenClaw setup 问题不是 async 框架问题；如果 sync 能进 turn 1+，那 async server_manager.generate 返回格式跟 hybrid engine 不一样，需要补 patch。

## v2 时间 / GPU 成本

- 17 min wall time
- 0 useful output
- 0 ckpt

下次先做不带 verl 的 OpenClaw smoke（直接 SSH 到 localhost 调 openclaw agent + 简单 prompt + 看完整 stderr）。
