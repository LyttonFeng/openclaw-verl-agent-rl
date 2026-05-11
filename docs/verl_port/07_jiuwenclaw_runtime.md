# 07 — Runtime switch: OpenClaw → JiuwenClaw bench

## Context

`05_first_bench_47pct.md` 和 `06_clean_split_47.8.md` 证明 veRL+OpenClaw+GRPO
能复现 main 分支 47.80% baseline。本文档记录把 **inference runtime 从
OpenClaw 切换到 JiuwenClaw**（同事 fork 已做完 WS-protocol adapter），同样
5 个 held-out test task 上跑 bench，验证 runtime 切换是否影响结果。

## Stack architecture

同事 fork (`gitcode.com/liulili-huawei/pinchbench`) 提供完整 WS-based 集成：

```
client (run_pinchbench_jiuwenclaw.py)
   ↓ ws://127.0.0.1:611/ws  (chat.send + tool_call events)
   │
JiuwenClaw Agent Server (port 611)
   ↓ HTTP /v1/chat/completions
   │
Gateway (port 613)
   ↓ forwards model_id to vLLM
   │
vLLM (port 614)  ← --enable-lora --lora-modules ...
```

跟 OpenClaw 单进程 subprocess 比，JiuwenClaw 是 **常驻服务 + WS protocol**，
session_id 维护，history.json 落盘。Workspace 用 `~/.jiuwenclaw/agent/jiuwenclaw_workspace/`。

## Tier 1: base Qwen3-4B + jiuwenclaw runtime

无 LoRA，直接 stack 起来跑 5 test tasks。

| task | status | score |
|---|---|---|
| `task_meeting_advisory_stakeholders` | timeout | 0.0 |
| `task_meeting_council_votes` | success | 0.0 (agent done but didn't produce expected output) |
| `task_meeting_gov_speaker_summary` | success | **64.2 %** |
| `task_meeting_tech_action_items` | success | **57.5 %** |
| `task_meeting_sentiment_analysis` | success | **87.5 %** |
| **Mean** | | **41.9 %** |

## Tier 2: OpenClaw-trained LoRA (step_8 / step_16) + jiuwenclaw runtime

把 OpenClaw-trained 的 LoRA adapter hot-load 到 jiuwenclaw stack 的 vLLM
（`--lora-modules <name>=<path>`），同 5 个 test tasks 重测。

| ckpt | jiuwenclaw runtime | OpenClaw runtime（参考 06 doc）| Δ runtime |
|---|---|---|---|
| **base 4B** (no LoRA) | **41.9 %** | (not measured) | — |
| step_8 LoRA | **10.2 %** | 47.7 % (leak) | **−37.5 pp** ⬇⬇ |
| step_16 LoRA | **27.7 %** | **47.8 %** (clean) | **−20.1 pp** ⬇ |

**LoRA 在 jiuwenclaw runtime 下不仅没涨，反而比 base 4B 还差**（step_8: -31.7pp，step_16: -14.2pp）。

### step_16 per-task breakdown

| task | base 4B (Tier 1) | step_16 LoRA |
|---|---|---|
| advisory_stakeholders | 0.0% (timeout) | 27.5% ⬆ |
| council_votes | 0.0% (no output) | 0.0% (error) |
| gov_speaker_summary | 64.2% | 64.2% |
| tech_action_items | 57.5% | 46.7% ⬇ |
| sentiment_analysis | 87.5% | **0.0%** (timeout) ⬇⬇ |
| **Mean** | **41.9%** | **27.7%** |

## 实际 root cause 诊断（看 transcript）

从 step_16 sentiment_analysis（最显著的退化：87.5% → 0%）transcript 对比：

**base 4B (turn 5)**：
```
[CALL write_file args={'file_path': '/root/.jiuwenclaw/.../sentiment_analysis.md', ...}]
→ toolResult: success=True (bytes_written=3452, created=True)
```

**step_16 LoRA (turn 5+)**：
```
[CALL write_memory args={'path': '/root/.jiuwenclaw/.../sentiment_analysis.md', ...}]
→ toolResult: success=False, error="Invalid path: directory traversal not allowed"
[CALL write_memory args={...same...}]  ← LoRA 死循环重试同样工具
→ error: ...
[CALL write_memory ...]
→ error: ...
（直到 timeout）
```

**Root cause: tool-name confusion**——LoRA RL 训练时 OpenClaw 暴露的工具
集跟 jiuwenclaw 的不一样。

- **OpenClaw**：文件操作走 `write_file` 之类的通用工具
- **JiuwenClaw**：暴露 `write_file` AND `write_memory`。`write_memory` 是
  jiuwen 专用的 memory subsystem，路径受限只能写 `memory/` 子目录

LoRA RL fine-tuning 把 policy 推向某种"偏好"工具，碰巧 jiuwenclaw 里有同名
工具但语义截然不同。LoRA 选了 `write_memory`（路径校验失败），陷入死循环，
直到超时。

base 4B 用 plain Qwen3-4B 的判断力还能选对 `write_file`；越训练，LoRA 越
"自信"地选错。

### Implication

**OpenClaw-trained LoRA 不能直接 deploy 到 jiuwenclaw runtime**。需要：

1. 要么 **训练时就在 jiuwenclaw runtime 下**——用 jiuwenclaw 的真实工具集
   做 rollout，LoRA 学到的就是 jiuwenclaw-compatible policy（这就是 Tier 3-4
   `JiuwenClawAgentLoop` 集成的真正动机）
2. 要么改 jiuwenclaw 的 system prompt 强制屏蔽 `write_memory`、只暴露
   `write_file`——但这是 runtime 配置 hack，治标不治本
3. 要么 **fine-tune LoRA 时同时混合两个 runtime 的样本**——agent 学到更
   robust 的工具选择

## 实验数字总结

| Runtime | Base 4B | step_8 LoRA | step_16 LoRA |
|---|---|---|---|
| **OpenClaw** | (not measured) | 47.7 % (leak) | **47.8 %** clean baseline |
| **JiuwenClaw** | **41.9 %** | **10.2 %** | **27.7 %** |
| Δ (jw − openclaw) | — | −37.5 pp | −20.1 pp |

## 踩坑记录

### 1. agent-core 依赖

stack 启动脚本 hardcode 引用 `${REFACTOR_ROOT}/agent-core/examples/jiuwenrl_online/run_online_rl.py`，但 pod 上只有 jiuwenclaw。
解决：clone `https://gitcode.com/openJiuwen/agent-core.git --branch develop`

### 2. omegaconf 缺包

`uv pip install omegaconf` 补齐

### 3. redis 缺

stack 用 `--redis-url` 必填。`apt-get install -y redis` + `redis-server --daemonize yes`

### 4. vLLM 不在 venv

jiuwenclaw venv 里没装 vllm，但 `/usr/local/lib/python3.12/dist-packages/vllm` 已有。
解决：patch `services.py` 把 `sys.executable` 改成 `/usr/bin/python3` 跑 vLLM subprocess

### 5. 缺多个 launcher CLI args

`--judge-port`、`--redis-url`、`--jiuwen-agent-server-port` 都要设置，validator 严格

### 6. DeepSeek judge backend

colleague fork 的 `call_judge_api` 默认 OpenRouter，不支持 DeepSeek 直连。
我们 main repo 的 `scripts/lib_agent.py`（runner 实际加载这份）也只有
custom base_url 路径，没默认 DeepSeek 分支。
解决：patch main repo 的 `call_judge_api` 增加 `deepseek-chat` / `deepseek/*` 分支，
auto-route 到 `https://api.deepseek.com/v1/chat/completions` + `DEEPSEEK_API_KEY`

## 复现命令

### Stack startup
```bash
bash /root/jiuwen_work/start_jw_pod.sh
```

### Tier 1 bench (base 4B)
```bash
bash /root/jiuwen_work/run_jw_bench_base.sh
```

### Tier 2 bench (LoRA hot-loaded)
```bash
# Restart stack with LoRA as primary model
bash /root/jiuwen_work/restart_stack_with_lora.sh step8-lora /workspace/verl_port/ckpt_openclaw/global_step_8/actor/lora_adapter
# Then bench
bash /root/jiuwen_work/run_jw_bench_base.sh   # bench uses MODEL_NAME from stack meta
```

## Architecture decision: 训练侧 JiuwenClawAgentLoop

要把 RL 训练 rollout 从 OpenClaw 切到 JiuwenClaw，需要：
- 写 `JiuwenClawAgentLoop(AgentLoopBase)` 替代现有 `OpenClawAgentLoop`
- 把 colleague 的 `run_pinchbench_jiuwenclaw.py::_execute_task` WS protocol 抽进 AgentLoop
- 处理 token mask（每 turn assistant 输出 mask=1，tool 结果 mask=0）
- 训练时 sync LoRA：每个 grad step 后 dump LoRA → jiuwenclaw stack 的 vLLM hot-load

这一步是 **out-of-scope for tonight**（涉及 ModelProxy/AgentLoop 200-400 行新代码 + smoke test）。
Tier 1+2 完成后，下次再做 Tier 3+4。
