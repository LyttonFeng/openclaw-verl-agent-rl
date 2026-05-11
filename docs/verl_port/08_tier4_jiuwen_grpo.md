# 08 — Tier 4: GRPO with JiuwenClaw rollouts (blocked on GPU/proxy)

## 目标

把 RL 训练 rollout 从 OpenClaw 切到 JiuwenClaw，看能不能修跨 runtime
LoRA 退化（07 doc 测到 step_16 LoRA 在 jiuwenclaw runtime 27.7% vs
OpenClaw runtime 47.8%，原因是 OpenClaw-trained policy 学了 `write_memory`
+ 绝对路径风格，在 jiuwenclaw 工具集下崩盘）。

预期：用 jiuwenclaw rollout 训出的 LoRA，在 jiuwenclaw runtime 下应能
回到 ~47.8%，并且**不会**再触发 write_memory 死循环。

## 当前完成

| 件 | 状态 |
|---|---|
| `JiuwenClawAgentLoop` 全实现（WS roundtrip + response_mask） | ✅ |
| `agent_loop/config.yaml` 注册 `jiuwenclaw_agent` | ✅ |
| Workspace 隔离（class-level asyncio.Lock 串行共享 workspace dir） | ✅ |
| `jiuwen_lora_sync.py` 模块 + CLI（dump → POST /v1/load_lora_adapter） | ✅ |
| 单测 16 个 PASS（pure-python，无 GPU） | ✅ |
| 端到端 smoke（pod 实跑 WS + history.json + mask） | ✅ |
| `launch_meeting_jiuwen_lora.sh` 训练脚本 | ✅（含 pre-flight） |
| **真正起一次 Tier 4 训练并产出 LoRA** | ❌ blocked |
| `jiuwenclaw_agent` 通过 ModelProxy 用 veRL vLLM 的桥（Path A） | ❌ not built |

## Blocker：GPU resource 死局

pod 是 2× A100-80GB。要让 jiuwenclaw rollout 的 token 真正反映"当前正在
训练的 policy"，jiuwenclaw 必须用 **veRL 的 vLLM**（veRL hybrid engine
每个 grad step 把 FSDP weights 同步给自己的 vLLM）。但是：

- jiuwenclaw stack 默认起自己的 vLLM（TP=2，吃满两张卡）
- veRL hybrid engine 也要起自己的 vLLM（吃满两张卡）
- 两个 vLLM 同时在 2× A100 上必 OOM —— 2026-05-11 v3 OpenClaw GRPO 跑到
  step 11 被 kernel kill 就是这个原因（残留 jiuwenclaw stack 没清掉）

## 两条解路

### Path A（推荐，长期正确）

jiuwenclaw stack 跑**无 vLLM 模式**：gateway 把 `/v1/chat/completions`
HTTP 请求转发到 veRL 内部 vLLM 的 HTTP endpoint。

需要做的工程：
1. 在 veRL `AgentLoopWorker` 里跑一个 aiohttp 服务，把
   `server_manager.generate(...)` 包成 `/v1/chat/completions` 接口
2. jiuwenclaw stack launcher 加 `--vllm-base-url` 参数（或 `NO_LOCAL_VLLM=1`），
   让它的 gateway 不起本地 vLLM，转发到给定 base url
3. 启动顺序：veRL 先起来 → 暴露 vLLM proxy 端口 → 再起 jiuwenclaw（用
   veRL 端口）

权重永远最新（veRL hybrid engine 已经管这事），**不需要** `jiuwen_lora_sync.py`。

工作量估计：200-400 行（veRL ModelProxy + 几个 stack launcher 改动）。

### Path B（短期取巧）

保留 jiuwenclaw 自己的 vLLM，每个 grad step 后 dump LoRA → POST
`/v1/load_lora_adapter` 同步到 jiuwenclaw vLLM。

`jiuwen_lora_sync.py` 已就绪，问题是 **GPU 不够分**：

| 方案 | jiuwenclaw vLLM GPU | veRL GPU | 可行？ |
|---|---|---|---|
| 共用 2 卡 | 0,1 (TP=2) | 0,1 | ❌ OOM（已验证） |
| 分开 | 0 (TP=1) | 1 (TP=1, FSDP unsharded) | ❌ Qwen3-4B+16k+ref+optim 单卡 80GB 不够 |
| 4 卡 pod | 0,1 (TP=2) | 2,3 (TP=2) | ✅ 但需要换 pod |

权重不是每 token 最新（每 grad step 才同步一次），on-policy 严格性差一点，但 GRPO 容忍度还行。

## 下次起手清单（Path A 落地）

1. 在 `experiments/verl_port_poc/` 加 `verl_vllm_proxy.py`：
   - aiohttp server 暴露 `/v1/chat/completions`
   - handler 转 `server_manager.generate(prompt_ids=..., sampling_params=...)`
   - 进程内绑定，跟 veRL 一起跑
2. 在 `jiuwen_work/start_jw_pod.sh` 加 `--vllm-base-url`（同事 fork 上游
   PR 或 patch）
3. 改 `launch_meeting_jiuwen_lora.sh`：先 start vLLM proxy port → 起 jiuwenclaw stack（无 vLLM 模式）→ 跑 veRL

## 立即可做的小事（不需要 GPU）

- 跑 `python3 -m unittest experiments.verl_port_poc.tests` 看 16/16 测试
- 用 `jiuwen_lora_sync.py` CLI 手动同步现有 ckpt 到运行中的 jiuwenclaw vLLM：
  ```
  python3 experiments/verl_port_poc/jiuwen_lora_sync.py \
    --ckpt-dir /workspace/verl_port/ckpt_openclaw/global_step_16/actor/lora_adapter \
    --lora-name step16-lora
  ```
- review `JiuwenClawAgentLoop` 的 response_mask 逻辑（cumulative
  `apply_chat_template` diff），看是不是符合训练侧 token-mask 的预期

## 已验证的实事

- WS+history.json+response_mask 端到端 smoke 在 pod 上 PASS：22 prompt
  + 700 response tokens（637 mask=1 / 63 mask=0），num_turns=2
- vLLM `/v1/load_lora_adapter` 热加载是真的工作：step8-lora + step16-lora
  同时 loaded
- jiuwenclaw stack 起着的时候，veRL 训练 OOM（v3, PID 393686, 2026-05-11）
