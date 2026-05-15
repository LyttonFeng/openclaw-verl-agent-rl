# Handoff: jiuwenclaw + veRL async RL debug (2026-05-15)

## 你接的盘是啥

我们想用 **veRL FullyAsyncTrainer + jiuwenclaw runtime** 训一个 Qwen3-4B LoRA，
在 5 道 meeting analysis test tasks 上 bench，目标 > base 44.68% / 追上
OpenClaw 同 setting 的 47.80%。

**现状**：训练能跑通了（v57 出了 3 个 ckpt），但 **1 个 GRPO step 后模型崩盘**：

| Metric | step 1 | step 4 |
|---|---|---|
| critic/score/mean | 0.166 | **0.047 (-71%)** |
| trajectory empty 率 | ~0% | **80%+** |

**v57 ckpt_2 bench 实测：5/5 task 0.00%**。

**❗❗ 致命发现（必看 `v57_trajectory_forensic.md`）**：
- bench 5/5 task 模型 **0 tool call**（base 模型 5-15 个）
- 模型**幻觉**自己调过 `list_files` / `glob`（实际从没调过）
- 输出模板化 "我找不到文件，请确认 1/2/3..." 给 user 让 user 自己解决

**所以问题不是 trajectory timeout / hermes parser / 工具调用格式错** — 是
**模型主动选择不调工具**。GRPO 单步 + 3 effective signal trajectory 把 LoRA
打成"只 think 不 tool"的脑残模式。

读这 3 份文档（顺序）：
1. `docs/v57_trajectory_forensic.md` — 5/5 bench task 详细取证 + 失败 pattern
2. `docs/jiuwenclaw_verl_async_debrief.md` — 完整诊断 §3 根因
3. 本文 — 怎么接盘

---

## 你的任务

让 jiuwenclaw + veRL async GRPO 训出**至少不退化**的 LoRA（bench ≥ 44.68% base）。
理想 ≥ 47.80%（追上 OpenClaw）。

**不要再调工程**（debrief §5 P0 已确认走不通）。**调 RL 算法层**（§5 P1）。

---

## 一句话根因

LoRA rank=32 × 6 modules × GRPO 单步 update，在**有效 signal trajectory 只有
3 条**（8 sample - 2 ZeroMaskFix - 3 race-to-bottom = 3）的情况下，把 model
tool-call 决策路径打坏。从此 model 输出 jiuwenclaw hermes parser 无法解析的
tool call → trajectory timeout → 下一 batch signal 更少 → 死循环。

→ **必须降 noise / 降 update size**。

---

## 优先建议尝试的 4 个 knob

按收益/风险排序：

### 1. lr 降 10×（最稳）
- 现在: `LR=2e-6`
- 试: **`LR=2e-7`** 或 `2e-8`
- 原理：单步 update 更小，模型不被一击打坏，需要更多 step 才学到东西
- 代价：收敛慢 10×（jiuwenclaw 已经慢，单 step ~30 min，可能需要 50+ step 才看趋势）

### 2. KL coef 加 10×（强约束）
- 现在: `actor_rollout_ref.actor.kl_loss_coef=0.01`（hard-coded 在 launcher line ~140）
- 试: **`kl_loss_coef=0.1`** 甚至 `0.5`
- 原理：强约束 policy 不远离 base model，本质降 LoRA 影响半径
- 代价：学得慢，可能根本学不到东西
- 风险：低，建议跟 #1 一起试

### 3. Batch 加大（提高 SNR）
- 现在: `REQUIRE_BATCHES=2 × BATCH_SIZE=4 = 8 samples/step`
- 试: **`REQUIRE_BATCHES=4-8` → 16-32 samples/step**
- 原理：更多 trajectory 平均掉噪声（race-to-bottom + ZeroMaskFix 后真有效 ≥ 8-15）
- 代价：单 step 时间 ×2-4（已经 30 min，可能变 60-120 min）
- 风险：jiuwenclaw 慢 + 16+ 并发 trajectory 可能撞奇怪问题

### 4. LoRA rank 减半（小容量）
- 现在: `LORA_RANK=32 LORA_ALPHA=64`
- 试: **`LORA_RANK=8 LORA_ALPHA=16`** （保持 alpha/rank=2）
- 原理：可训参数减 4×，单步 update magnitude 物理上小
- 代价：long-run 收敛上限可能也低（但 1-step 不崩比上限重要）

**推荐组合**：**先试 lr=2e-7 + kl_coef=0.1**（只动 2 个 knob，最容易归因）。
跑出 step 4-8 看 critic/score 趋势：
- 持续涨/平 → 这条路通
- 仍然退化 → 试 #3 或 #4
- 退化更快 → 整个 jiuwenclaw + GRPO 范式可能根本不行（fallback：debrief §5 P2）

---

## 起步命令（v57 baseline，你的起点）

```bash
ssh -p 17949 -i ~/.ssh/id_ed25519 root@154.54.102.52

# 在 pod 上 — clean state launcher 自带 pre-flight
TOTAL_EPOCHS=10 \
TOTAL_TRAINING_STEPS=48 \   # 加大上限，让你看趋势
AGENT_TIMEOUT=600 \
JW_N_STACKS=2 \
VLLM_MAX_MODEL_LEN=65536 \
VLLM_MAX_NUM_SEQS=4 \
BATCH_SIZE=4 \
REQUIRE_BATCHES=2 \         # 改这个调 batch (P1 #3)
ROLLOUT_N=2 \
MAX_TURNS=8 \
MAX_PROMPT_LENGTH=25000 \
MAX_RESPONSE_LENGTH=6000 \
ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=40000 \
SAVE_FREQ=1 \
TRIGGER_PARAM_SYNC_STEP=2 \
TEST_FREQ=-1 \
USE_RL_ONLINE_RAIL=1 \
MEMORY_ENABLED=false \
RAIL_V1_WAIT_S=60 \
LR=2e-7 \                    # ← P1 #1 调这个 (默认 2e-6)
EXPERIMENT_NAME=v58_lr_low \
bash /workspace/openclaw-verl-agent-rl/experiments/verl_port_poc/launch_meeting_jiuwen_async.sh \
> /tmp/jw_async/run_v58.log 2>&1 &
```

⚠️ **LR / KL_LOSS_COEF 目前 launcher 没暴露成 env**，hard-coded 在
`launch_meeting_jiuwen_async.sh:135` (`LR=2e-6`) + `:140` (`kl_loss_coef=0.01`)。
要调这两个先改 launcher 让它们读 env：

```bash
# 在 launch_meeting_jiuwen_async.sh top 加：
LR="${LR:-2e-6}"
KL_LOSS_COEF="${KL_LOSS_COEF:-0.01}"
# 然后替换 hard-coded actor.optim.lr=2e-6 → actor.optim.lr=${LR}
#         kl_loss_coef=0.01 → kl_loss_coef=${KL_LOSS_COEF}
```

---

## 监控关键指标

每 30 min 看一次：

```bash
ssh root@... -p 17949 "VLOG=\$(ls -t /tmp/jw_async/verl_*.log | head -1); \
  echo '=== steps ==='; grep 'global_steps:' \$VLOG | tail; \
  echo '=== ckpt ==='; ls /workspace/verl_port/ckpt_jw_async/; \
  echo '=== critic 趋势 ==='; \
  grep -oE 'critic/score/(mean|max):[0-9.]+|actor/(loss|grad_norm):[0-9.e+-]+|filter/race_to_bottom_ratio:[0-9.]+' \$VLOG | tail -20; \
  echo '=== empty rate ==='; grep -c 'produced empty' \$VLOG; \
  echo '=== fatal ==='; grep -nE 'Traceback' \$VLOG | grep -vE 'TaskCancelled|hermes|context length|serving_chat' | tail -3"
```

**红灯**（停 + 换 knob）：
- critic/score/mean 连续 2 step 下降 > 30%
- empty rate > 60%
- Traceback in 非 hermes/context noise

**绿灯**（继续）：
- critic/score/mean 持平或上升
- empty rate < 40%
- grad_norm < 0.3 不爆

---

## Bench 命令（出 ckpt 后用）

```bash
# 1. Merge FSDP shard → HF LoRA adapter
mkdir -p /workspace/verl_port/bench_vXX/ckpt_N_hf
python3 -m verl.model_merger merge --backend fsdp \
  --local_dir /workspace/verl_port/ckpt_jw_async/global_step_N/actor \
  --target_dir /workspace/verl_port/bench_vXX/ckpt_N_hf
# ⚠️ adapter_config.json lora_alpha 会是 0，patch 回 64：
python3 -c "import json; p='/workspace/verl_port/bench_vXX/ckpt_N_hf/lora_adapter/adapter_config.json'; \
  c=json.load(open(p)); c['lora_alpha']=64; c['task_type']='CAUSAL_LM'; \
  c['base_model_name_or_path']='/root/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c'; \
  json.dump(c, open(p,'w'), indent=2)"

# 2. Start vLLM with LoRA
CUDA_VISIBLE_DEVICES=0 HF_HOME=/root/hf_cache HF_HUB_CACHE=/root/hf_cache/hub \
  nohup vllm serve /root/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/1cfa9a7208912126459214e8b04321603b3df60c \
  --served-model-name Qwen3-4B --port 8000 --gpu-memory-utilization 0.8 \
  --max-model-len 32768 --enable-auto-tool-choice --tool-call-parser hermes \
  --enable-lora --max-loras 1 --max-lora-rank 32 \
  --lora-modules vXX_ckptN=/workspace/verl_port/bench_vXX/ckpt_N_hf/lora_adapter \
  > /tmp/vllm_bench.log 2>&1 &
# wait for /v1/models 200

# 3. Start jiuwenclaw stack pointing at LoRA model
rm -rf /root/.jiuwenclaw_bench /tmp/jw_bench; mkdir -p /tmp/jw_bench
cd /workspace/openclaw-verl-agent-rl/experiments/verl_port_poc && \
  API_BASE=http://127.0.0.1:8000/v1 MODEL_NAME=vXX_ckptN \
  WS_PORT=611 AGENT_SERVER_PORT=18095 GATEWAY_PORT=19095 \
  JIUWENCLAW_DATA_DIR=/root/.jiuwenclaw_bench LOG_DIR=/tmp/jw_bench \
  MEMORY_ENABLED=false USE_RL_ONLINE_RAIL=0 \
  bash ./start_jw_headless.sh

# 4. Run pinchbench
set -a; source /root/.pinchbench_env; set +a
cd /root/jiuwen_work/jiuwenclaw && \
  nohup uv run --frozen python /root/jiuwen_work/pinchbench/scripts/run_pinchbench_jiuwenclaw.py \
    --skill-root /workspace/openclaw-verl-agent-rl --ws-url ws://127.0.0.1:611/ws \
    --suite task_meeting_advisory_stakeholders,task_meeting_council_votes,task_meeting_gov_speaker_summary,task_meeting_tech_action_items,task_meeting_sentiment_analysis \
    --judge-model deepseek-chat --judge-backend api \
    --output-dir /workspace/verl_port/bench_vXX/results > /tmp/jw_bench/pinchbench_run.log 2>&1 &

# 5. results.json 出来后看 summary.overall_pct
```

---

## 已经踩过的坑（不要再踩）

| 现象 | 真因 | Fix（已在 repo） |
|---|---|---|
| MessageQueue actor PENDING 不启动 | `num_cpus=2` 抢不到 | `verl_patches/` patch 1，已应用 |
| `param_version` TypeError None-None | STANDALONE vLLM | patch 2 |
| Empty trajectory 全 batch torch.cat 崩 | response_mask=0 也进 MQ | patch 3 (RolloutFilter) + ZeroMaskFix |
| GRPO 学差样本 (N=2 race-to-bottom) | 单 group 都低分时还学 | QualityFilter (patch 4) |
| FSDP2 LoRA ckpt save 报 No DTensor | LoRA-only mode + FSDP | patch 7 (`transformer_impl.py`) |
| jiuwenclaw chat.send 20s 假死 | 冷启动 25-40s 我们 20s 就 timeout | `jiuwenclaw_agent_loop.py` ws.recv 改 120s/45s |
| MEMORY_ENABLED=False 不生效 | jiuwenclaw 需要三 flag 都开 | chat.send params 三联 `enable_memory:False + group_digital_avatar:True + is_group_chat:True` |
| `daily_memory.md` 5MB 累积启动慢 | jiuwenclaw 自动 memory | start_jw_headless.sh 启动前 wipe |
| RLOnlineRail 没 inject 到 stack | 同事 Rail 是 callback hook，没自动注册 | `sitecustomize.py` + `inject_rl_online_rail.py` (sys.meta_path) |
| Stack 多进程 inject 没生效 | jiuwenclaw spawn subprocess | `sitecustomize.py` 在 PYTHONPATH，所有 python 进程都 import |
| `MAX_TURNS=8` env 没生效 | jiuwenclaw 内部 ReAct config 不读这个 | 仍未 fix（num_turns/max 实测=37）|
| veRL torch.cat prompt_ids 不同 shape | rail 返回原始 39k prompt vs fallback 短 prompt | agent_loop left-truncate 到 max_prompt_length |
| veRL torch.cat response_logprobs None vs Tensor | rail-v1 命中 vs fallback 不一致 | agent_loop 永远返回 `response_logprobs=None` |
| GPU OOM in update_actor | 40k prompt + 6k resp > 32k token cap | MAX_PROMPT_LENGTH=25000 + ACTOR_PPO_MAX_TOKEN_LEN_PER_GPU=40000 |
| pkill 杀自己 SSH | `pkill -f 'jiuwenclaw.app'` 匹配 ssh cmdline | pkill 不再用宽匹配，按显式 PID |
| pkill 杀自己 launcher | `pkill -f 'launch_meeting_jiuwen_async'` 自杀 | launcher 跳过 self PID |
| v53 stale stack 占端口 | 老 jiuwenclaw 没被前一次 pkill 杀干净 | launcher pre-flight 验证端口 + GPU clean，失败 exit 7/8 |

---

## 关键文件 / 目录速查

### 本仓库 `jiuwenclaw-agent-loop-impl` 分支
```
experiments/verl_port_poc/
├── launch_meeting_jiuwen_async.sh    ★ 主 launcher (改这个调 RL knob)
├── start_jw_headless.sh              jiuwenclaw stack 启动 (改 MAX_ITERATIONS / MEMORY_ENABLED)
├── jiuwenclaw_agent_loop.py          ★ veRL 内部用 (left-truncate, rail-v1 数据源)
├── inject_rl_online_rail.py          sys.meta_path 钩 jiuwenclaw 注入 RLOnlineRail
├── sitecustomize.py                  让 jiuwenclaw 子进程都 import 上面
├── run_jw_app_with_rl_rail.py        wrapper exec jiuwenclaw.app via runpy
├── mock_trajectory_gateway.py        aiohttp :9000 接 PerTurnSample JSONL
├── tests/test_jiuwenclaw_agent_loop.py  16/16 unit tests
├── verl_patches/
│   ├── verl_async_for_claw_agentic_rl.patch   7-file unified diff (verl@8c3bee47)
│   └── README.md
└── docs/
    ├── openclaw_vs_jiuwenclaw_for_rl.md       根本架构对比（白盒 vs 黑盒）
    ├── jiuwenclaw_verl_async_debrief.md       ★ 你必读的诊断
    └── handoff_to_codex.md                    本文件
```

### Pod 上需要的（已就位，不要删）
```
/root/verl/                          ★ patches 已应用，commit base 8c3bee47
  - verl/checkpoint_engine/base.py            (modified)
  - verl/experimental/fully_async_policy/*.py (4 files modified)
  - verl/experimental/separation/ray_trainer.py
  - verl/workers/engine/fsdp/transformer_impl.py
/root/jiuwen_work/jiuwenclaw/        jiuwenclaw runtime
/root/jiuwen_work/agent-core/        含同事 RLOnlineRail
/root/jiuwen_work/pinchbench/        bench harness
/root/hf_cache/                      Qwen3-4B 缓存
/root/.pinchbench_env                DEEPSEEK_API_KEY
/workspace/verl_port/data_meeting/   train.parquet (23 tasks) + val_5test.parquet
/workspace/verl_port/bench_v57/      v57 ckpt_2 LoRA + bench results (反向对照)
```

### 训练数据格式（已生成，不用动）
`/workspace/verl_port/data_meeting/train.parquet`：23 个 task_meeting_* 任务，
extra_info 含 `task_id` 但**没有 `workspace_files`**（agent_loop 内部自动从 task
YAML frontmatter load 文件，见 `_load_task_workspace_files`）。

---

## 已知 OPEN 问题（debug 路上可能撞）

1. **`MAX_TURNS=8` 没生效**：实测 trajectory 跑到 37 turns。`MAX_ITERATIONS` env
   传给 jiuwenclaw 但 ReAct config 不用它。jiuwenclaw 同事说 `chat.send` 不接受
   `max_turns` param。**workaround**：在 agent_loop 收到 N turns 后主动断 WS。
   或忽略（让长 trajectory 撞 6k token cap 自然结束）。

2. **rail-v1 数据有时找不到 session_id 匹配**：trajectory 完成 → rail upload
   异步 → agent_loop 等 60s 还找不到匹配 JSONL → fallback 到 history.json。
   实测 ~30-50% rail miss rate。**不致命**（fallback 路径也 work），但意味着
   prompt_ids 一会儿是 rail 原始的、一会儿是 chat-template 的。**可能后续撞坑**
   （目前 left-truncate 兜底，但训推一致打折）。

3. **vLLM `hermes_tool_parser.py:417 AssertionError`**：streaming tool call
   解析错误，模型输出 partial JSON 解析失败。**v50/v52/v57 都有**，单条 trajectory
   级别 noise，不致命。

4. **/tmp 满**：训练日志 + ray + jiuwenclaw 写 7+GB 在 /tmp（30GB overlay 满了
   会卡死）。定期 `rm -rf /tmp/jw_async/jw_2026* /tmp/jw_async/verl_*.log` 清。
   ckpt 进 `/workspace`（networked fs，足够大）。

5. **`actor/use_rollout_log_probs=True`**：launcher 里设了 True，但我们
   `response_logprobs=None`，veRL 自己重算。换 False 可能更快（少一次 forward）
   但没测过会不会破啥。

---

## Bench 当前 baseline 表

| 配置 | bench 5-task | 来源 |
|---|---|---|
| Base Qwen3-4B | **44.68%** | `main` 分支历史数据 |
| OpenClaw + GRPO (R4' clean chain) | **47.80%** | `main` 分支，已 bench |
| OpenClaw + verl sync GRPO step 16 | **47.80%** | `experiment/verl-port` 分支，已 bench |
| **v57 jiuwenclaw + verl async ckpt_2** | **0.00%** | 你接手的失败案例 |

→ 你的目标：训出 ≥ 44.68% 的 ckpt（不退化就行）。能 ≥ 47.80% 就是新 SOTA。

---

## 心态准备

这条路 4 天踩了 26 个 bug 才让训练能跑通。**不要重新走工程坑**，所有工程层 fix
都在 repo 里了。**调 RL 算法层**才有进展可能。

如果调 5-10 个组合都崩，说明 jiuwenclaw + GRPO + LoRA 这条范式根本不工作 ——
那就**结论性失败**，落进 debrief，建议团队改 jiuwenclaw runtime（debrief §5 P2）
或者放弃这条路、回 OpenClaw。

祝好运。

—— 上一任（claude opus 4.7）
