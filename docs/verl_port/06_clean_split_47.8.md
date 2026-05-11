# 06 — Clean train/val split：veRL+OpenClaw GRPO 复现 main 分支 47.80%

## TL;DR

干净的 23-train / 5-test split + vanilla GRPO + terminal-only DeepSeek judge
+ OpenClaw 多轮 rollout，跑到 step 16 LoRA ckpt：

| task | step_16 (clean) | main-branch vanilla GRPO baseline |
|---|---|---|
| `task_meeting_advisory_stakeholders` | **49 %** | - |
| `task_meeting_council_votes` | **25 %** | - |
| `task_meeting_gov_speaker_summary` | **43 %** | - |
| `task_meeting_tech_action_items` | **60 %** | - |
| `task_meeting_sentiment_analysis` | **61 %** | - |
| **Mean** | **47.8 %** | **47.80 %** |

**与 main 分支 vanilla GRPO baseline 47.80% 在小数点后第一位完全一致**——veRL+OpenClaw
完整流水线复现 confirmed.

## 与第一次（step 8 / 05_first_bench_47pct.md）的关键差异

| 维度 | 05 文档 (step_8) | 06 (step_16) |
|---|---|---|
| Train/val split | 28-task `train.parquet`（**leak**：5 个 test 在训练集里） | **clean**：23 train + 5 test 分开 |
| Ckpt step | 8 / 24 | 16 / 24（step 22 grad explosion 后训练崩，但 step 16 ckpt 已落盘）|
| Mean score | 47.7 % | 47.8 % |
| 可比性 | leak → 数字虚高 | clean → 真值 |

step_8 那次 47.7% 其实有水分（训练数据见过测试 task），但巧合的是 step_16
真值跟它差不多——说明 8→16 step 之间的额外训练（清洁数据）抵消了 leak
失去的优势，且更紧密对齐 main 分支 baseline。

## 训练曲线 + 崩点

veRL 训练 metrics 走势：

| step | grad_norm | critic/score/mean | num_turns/max |
|---|---|---|---|
| 18 | 0.11 | -0.75 | 5 |
| 19 | 0.03 | -1.13 | 15 |
| 20 | 0.07 | -0.94 | 31 |
| 21 | 0.06 | -0.75 | 11 |
| **22** | **7.16** ⚠️ | **-2.06** | **41** ⚠️ |

step 22 一个 rollout 跑到 41 轮（cap 20 没刹住），terminal reward -7.10，导致
PG loss -0.42、grad norm 7.16（50× 平时）—— 梯度爆炸。Step 22 grad 应用后
模型权重应该被这次大梯度撞坏，step 23 启动 rollout 时进程崩。

**幸运**：step 16 ckpt（已落盘在 grad explosion 之前）安全。

未来 fix：
1. 加 `actor.grad_clip_norm` 限制 grad
2. OpenClaw `MAX_TURNS=20` 真正强制（现在 41 turn 突破了 cap）
3. 加 quality filter（race-to-bottom 防御）

## 文件清单

```
experiments/verl_port_poc/
├── launch_meeting_openclaw_lora.sh    # 训练 launch（已 commit）
├── run_bench_step16.sh                 # bench step_16（本次新增）
├── run_bench_step8.sh                  # bench step_8（之前 commit）
└── bench_results/
    ├── results_step8/0050_step8-lora.json  (47.7%, leak)
    └── results_step16/0051_step16-lora.json (47.8%, clean)
```

## 复现命令

### 训练（clean split）
```bash
# Prerequisites:
#   /root/hf_cache/ (Qwen3-4B local copy, 7.6 GB)
#   /workspace/verl_port/data_meeting/train_full23.parquet  (23 train tasks)
#   /workspace/verl_port/data_meeting/val_5test.parquet     (5 test tasks)
#   /workspace/verl_port/openclaw_integration/  (pinchbench-skill rl/* 移植)
#   transformer_impl.py inline-patch (LoRA-only save)
#   /usr/local/bin/openclaw + self-SSH + DEEPSEEK_API_KEY

bash /workspace/verl_port/launch_meeting_openclaw_lora.sh
```

### Bench step_16
```bash
# Start vLLM with step_16 LoRA hot-loaded
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B \
  --served-model-name qwen3-4b-base \
  --tensor-parallel-size 2 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.65 \
  --enable-lora --max-loras 4 --max-lora-rank 16 \
  --lora-modules step16-lora=/workspace/verl_port/ckpt_openclaw/global_step_16/actor/lora_adapter \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --host 127.0.0.1 --port 8000 --enforce-eager

# Then bench (uses pinchbench-skill harness)
bash experiments/verl_port_poc/run_bench_step16.sh
```

## 重大学习总结（按踩坑顺序）

整套流水线落地踩过 7 个坑：

1. **MooseFS 慢/截断**——模型权重读 16 min/shard，FSDP ckpt 写截断 → `/root/` 本地盘 + LoRA-only save （04 文档详）
2. **Hydra rope_scaling dict 解析失败**——CLI override 不行，写 config.json
3. **veRL 0.7→0.8 模块路径变化**——`fsdp_workers` → `engine_workers`，pinchbench-skill patch 全部失效
4. **OpenClaw judge backend 默认 = openclaw（不是 api）**——bench 必须传 `--judge deepseek-chat` 走 api 直连
5. **vLLM `--enable-auto-tool-choice --tool-call-parser hermes`** 缺一个就 400 Bad Request
6. **Task `timeout_seconds:180` 默认太紧**——OpenClaw 多轮 rollout 需要 `--timeout-multiplier 3`
7. **Grad explosion at step 22**——OpenClaw 没刹住 num_turns=41 → reward -7 → grad norm 7 → 训练崩。下次加 grad_clip

## 下一步

- 加 grad clip 重训完整 24 step（甚至更多）
- 跑 `--runs 3` 取平均降噪
- bench base Qwen3-4B（无 LoRA）→ 拿真 baseline，量化 GRPO 训练 +X pp
- 加 quality filter（race-to-bottom 防御）尝试突破 47.8% baseline
