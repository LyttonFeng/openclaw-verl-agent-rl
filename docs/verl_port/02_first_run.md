# 02 — 第一次跑通 GSM8K（实操）

## 时间线（2026-05-10/11 一晚）

| 阶段 | 用时 | 状态 |
|---|---|---|
| 安装 flash-attn 预编译 wheel | ~30 min（含磁盘清理踩坑） | ✅ |
| 启动 veRL 训练 | T+0 | — |
| Ray init + dataset filter | ~30s | ✅ |
| Worker init + model load | ~60s | ✅ |
| vLLM warmup + CUDA graph capture | ~60s | ✅ |
| 训练 step 1 启动 | T+360s（~6 min） | ✅ |
| 完成 step 9 | T+600s（4 min 内 9 步） | ✅ |
| **卡在 step 10 ~10 min**（疑似 validation） | T+600s ~ T+1200s | ⚠️ |

## 训练步速度

```
step 1: 42.74s（包含 first batch overhead）
step 2-9: ~25s/step 稳定
ETA full epoch (233 steps): ~1.5 hr
```

跟主分支自定义 250 行训练对比：
- **自定义**：fp32 + 30 sample 训练 + 多卡 device_map="auto"，~9 min full step（不含 rollout/bench）
- **veRL**：bf16 + 32 sample × 233 step，每 step 25s，~1.5 hr full epoch

veRL 的优势在 **rollout 跟训练共享 GPU 时间**（hybrid engine），不需要单独 vLLM
serve；但单 step 速度比自定义慢（~25s vs 我们 ~5s/sample），因为 veRL 一个
step = rollout + log_prob 计算 + advantage + PPO update 全干。

## 关键观察：内存占用

每 GPU 峰值 ~52 GB，分布：
- FSDP shard of Qwen3-4B (bf16): ~4 GB
- vLLM engine (sleep mode): ~9 GB
- vLLM warmup + KV cache: ~30 GB
- Activations + gradients (bf16, dynamic batch): ~10 GB

veRL 比我们自定义 setup 显存使用**更聪明**——通过 **enable_sleep_mode=True** 让
vLLM 在训练期暂停占用，把显存让出来给 FSDP backward/optim step。

## 跟主分支自定义对比

| 维度 | 自定义 250 行 | veRL |
|---|---|---|
| 算法 | 自己写 PPO clip + KL k3 + per-turn loss | 标准 PPO + low_var_kl |
| 显存管理 | hack（device_map="auto"） | 工业级（FSDP + vllm sleep mode） |
| Rollout | OpenClaw agent loop（多轮 tool） | 单轮 vllm generate |
| Reward | DeepSeek LLM judge | rule-based GSM8K（POC）/ reward model |
| 数据格式 | 自定义 graded_trajectories.jsonl | parquet with prompt/reward_model/extra_info |
| 配置量 | ~6 个 CLI flag | ~30 个 Hydra config 项 |
| 启动开销 | ~30s（model load） | ~360s（Ray + workers + vllm warmup） |
| 学习曲线 | 1 天读懂全部 | 1 周读懂主路径，1 月精通 |

## 已确认 POC 目标达成

✅ veRL 0.8.0.dev0 在本 pod (Qwen3-4B + 2x A100-80GB) 跑通  
✅ flash-attn 2.8.3 cu12torch2.8 wheel 兼容  
✅ Ray + FSDP + vLLM hybrid engine 启动正常  
✅ 训练循环执行（连续 9 个 step）  
✅ GRPO advantage estimator 配置生效  

## 已识别的集成 gap（要把 meeting 任务接进 veRL）

### Gap 1：Rollout 是单轮 vs 多轮 agentic
veRL 默认 rollout = `vllm.generate(prompt) → response`，**单轮**。  
我们 meeting 任务需要 OpenClaw agent loop（read 文件 → 思考 → write 文件 → ...，3-7 turn）。

veRL 有 `actor_rollout_ref.rollout.mode=async` + `data.return_raw_chat=True` 支持
agent loop（见 docs/start/agentic_rl.rst），但要写自定义 `AgentLoop` 子类（参见
recipe/langgraph_agent/example）。**工作量：1-2 周**。

### Gap 2：Reward 是 LLM judge vs rule-based
veRL 标准 reward 是 reward model 或 rule-based（GSM8K 这种 `#### 数字` 字符串匹配）。  
我们要把 DeepSeek-Chat LLM judge 包成 veRL 的 `reward_fn`。这部分相对简单，写个
`compute_score` 函数发 API 即可。**工作量：1-2 天**。

### Gap 3：Custom features 没法直接搬
- **Quality filter**（race-to-bottom 防御）：veRL 没有对应 hook，要在 dataset 阶段
  pre-filter，或 fork 修改 trainer。**工作量：1-2 天**。
- **Per-turn loss weighting**：veRL 默认 token-level，要 fork PPO loss 函数。
  **工作量：3-5 天**。
- **PRM with reward gate**：完全自定义，要写 reward post-processing。
  **工作量：3-5 天**。

总计移植到完整 feature parity：**~3 周工程**。

## 当前状态（2026-05-11 晨写时）

POC 跑通了第一个 9 步训练。validation 阶段超长，下一步如果继续：
- 关掉 validation（test_freq=999 或大数）
- 让训练跑完一个 epoch 出 checkpoint
- 或者：缩到只 5 步的 mini-test，确认 save / checkpoint 路径通

实际操作记录在 `02b_run_artifacts.md`（如果跑完）或本文 "时间线"。
