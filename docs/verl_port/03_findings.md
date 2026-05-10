# 03 — 关键发现 & 决策建议

## 一句话结论

veRL **能在本 pod 跑起来**（Qwen3-4B + 2x A100 + GSM8K POC），但**集成到我们
meeting agentic 任务约需 3 周工程**。短期没有立即迁移的回报。

## 已成功验证的事实

✅ veRL 0.8.0.dev0 在新 pod（torch 2.8 + cu128 + python 3.12）兼容运行  
✅ FSDP + vLLM hybrid engine 在 2x A100-80GB 上正常工作（每 GPU 峰值 ~52GB）  
✅ Ray 调度、AgentLoop worker、CUDA graph capture 全流程跑通  
✅ 训练循环执行：9 个 step 在 4 分钟内完成，每 step ~25s 稳定  
✅ Checkpoint 在 `save_freq=10` 触发自动保存：`global_step_10/actor/` 含 FSDP 4 个分片文件，**42 GB 全参数**（注意：不是 LoRA）

## 关键 trade-off：veRL vs 自定义 250 行

### veRL 强项

1. **工业级显存管理**：vLLM `enable_sleep_mode=True` 让 rollout / training
   交替使用 GPU，hybrid engine。我们自定义代码用 `device_map="auto"` 是 hack。
2. **标准算法**：PPO + GAE + clip + KL + low-var-kl 都是经过验证的实现，参数明确。
3. **多 backend**：vllm / sglang / trtllm 一键切换。
4. **FSDP 原生**：未来上 7B+ 或多节点直接可用，不用改代码。
5. **生态**：mlflow trace、checkpoint 一致性、agent loop API 都是为 production 准备。

### veRL 短板（对我们当前场景）

1. **启动开销大**：~6 分钟（Ray + workers + vllm warmup）vs 自定义 ~30 秒。
   POC 调试时尤其痛。
2. **保存的是全参数 checkpoint**（42 GB / step）vs 我们 LoRA adapter（~130 MB）。
   要保存 LoRA 需要改 `actor_rollout_ref.model.lora_rank=16` + 配 PEFT。
3. **配置复杂**：30+ Hydra 参数 vs 我们 6 个 CLI flag。学习曲线陡。
4. **rollout 默认单轮**：要做 agentic 必须写自定义 `AgentLoop`。
5. **reward 默认 rule-based 或 reward model**：LLM judge 要包成 reward_fn。

### 量化对比

| 维度 | 自定义 250 行 | veRL POC |
|---|---|---|
| **启动时间** | ~30s | ~360s |
| **每 step** | ~5s/sample（vanilla PG） | ~25s/step（含 rollout + GRPO + log） |
| **Checkpoint size** | 130 MB（LoRA） | 42 GB（FSDP 全参） |
| **达到 47.80%** | ✅（已经做到） | ❌（一晚 POC 不够，需要 3 周集成） |
| **改 reward shaping 公式** | 改一行 | fork + 改 PPO loss |
| **Per-turn loss 等定制** | 已实现 | 都得自己加 hook |
| **scale 到 7B+** | 必须重写 | 改个 MODEL_PATH 即可 |
| **工程稳定性** | 我们自己负责 | 字节生产级 |

## 集成 meeting 任务的具体 gap & 工作量

### Gap 1: Agent loop 集成（**最难**）

veRL 提供 `recipe/langgraph_agent/`、`docs/sglang_multiturn/multiturn.rst` 作为
agentic 模板，但不直接支持 OpenClaw（CLI-based agent，跟 langgraph 模型不同）。

**工作量评估**：
- 写 `OpenClawAgentLoop` 子类，封装 `subprocess.Popen("openclaw", ...)`
- 实现 `run(prompt) → trajectory` 接口，把 OpenClaw stdout/stderr/transcript 解析回 token
- 处理 token 边界一致性（veRL 强调"训练用 inference 实际生成的 token"，OpenClaw 走自己的 chat template，要小心对齐）
- **3-5 个工程日，需要熟悉 veRL agent loop API**

### Gap 2: LLM judge 包成 reward function

veRL 用 `compute_score(data_source, solution_str, ground_truth, extra_info) → float`
接口（见 `verl/utils/reward_score/`）。

**工作量评估**：
- 写 `meeting_judge_score()` 包装 DeepSeek-Chat API 调用
- 复用现有 `rewards/meeting_reward.py` 逻辑
- 处理 async / batch 优化（避免训练 step 卡 LLM API）
- **1-2 个工程日**

### Gap 3: 数据格式适配

把现有 `pinchbench_tasks/meeting_analysis/` 的 23 个任务转成 veRL parquet：
- `data_source = "meeting_analysis"`
- `prompt = [{"role": "user", "content": <任务描述>}]`
- `reward_model = {"style": "model", "ground_truth": <expected outputs hint>}`
- `extra_info = {<assets, transcripts paths>}`

**工作量评估**：1 工程日。

### Gap 4: Custom features 移植

| Feature | 自定义代码位置 | veRL 移植方式 | 工作量 |
|---|---|---|---|
| Quality filter (race-to-bottom) | `apply_quality_filter.py` | 改 `verl.utils.dataset` 加 hook 或 pre-filter | 1-2 天 |
| PPO + KL k3 estimator | `train_meeting_grpo_step.py` | veRL 已有 `kl_loss_type=low_var_kl` | 0 天（直接用） |
| Per-turn loss weighting | 同上 | fork PPO loss 函数 | 3-5 天 |
| PRM + reward gate | 同上 + `apply_quality_filter.py` | 完全自定义 reward post-processing | 3-5 天 |

### 总计

**3 周工程**（含调试）才能用 veRL 完整复现当前 47.80% setting。

## 决策建议

### 该用 veRL 的场景

- 模型上 **7B+** 或多节点：自定义代码撑不住，必须 FSDP + 工业基础设施
- 长期项目，需要持续集成新 RLHF 算法（DPO / KTO / GRPO 变种）
- 团队有 RL framework 维护 budget

### 留在自定义代码的场景（**推荐当前**）

- 模型 4B 单机够用
- 实验期，需要快速迭代 reward shaping / loss 公式
- 算法专一（GRPO + 我们已实现的几个 fix）
- 用户主要目标是**学习和论文/报告**，不是 production

### 真要迁，建议路径

1. **不要从头开始**——保留 main 分支当前自定义代码 working
2. 在 `experiment/verl-port` 上**只做 Gap 2 和 3**（reward + 数据），用 veRL 标准
   single-turn rollout 跑通一次（哪怕分数低于 47.80%），证明集成可行
3. 然后做 Gap 1（agent loop）—— 这是最有学习价值的部分
4. Gap 4 custom features 视情况再说

## 本次 POC artifacts

- `/workspace/verl_port/launch_gsm8k_poc.sh` — 启动脚本（POC 配置）
- `/workspace/verl_port/run2.log` — 完整运行日志
- `/workspace/verl_port/checkpoints/global_step_10/` — 第一个 checkpoint（42 GB）
- `/workspace/verl_port/gsm8k/{train,test}.parquet` — 预处理数据
- `/workspace/verl_port/flash_attn-2.8.3+...whl` — 预编译 wheel（备用）

repo 内：
- `docs/verl_port/README.md` — 本目录入口
- `docs/verl_port/01_setup.md` — 环境准备
- `docs/verl_port/02_first_run.md` — 第一次跑通过程
- `docs/verl_port/03_findings.md` — 本文（findings + 决策）
