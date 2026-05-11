# 05 — 第一次 veRL+OpenClaw bench：step_8 LoRA = **47.7%**

## TL;DR

veRL 0.8 + OpenClaw 多轮 rollout + GRPO + LoRA r=16 训练 **8 个 step** 拿到的
LoRA adapter，在 main 分支同一份 5-task held-out 测试集上跑 PinchBench
official harness：

| task | score |
|---|---|
| `task_meeting_advisory_stakeholders` (NTIA) | **48 %** |
| `task_meeting_council_votes` (Tampa) | **16 %** |
| `task_meeting_gov_speaker_summary` (NASA UAP) | **41 %** |
| `task_meeting_tech_action_items` (GitLab) | **65 %** |
| `task_meeting_sentiment_analysis` (GitLab) | **69 %** |
| **Mean** | **47.7 %** |

跟 main 分支 vanilla GRPO 47.80% baseline **几乎一模一样**。

## 重要警告 — 这不是"涨了"的证明

1. **数据 leak**：本次训练用的是早期 28-task 老 `train.parquet`（5 个 test
   task 也在训练集里）。后来生成的 `train_full23.parquet` + `val_5test.parquet`
   是 clean split，**下次重训要换上**。当前这个 47.7% 数字里有训练数据
   contamination 的成分。
2. **样本太小**：5 task × 1 run。main 分支 doc 推荐 `--runs 3` 取平均。
   单次 noise 很大（task_council_votes 拿 16 %、task_sentiment_analysis 69 %）。
3. **训练量极小**：只 8 个 grad step。完整一遍 24-step 都没跑完。这个数字
   可能更多反映"base Qwen3-4B + OpenClaw 多轮"的能力，不是 RL 训练的功劳。

下次干净对比需要：
- base Qwen3-4B 没有 LoRA 跑同一组 bench → 拿真 baseline
- 用 clean split + `--runs 3` 重训完整 24 step → 拿无 leak 的训完数字

## 流水线

整套**完整跑通**，所有非平凡 bug 都已踩过 + 修：

| 阶段 | 关键文件 / 命令 |
|---|---|
| 训练 | `experiments/verl_port_poc/launch_meeting_openclaw_lora.sh` |
| OpenClaw 集成 | `verl.experimental.agent_loop.AgentLoopBase` 子类 `OpenClawAgentLoop`（pinchbench-skill 移植，本地运行无需 ECS） |
| LoRA-only ckpt save | inline patch 进 `verl/workers/engine/fsdp/transformer_impl.py::FSDPEngine.save_checkpoint`（绕 MooseFS 8 GB 写截断坑） |
| HF 缓存路径 | `/root/hf_cache/`（本地 NVMe，避免 MooseFS 16 min/shard 慢读）|
| Rope=2 64K context | 直接写 `Qwen3-4B/config.json` 的 `rope_scaling` + `max_position_embeddings`，**不要**走 Hydra CLI dict |
| Bench harness | `experiments/verl_port_poc/run_bench_step8.sh` → `scripts/benchmark.py` |
| Judge backend | `--judge deepseek-chat` 触发 `judge_backend=api`，否则默认起 OpenClaw judge agent 会失败 |
| Per-task timeout | `--timeout-multiplier 3` 把每 task 180s 顶到 540s，给 LoRA 多轮 recover 空间 |
| vLLM serve | `--enable-lora --lora-modules step8-lora=<adapter_dir> --enable-auto-tool-choice --tool-call-parser hermes` |

## 复现命令

### 训练
```bash
# Pod prerequisites:
#   /root/hf_cache/  with Qwen3-4B (cp from /workspace/hf_cache once)
#   /workspace/verl_port/data_meeting/train_full23.parquet
#   /workspace/verl_port/data_meeting/val_5test.parquet
#   /workspace/verl_port/openclaw_integration/  (pinchbench-skill rl/* + agent_loop/)
#   inline source-patch transformer_impl.py with LoRA-only save
#   /usr/local/bin/openclaw + self-ssh + DEEPSEEK_API_KEY

bash /workspace/verl_port/launch_meeting_openclaw_lora.sh
```

### Bench
```bash
# Start vLLM serve with LoRA adapter hot-loaded
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B \
  --served-model-name qwen3-4b-base \
  --tensor-parallel-size 2 \
  --max-model-len 65536 \
  --gpu-memory-utilization 0.65 \
  --enable-lora --max-loras 4 --max-lora-rank 16 \
  --lora-modules step8-lora=/workspace/verl_port/ckpt_openclaw/global_step_8/actor/lora_adapter \
  --enable-auto-tool-choice --tool-call-parser hermes \
  --host 127.0.0.1 --port 8000 --enforce-eager

# Then run the bench
bash experiments/verl_port_poc/run_bench_step8.sh
```

## 训练摘要

| 维度 | 设置 |
|---|---|
| 算法 | vanilla GRPO, `norm_adv_by_std_in_grpo=False` (std≡1 for group=2) |
| Reward | terminal-only DeepSeek-chat judge（无 PRM、无 quality filter） |
| Rollout | OpenClaw 多轮（max 20 turns）+ DeepSeek terminal grading |
| 基座 | Qwen3-4B (bf16) + LoRA r=16 α=32 lr=2e-6 |
| 训练量 | step 8/24 — 因 OpenClaw 多轮 wall 慢，我们截在 step 8 ckpt 跑 bench |
| Wall time | step 8 落盘前约 50 min |

整套 setup 跟 main 分支 `run_meeting_grpo_prm_round.sh` 默认配置基本一致，
区别在框架（veRL 0.8 hybrid engine vs main 250 行自定义训练循环）。

## 下一步

1. 用 clean split (`train_full23.parquet` + `val_5test.parquet`) 重训
2. 跑 24 完整 step + step 16 / step 24 中间 ckpt 都 bench 一遍看趋势
3. 同时跑 base Qwen3-4B（无 LoRA）的 bench 作为 baseline
4. `--runs 3` 取平均
