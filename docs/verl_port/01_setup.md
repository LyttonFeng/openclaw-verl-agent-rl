# 01 — veRL POC Setup（实操记录）

## 环境

| 项目 | 值 |
|---|---|
| Pod | RunPod 2x A100-80GB（同主分支实验 pod） |
| Python | 3.12.3 |
| Torch | 2.8.0+cu128 |
| veRL | 0.8.0.dev0（editable install at `/root/verl`，pip 显示 `Editable project location: /root/verl`） |
| 数据集 | GSM8K（math QA，single-turn，简单可控）— 不是 meeting 数据 |

## 踩到的坑（按发生顺序）

### 坑 1：FlashAttention 2 没装
```
ImportError: FlashAttention2 has been toggled on, but it cannot be used due to
the following error: the package flash_attn seems to be not installed
```

veRL 默认 `actor_rollout_ref.model.use_remove_padding=True`，需要 flash_attn。
两条路：装 flash_attn 或 关 use_remove_padding（性能低）。选装。

### 坑 2：编译 flash-attn 太慢
直接 `pip install flash-attn==2.7.4.post1` 触发源码编译（~30 min）。砍掉，下载预编译 wheel：
```
flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
```
（torch2.8 + cu12 + cp312 + abiTRUE 严格匹配）

### 坑 3：`/` 盘满
pod 的 `/` 只 30G，已用 97%（pip cache + /root/.cache 等）。pip install 因
"No space left on device" 失败。

清理动作：
- `pip cache purge`（44 MB）
- 删 `/root/.cache/huggingface`（7.6 GB，跟 `/workspace/hf_cache` 是 dup）
- `/` 释放到 7.8GB 可用，flash_attn 安装通过

### 坑 4：下载 flash-attn wheel 中断
`/tmp` 同样在 `/` 上盘满，curl 写不完 244MB。下到 `/workspace/verl_port/` 解决。

## 数据准备

```bash
python3 /root/verl/examples/data_preprocess/gsm8k.py \
  --local_save_dir /workspace/verl_port/gsm8k
# 输出 train.parquet (7473 records) + test.parquet (1319 records)
```

veRL 要求 parquet 列：`data_source`, `prompt`（list of message dicts）, `ability`,
`reward_model: {style, ground_truth}`, `extra_info`。

GSM8K 用 `reward_model.style="rule"`（纯字符串匹配 `#### <number>`）—
不需要 LLM judge。**这正是为啥选 GSM8K 当 POC 起点**：reward 路径已经
打通，不用做我们 meeting 那种 LLM-judge 集成。

## 启动配置（自定义 launcher）

`/workspace/verl_port/launch_gsm8k_poc.sh`，基于
`/root/verl/examples/grpo_trainer/run_qwen3_8b_fsdp.sh` 改：

| 参数 | 8B 例子默认 | 我们 POC |
|---|---|---|
| MODEL_PATH | Qwen/Qwen3-8B | **Qwen/Qwen3-4B** |
| NGPUS_PER_NODE | 8 | **2** |
| TRAIN_BATCH_SIZE | 1024 | **32** |
| PPO_MINI_BATCH_SIZE | 256 | **16** |
| MAX_PROMPT_LENGTH | 1024 | **512** |
| MAX_RESPONSE_LENGTH | 2048 | **1024** |
| PPO_MAX_TOKEN_LEN_PER_GPU | 24576 | **8192** |
| ROLLOUT_TP | 2 | **1**（每 GPU 自带 vllm engine） |
| ROLLOUT_GPU_MEM_UTIL | 0.6 | **0.4** |
| ROLLOUT_N | 5 | **2**（跟我们 GRPO group N=2 对齐） |
| TOTAL_EPOCHS | 15 | **1**（POC 跑完一遍就够） |
| TEST_FREQ | 5 | **10** |
| 数据 | gsm8k+math 多源 | **只 gsm8k** |
| logger | console+wandb | **console** only（pod 没 wandb 配置） |
| trainer.default_local_dir | (默认) | `/workspace/verl_port/checkpoints` |

## 启动后观察的关键阶段

1. **Ray init** (~10s)
2. **dataset filter & tokenize** (~30s)
3. **Worker init + model load** (~60s) — 包含 FSDP shard + ref policy
4. **vLLM warmup** (~60s) — capture CUDA graphs（每 GPU 一份 vllm engine）
5. **Training loop start** (~T+360s) — Training Progress: 0/233

每 step ~25s（rollout + train + log）。在 batch=32, ROLLOUT_N=2 配置下。

显存峰值：每 GPU ~52GB（FSDP shard + vllm engine + activations）。
