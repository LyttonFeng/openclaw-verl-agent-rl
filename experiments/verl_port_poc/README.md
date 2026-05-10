# veRL POC artifacts（experiment/verl-port 分支）

> ⚠️ **本目录在 `experiment/verl-port` 分支，不会合并到 main**。是一晚的探索性
> POC，目标"看 veRL 能不能跑起来"，不追求复现 47.80% SOTA。

## 文件

- `launch_gsm8k_poc.sh` — 启动脚本（pod 上 `/workspace/verl_port/launch_gsm8k_poc.sh`）。
  基于 veRL 自带的 `examples/grpo_trainer/run_qwen3_8b_fsdp.sh` 改的，缩到 2x A100 + Qwen3-4B + GSM8K。
- `run2.log` — 完整训练 log（823 行）。截至 step 9 完成 + step 10 checkpoint 保存。

## POC 结果摘要

✅ **训练循环跑通**（step 1-9 各 ~25 秒）：
```
Training Progress: 9/233 [04:00<1:32:43, 24.84s/it]
```

✅ **第一个 checkpoint 自动保存**（pod 上 `/workspace/verl_port/checkpoints/global_step_10/actor/`，42 GB FSDP 4 个分片）

⚠️ **step 10 后卡死**（疑似 validation hang 或 Ray worker 死锁；进程 zombie 化，GPU 内存未释放）

## 详情

完整分析见 `docs/verl_port/{README.md, 01_setup.md, 02_first_run.md, 03_findings.md}`。
