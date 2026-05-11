# 04 — RunPod MooseFS 读取性能坑（与本地 NVMe 对比 600x）

## 一句话结论

**RunPod 的 `/workspace/`（MooseFS 分布式存储）在高 I/O 竞争或 chunk-server
负载下，读 8 GB safetensors 模型分片的速度可能从理论 200 MB/s 跌到 ~4 MB/s**
（16 min/shard）。**把模型拷到 `/root/`（pod 本地 NVMe overlay）后加载从 48 min
→ 5 秒**，600× 加速。

> 凡是要在 RunPod 上跑训练，**第一件事就是把 HF 缓存 + safetensors 拷到本地盘**。
> 别相信 `/workspace/` 的 IO 性能，它对训练负载根本不靠谱。

## 实测数字

**场景**：Qwen3-4B（bf16，3 个 safetensors shard 共 7.6 GB），FSDP 2 个 worker
并发 mmap 同一份模型。

| 路径 | 文件系统 | 单 shard 加载时间 | 总加载时间 | 速率 |
|---|---|---|---|---|
| `/workspace/hf_cache/` | MooseFS（网络分布式） | **974 s** ≈ 16 min | **48 min**（外推） | ~4 MB/s |
| `/root/hf_cache/` | overlay（pod 本地 NVMe） | **~2 s** | **5 s** | ~1.5 GB/s |

`dd` 顺序读测试看到 6 GB/s（page cache 命中），但 safetensors 的 mmap +
随机访问让 MooseFS 退化到 ~4 MB/s。两个 worker 同时读放大了 chunk server 争抢。

**`cp -r /workspace/hf_cache/hub /root/hf_cache/` 本身花 5m53s**（22 MB/s 读
MooseFS），但只做一次，比每次训练前 hang 48 min 加载值。

## 怎么把模型搬到本地盘

```bash
# 1. 看 /root/ 空间够不够
df -h /root /tmp
# 通常 overlay 给 16-30 GB，Qwen3-4B bf16 只占 8 GB，够

# 2. cp 整个 HF hub 缓存目录
mkdir -p /root/hf_cache
cp -r /workspace/hf_cache/hub /root/hf_cache/

# 3. launch 脚本里把 HF_HOME / HF_HUB_CACHE 指到本地
export HF_HOME=/root/hf_cache
export HF_HUB_CACHE=$HF_HOME/hub
export TRANSFORMERS_CACHE=$HF_HUB_CACHE

# 4. 同步给 Ray actor 子进程
python3 -m verl.trainer.main_ppo \
  +ray_kwargs.ray_init.runtime_env.env_vars.HF_HOME=/root/hf_cache \
  +ray_kwargs.ray_init.runtime_env.env_vars.HF_HUB_CACHE=/root/hf_cache/hub \
  +ray_kwargs.ray_init.runtime_env.env_vars.TRANSFORMERS_CACHE=/root/hf_cache/hub \
  ...
```

## 不止读慢——MooseFS 写也不靠谱

之前观察到的两个独立现象：

1. **写 8 GB+ 单文件会截断**：torch.save 的 FSDP 整模型 ckpt（FSDP shards
   3-8 GB 每个）写完后 ZIP archive EOCD 缺失，`PytorchStreamReader failed
   reading zip archive`。详见 `docs/verl_port/03_findings.md` 的 ckpt corruption
   章节。**workaround：LoRA-only save（130 MB safetensors，单文件，可靠）**

2. **读高 IO 竞争下退化 50×**：上面这章。**workaround：模型放本地盘**

## 为什么 MooseFS 这么差

RunPod 的 `/workspace/` 是 `mfs#us-md-1.runpod.net:9421` MooseFS 集群挂载，
跨 pod 共享（所以容量大、可以 attach 到别的 pod），但：

- Chunk server 跟 pod 之间走网络（区域内 LAN，但 latency / 带宽不稳）
- 多个 pod 同时读同一份文件会去 chunk server 拉，带宽被抢
- mmap + 随机读尤其差（fault page 一次一次网络 roundtrip）
- 大文件 fsync 不一定真 sync（前述 ckpt 截断的根因之一）

适合：**只读冷数据**（如训练数据 parquet，~MB 级别一次读完）、**多 pod 共享**
（结果上传、跨 pod 备份）。

**不适合**：模型权重 mmap 加载、训练 ckpt 写、任何延迟敏感 IO。

## 现在生效的工程实践

1. **模型权重**：`/root/hf_cache/` 本地盘
2. **LoRA ckpt 保存**：`/workspace/verl_port/ckpt_openclaw/`（MooseFS 但 LoRA
   adapter 才 130 MB，写盘不会截断；多 pod 共享便于 bench）
3. **训练数据**：`/workspace/verl_port/data_meeting/` MooseFS 仍 OK（28 行
   parquet 才 22 KB，读一次进内存）
4. **日志**：`/workspace/verl_port/*.log` MooseFS（流式 append，写小块没问题）

## 时间线

- 2026-05-11 早上：训练在 step 0 卡 14 min 不动，诊断到 `Loading checkpoint
  shards 0/3` 长时间无进展
- 同日：测 MooseFS uncached read 速度，发现 974s/shard
- 当日：cp 模型到 `/root/`，5m53s 完成
- 重启训练：5s 加载完所有 shards，~600× 提速
