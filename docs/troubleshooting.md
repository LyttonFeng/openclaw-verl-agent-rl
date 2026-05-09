# 故障排查

## vLLM 不加载新的 LoRA adapter

vLLM 必须在启动时带上 `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` 和
`--enable-lora`。否则 `/v1/load_lora_adapter` endpoint 会静默 no-op。
热加载后请验证：

```bash
curl -s http://127.0.0.1:8021/v1/models | python -c "import json,sys; print(json.load(sys.stdin))"
```

新 LoRA 名称应该出现在 `data[].id` 之下。

## 训练与 bench 之间的 rope-scaling 不匹配

vLLM 必须用 `--rope-scaling '{"type":"dynamic","factor":2.0}'` 和
`--max-model-len 81920` 运行。训练用 `--rope-scaling-factor 2.0` 和
`--max-seq-length 81920`。Rope 不匹配是最常见的静默可复现性陷阱 — 模型在
训练和推理时收到不同的位置编码并产生无意义的分数。

## DeepSeek judge 限流 / 失败

PRM scoring 步骤每个 turn 调用 DeepSeek（默认 4 个并行 worker）。
如果你撞到限流：

```bash
SCORE_MAX_WORKERS=2 \
    python -m agent_loop.roadmap_prm.scripts.score_trajectories \
    --graded-file ... --max-workers 2
```

检查 API key：

```bash
curl -s -H "Authorization: Bearer $DEEPSEEK_API_KEY" https://api.deepseek.com/v1/models | head
```

## Rollout 卡住 / 慢

并发必须 ≥ 4。`rl/train/run_meeting_grpo_prm_round.sh` 中默认设
`NUM_WORKERS=4`。如果 log 行显示 `Concurrency: 1 worker(s)`，意味着脚本
启动时没带并行 flag — kill 掉用 `NUM_WORKERS=4` 重启。

也检查 vLLM 的 GPU 利用率（`nvidia-smi`）。如果 GPU 1 已经 100% 但
rollout 仍慢，瓶颈在 judge 调用 — 增加 `NUM_WORKERS`（同时也提升
DeepSeek QPS）。

## OpenClaw `command not found`

OpenClaw 必须装在与训练同一台 host 上（不使用 SSH/ECS 路径）。参考版本
`2026.4.5 (3e72c03)`：

```bash
which openclaw && openclaw --version
```

如果缺失：用与 source pod 相同的流程安装（从 ECS rsync 或从 OpenClaw repo
pip install）。

## Grading 时 workspace 文件缺失

Bench 在一次 benchmark.py 调用的所有任务间共享
`/tmp/pinchbench/<NNNN>/agent_workspace`。benchmark 在每个任务的 rollout
完成后立即对其 workspace 做快照，先于下一个任务开始。如果 grading 看到
空文件，请确认 `scripts/benchmark.py` 中的快照逻辑没有静默失败
（查找 `Snapshotted workspace to ...` 行）。

## Diagnostics 报告 `output_not_written` 但文件存在

Diagnostics 主要检查 in-trajectory 的 `write` tool 调用，然后回退到
filesystem 读取。如果 workspace 已经被后续 bench task 覆盖，只有
in-trajectory 记录是可靠的。Diagnostics 模块已经处理了这种情况 — 但如果
报告错了，看 `result.json` 中的 `task.workspace` 字段，确认它指向正确的
NNNN 目录。

## 训练时 OOM

降低序列上限或 grad-accum：

```bash
MAX_SEQ_LEN=40960 GRAD_ACCUM=1 ROUND_NUM=1 \
    bash rl/train/run_meeting_grpo_prm_round.sh
```

注意：把 `MAX_SEQ_LEN` 降到最长训练 transcript 之下会截断它。Council
transcript 是 206KB；rope=2 下刚好能在 80K 内放下。

## 单次 run 的结果与实验报告不符

实验报告使用的是 **3-run 平均**。单次 run 分数即使在同一个 checkpoint 上
也有 ~5-10pp 的 variance。始终用 `--runs 3` 重新 bench。
