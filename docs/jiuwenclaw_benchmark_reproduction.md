# JiuwenClaw Meeting Benchmark Reproduction

本文档给同事复现 JiuwenClaw runtime 上的 PinchBench `meeting_analysis` 5-task benchmark。它包含：环境要求、启动脚本、benchmark 命令、已有结果和常见坑。

## 1. Benchmark 范围

默认复现 5 个 held-out meeting_analysis 任务：

```text
task_meeting_advisory_stakeholders
task_meeting_council_votes
task_meeting_gov_speaker_summary
task_meeting_tech_action_items
task_meeting_sentiment_analysis
```

这些任务来自 `rl/train/meeting_analysis_split.json` 的 test split；任务定义在 `pinchbench_tasks/meeting_analysis/`，会议 transcript 在 `assets/meetings/`。

## 2. 已有结果

### 2.1 JiuwenClaw runtime 结果

结果文件：`experiments/verl_port_poc/bench_results/jiuwen_runtime/base_4b.json`

| Runtime | Model | Runs | Overall | 备注 |
|---|---|---:|---:|---|
| JiuwenClaw | Qwen3-4B base | 1 × 5 tasks | **41.85%** | `completed=false`，但 5 个任务均有记录；`advisory_stakeholders` timeout，`council_votes` 产物缺失为 0 |

Per-task：

| Task | Status | Score |
|---|---|---:|
| `task_meeting_advisory_stakeholders` | timeout | 0.0000 |
| `task_meeting_council_votes` | success | 0.0000 |
| `task_meeting_gov_speaker_summary` | success | 0.6424 |
| `task_meeting_tech_action_items` | success | 0.5750 |
| `task_meeting_sentiment_analysis` | success | 0.8750 |
| **Mean** |  | **0.4185** |

JiuwenClaw runtime 下也留了两个 OpenClaw-trained LoRA 的 sanity run，主要用于证明 runtime 分布不一致会导致退化，不建议作为部署结果：

| Runtime | Model | Result file | Overall |
|---|---|---|---:|
| JiuwenClaw | step8 LoRA | `experiments/verl_port_poc/bench_results/jiuwen_runtime/step8_lora.json` | 10.17% |
| JiuwenClaw | step16 LoRA | `experiments/verl_port_poc/bench_results/jiuwen_runtime/step16_lora.json` | 27.68% |

### 2.2 参考口径，不能和 JiuwenClaw runtime 直接混用

| Runtime / Pod | Model | Source | Overall |
|---|---|---|---:|
| OpenClaw, 2026-05 复现 pod | Qwen3-4B base, rope=2 / 64K, fp32 | `docs/experiment_report.md` | **44.68%** |
| OpenClaw, old SOTA pod | Qwen3-4B base, rope=2 / 80K | `docs/experiment_report.md` §5 | **50.6%** |

上面两个数字是 OpenClaw 链路参考值，不是 JiuwenClaw runtime 的复现结果。JiuwenClaw 当前 artifact 的客观数值是 **41.85%**。

## 3. 环境要求

### 3.1 机器

推荐 RunPod / A100-80GB。base Qwen3-4B benchmark 至少需要：

| 组件 | 建议 |
|---|---|
| GPU | 1 × A100-80GB 跑 vLLM；JiuwenClaw stack 和 judge 走 CPU/API |
| Container disk | 放 Qwen3-4B 权重和 node/python 依赖，避免网络盘 I/O 抖动 |
| Network disk | 放输出、benchmark artifacts、LoRA checkpoint |
| Python | 3.12 |
| vLLM | 0.10.2，使用 `qwen3` reasoning parser 和 `hermes` tool parser |
| JiuwenClaw | `/root/jiuwen_work/jiuwenclaw`，使用 uv venv |
| PinchBench runner | `/root/jiuwen_work/pinchbench/scripts/run_pinchbench_jiuwenclaw.py` |
| OpenClaw-RL repo | `/workspace/openclaw-verl-agent-rl` 或本仓库路径 |

### 3.2 API key

需要 DeepSeek judge key：

```bash
cat > ~/.pinchbench_env <<'EOF_KEY'
export DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxx
EOF_KEY
chmod 600 ~/.pinchbench_env
source ~/.pinchbench_env
```

### 3.3 模型权重

默认脚本假设 Qwen3-4B 在：

```bash
/root/hf_cache/Qwen3-4B
```

如果不在这个路径，用 `MODEL_DIR` 覆盖：

```bash
MODEL_DIR=/path/to/Qwen3-4B bash experiments/verl_port_poc/launch_vllm_qwen3_base.sh
```

## 4. 复现脚本

### 4.1 一键入口

```bash
source ~/.pinchbench_env
bash experiments/verl_port_poc/reproduce_jiuwenclaw_benchmark.sh /workspace/verl_port/bench/jiuwen_base_4b_repro 3
```

参数：

```text
$1: 输出目录，默认 /workspace/verl_port/bench/jiuwen_base_4b_repro
$2: run 次数，默认 3
```

这个 wrapper 会：

1. 启动或复用 Qwen3-4B vLLM；
2. 启动或复用 JiuwenClaw headless stack；
3. 跑 5 个 meeting_analysis held-out tasks；
4. 用 DeepSeek judge 打分；
5. 调 `bench_summarize.py` 汇总每轮和跨轮均值。

### 4.2 分步命令

启动 vLLM：

```bash
MODEL_DIR=/root/hf_cache/Qwen3-4B \
PORT=8123 \
SERVED_NAME=Qwen3-4B \
CUDA_VISIBLE_DEVICES=0 \
bash experiments/verl_port_poc/launch_vllm_qwen3_base.sh
```

跑 JiuwenClaw benchmark：

```bash
source ~/.pinchbench_env
API_BASE=http://127.0.0.1:8123/v1 \
MODEL_NAME=Qwen3-4B \
JIUWENCLAW_REPO=/root/jiuwen_work/jiuwenclaw \
SKILL_ROOT=/workspace/openclaw-verl-agent-rl \
JIUWENCLAW_DATA_DIR=/root/.jiuwenclaw \
bash experiments/verl_port_poc/bench_jw_baseline.sh \
  /workspace/verl_port/bench/jiuwen_base_4b_repro \
  3
```

汇总已有结果：

```bash
python3 experiments/verl_port_poc/bench_summarize.py \
  experiments/verl_port_poc/bench_results/jiuwen_runtime/base_4b.json
```

## 5. 关键环境变量

| 变量 | 默认值 | 说明 |
|---|---|---|
| `MODEL_DIR` | `/root/hf_cache/Qwen3-4B` | 本地 Qwen3-4B 权重目录 |
| `PORT` | `8123` | vLLM OpenAI-compatible 端口 |
| `SERVED_NAME` | `Qwen3-4B` | vLLM served model name，必须和 `MODEL_NAME` 一致 |
| `API_BASE` | required | JiuwenClaw stack 转发到的 vLLM endpoint，例如 `http://127.0.0.1:8123/v1` |
| `MODEL_NAME` | required | JiuwenClaw 使用的模型名，例如 `Qwen3-4B` |
| `DEEPSEEK_API_KEY` | required | LLM judge key |
| `JIUWENCLAW_REPO` | `/root/jiuwen_work/jiuwenclaw` | JiuwenClaw repo |
| `SKILL_ROOT` | `/workspace/openclaw-verl-agent-rl` | 本 benchmark repo 根目录 |
| `JIUWENCLAW_DATA_DIR` | `/root/.jiuwenclaw` | JiuwenClaw 数据目录；必须和 `--jiuwen-data-dir` 一致 |
| `WS_PORT` | `611` | JiuwenClaw websocket port |
| `AGENT_SERVER_PORT` | `18092` | JiuwenClaw agent server port |
| `GATEWAY_PORT` | `19001` | JiuwenClaw gateway port |
| `JIUWENCLAW_PROGRESSIVE_VISIBLE` | `read_file,write_file,list_files,edit_file,grep,glob,todo_create,todo_list,bash,memory_search,code` | 工具白名单 |

## 6. 关键正确性点

1. **vLLM 必须开 rope=2 长上下文。** 任务里 council transcript 很长，Qwen3-4B 原生 40960 context 不够。`launch_vllm_qwen3_base.sh` 会把 `config.json` patch 成 yarn rope factor 2.0，`MAX_MODEL_LEN=81920`。
2. **reasoning parser 用 `qwen3`，tool parser 用 `hermes`。** 这个组合对应 vLLM 0.10.2；不要换成 `deepseek_r1`，它会把有效 content 吃掉，JiuwenClaw 看到空回复后容易循环。
3. **`JIUWENCLAW_DATA_DIR` 必须和 bench 的 `--jiuwen-data-dir` 一致。** 不一致会导致 workspace 文件复制到 A 目录、JiuwenClaw 去 B 目录读，最后 File not found。
4. **工具白名单不要暴露 `write_memory` 和 `wiki_ingest`。** `write_memory` 容易和 `write_file` 混淆，产物不落到 workspace；`wiki_ingest` 可能吃掉 transcript，导致后续 read_file 失败。
5. **不要把 OpenClaw-trained LoRA 结果当作 JiuwenClaw runtime 可部署结果。** 现有 step8/step16 JiuwenClaw runtime 分别只有 10.17% / 27.68%，主要是工具分布和路径风格不一致导致。

## 7. 输出结构

`bench_jw_baseline.sh` 每轮会生成一个 run 目录，例如：

```text
/workspace/verl_port/bench/jiuwen_base_4b_repro/
  run1/
    <timestamp>/results.json
    <timestamp>/workspaces/...
    <timestamp>/transcripts/...
  run2/
  run3/
```

最终汇总由：

```bash
python3 experiments/verl_port_poc/bench_summarize.py /workspace/verl_port/bench/jiuwen_base_4b_repro
```

输出每个 task 的 status / score，以及跨 run 的 mean / std / range。

## 8. 相关文件

| 文件 | 用途 |
|---|---|
| `experiments/verl_port_poc/reproduce_jiuwenclaw_benchmark.sh` | 一键复现入口 |
| `experiments/verl_port_poc/launch_vllm_qwen3_base.sh` | 启动 Qwen3-4B vLLM，含 rope=2 patch |
| `experiments/verl_port_poc/bench_jw_baseline.sh` | JiuwenClaw benchmark runner |
| `experiments/verl_port_poc/bench_summarize.py` | benchmark 结果汇总 |
| `experiments/verl_port_poc/bench_results/jiuwen_runtime/base_4b.json` | 已有 JiuwenClaw base 4B 结果，41.85% |
| `docs/verl_port/07_jiuwenclaw_runtime.md` | JiuwenClaw runtime 切换诊断 |
| `docs/experiment_report.md` | OpenClaw reference benchmark 和训练结果 |
