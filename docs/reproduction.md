# 复现

复现本仓库 meeting_analysis GRPO 训练的端到端流程。
目标：与 [`experiment_report.md`](experiment_report.md) 中报告的 5-task bench
分数一致。

通过同一个 wrapper 支持两条训练路径：

- **Terminal-only**（无 PRM，baseline）：`PRM_BETA=0`
- **Terminal + PRM**（Roadmap PRM，judge-gate）：默认设置

## 1. 环境

| 组件 | 版本 | 备注 |
|---|---|---|
| Python | 3.12 | 推荐使用 venv |
| **veRL** | **可选** | 本配方使用的 GRPO 训练器（`train_meeting_grpo_step.py`）是一个自包含的 PyTorch + transformers + peft 循环，**不**导入 veRL。veRL 仅遗留 PPO 脚本（`launch_main_ppo.py`、`run_reinforce_lora.sh`）需要，而那些脚本不是用来产生 SOTA 结果的。如果你只需要本文档配方，可跳过下面的 veRL 安装步骤。 |
| vLLM | 0.10.2 | 配合 `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` 实现热加载 |
| Transformers | 4.57.1 | |
| PEFT | latest compatible | LoRA 训练 |
| Torch | 2.8.0+cu128 | 在 A100 上测试 |
| OpenClaw CLI | `2026.4.5` (3e72c03) | 在本地运行 multi-turn agent |
| PinchBench | 1.2.1 | 任务定义 + grader（本仓库已嵌入所需子集） |
| GPU | 2 × A100-80GB | GPU 0 = 训练，GPU 1 = vLLM |

> **不需要 ECS / 外部 runtime。** OpenClaw 在与训练相同的 pod 上本地运行。
> 之前 task16 路径用的 SSH-to-OpenClaw 模式在这里不再使用。

### 安装 Python 依赖

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 从源码安装 veRL（可选 — 仅在需要遗留 PPO 时才装）

> **当前 GRPO 训练器（`train_meeting_grpo_step.py`）不 import veRL。**
> 下面 §3 的端到端配方在没装 veRL 的情况下也能工作。安装它的唯一理由是
> 你还想运行遗留的 PPO 脚本（`rl/train/launch_main_ppo.py` /
> `run_reinforce_lora.sh`）— 这些都没有用于 `experiment_report.md`
> 中的 SOTA 结果。
>
> 两个 pod（新的复现 pod 和原始 2026-05 SOTA pod）所有报告数字都跑的是
> 同一个极简的 `train_meeting_grpo_step.py` — 仓库名 `openclaw-verl-agent-rl`
> 是历史遗留。

如果你确实想装 veRL：

```bash
git clone https://github.com/volcengine/verl.git ~/verl
cd ~/verl
# Reference state: 0.8.0.dev0 head as of 2026-04.
pip install -e .
python -c "import verl; print(verl.__version__, verl.__file__)"
# expected: 0.8.0.dev0 ~/verl/verl/__init__.py
```

### 安装 OpenClaw CLI

OpenClaw 是驱动每个 rollout 的 multi-turn agent runtime。它是一个公开的
**npm package**：

```bash
# Make sure node + npm are available (e.g. via nvm)
node --version && npm --version

# Install globally to local disk — DO NOT install under /workspace or any
# network-mounted FS; OpenClaw has hundreds of transitive deps and a
# network-FS install can take 50+ minutes vs. ~30s on local disk.
npm install -g openclaw@2026.4.5

# If your `node` binary lives in a non-standard location (e.g. nvm puts it
# under /workspace/nvm/...), the openclaw shebang `#!/usr/bin/env node`
# may not find it after `npm install -g`. Symlink node into PATH:
ln -sf "$(which node)" /usr/local/bin/node

# Verify
which openclaw
openclaw --version    # → OpenClaw 2026.4.5 (3e72c03)
```

安装后的 package 在磁盘上约 1.3 GB（扩展 + 传递依赖）；本地安装目录默认为
`/usr/local/lib/node_modules/openclaw/`。如果你的环境中设置了
`OPENCLAW_BIN`，训练 wrapper 会优先使用它而不是查 PATH。

### API keys

DeepSeek 是 **terminal grading 和 PRM scoring 共同的**默认 judge：

```bash
cat > ~/.pinchbench_env <<'EOF'
export DEEPSEEK_API_KEY=sk-xxxxxxxxxxxx
EOF
chmod 600 ~/.pinchbench_env
```

DeepSeek 是本仓库默认配置的唯一 judge provider。可以通过覆盖
`MEETING_JUDGE_BASE_URL` + `MEETING_JUDGE_MODEL` 来换其他 OpenAI 兼容
endpoint（并以 `DEEPSEEK_API_KEY` env var 提供匹配的 key — judge 解析链
读的就是这个名字）。

## 2. 数据

训练/测试 split 定义：

```text
rl/train/meeting_analysis_split.json
```

| Split | 任务 | 源会议（4 份真实 transcript） |
|---|---|---|
| **Train** | 23 个任务 | NTIA spectrum advisory (71KB), GitLab PMM (34KB), Tampa City Council (206KB), NASA UAP hearing (265KB) |
| **Test** | 5 个任务 | 同样 4 份 transcript，held-out 任务：`advisory_stakeholders`, `council_votes`, `gov_speaker_summary`, `tech_action_items`, `sentiment_analysis` |

任务定义在 `pinchbench_tasks/meeting_analysis/`。
Transcript 在 `assets/meetings/`。
Roadmap（PRM 的每任务专家 milestone）在 `agent_loop/roadmap_prm/roadmaps/`。

构建训练 prompt（parquet）一次：

```bash
python rl/train/build_meeting_analysis_prompts.py \
    --tasks-dir pinchbench_tasks/meeting_analysis \
    --split-file rl/train/meeting_analysis_split.json \
    --output-dir data/meeting_prompts
```

## 3. 启动 vLLM (GPU 1)

```bash
CUDA_VISIBLE_DEVICES=1 \
VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen3-4B \
    --served-model-name Qwen3-4B \
    --host 0.0.0.0 --port 8021 \
    --max-model-len 81920 \
    --rope-scaling '{"type":"dynamic","factor":2.0}' \
    --gpu-memory-utilization 0.85 \
    --dtype bfloat16 \
    --trust-remote-code \
    --enable-auto-tool-choice --tool-call-parser hermes \
    --reasoning-parser qwen3 \
    --enable-lora --max-loras 1 --max-lora-rank 16
```

> 重要：`rope-scaling factor=2.0` 和 `max-model-len=81920` 必须与训练所用一致。
> rollout 与训练之间的 rope mismatch 是最常见的可复现性陷阱。

## 4. 训练

### 单轮，terminal + PRM（默认，推荐）

```bash
ROUND_NUM=1 bash rl/train/run_meeting_grpo_prm_round.sh
```

这会运行完整 pipeline（每步写自己的 log + 中间文件）：

1. **Rollouts** — 23 个训练任务 × 2 个 response = 46 条轨迹，4 个并行 worker 通过 OpenClaw。
2. **Terminal grade** — 自动 check + DeepSeek LLM judge → `graded_trajectories.jsonl`。
3. **PRM scoring** — 带 terminal-completion gate 的 DSv4 judge（如果 `SKIP_PRM_SCORING=1` 则跳过）→ `graded_trajectories_prm.jsonl`。
4. **Variance filter + pos-only clip** — `select_grpo_samples.py` 丢弃零方差 group；默认 `POS_ONLY_CLIP=1` 把 -1 turn 分数 clip 到 0 → `graded_trajectories_prm_pos_only.jsonl`。
5. **GRPO step** — 15 次更新，batch=2，lr=2e-6，mode=`PRM_MODE`。LoRA 保存到 `$BASE_DIR/round_1/checkpoint/lora_adapter`。
6. **Hot-load** — `POST /v1/load_lora_adapter` 把新 adapter 加载到 vLLM（无需重启）。
7. **3-run bench** — `scripts/benchmark.py` 在 5 个 held-out 测试任务上运行。

### 单轮，terminal-only (baseline)

```bash
PRM_BETA=0 \
ROUND_NUM=1 \
EXPERIMENT=meeting_grpo_terminal_v1 \
bash rl/train/run_meeting_grpo_prm_round.sh
```

> **注意：** `PRM_BETA=0` 让 PRM 分数对 GRPO advantage 没有影响，
> 所以更新就退化为 terminal-only。**PRM scoring 本身仍然会运行**
> （每个 assistant turn 一次 DeepSeek 调用，每轮 ~$0.05）— 它便宜，
> 而且作为诊断中的 control signal 有用。如果你想完全跳过 PRM 阶段
> （零 DeepSeek PRM 调用），设置 `SKIP_PRM_SCORING=1`（wrapper 会从
> grade 直接跳到 train，使用全零 per-turn 分数）。

### 从前一个 LoRA 继续

```bash
ROUND_NUM=2 \
PREV_LORA=/workspace/meeting_grpo_prm_v1/round_1/checkpoint/lora_adapter \
bash rl/train/run_meeting_grpo_prm_round.sh
```

### 关键参数

所有这些都可以通过环境变量覆盖。

| 变量 | 默认 | 含义 |
|---|---|---|
| `ROUND_NUM` | required | 轮次计数 |
| `PREV_LORA` | empty | 留空则从 base 开始，否则是 LoRA 路径 |
| `EXPERIMENT` | `meeting_grpo_prm_v1` | 输出子目录名（在 `BASE_DIR` 之下） |
| `BASE_DIR` | 自动检测：如果 `/workspace` 存在且可写则 `/workspace/$EXPERIMENT`，否则 `$HOME/grpo_runs/$EXPERIMENT` | 运行输出根目录 |
| `PRM_ALPHA` | `1.0` | advantage 公式中的 terminal weight |
| `PRM_BETA` | `0.10` | PRM weight (0 = terminal-only 消融；除非 `SKIP_PRM_SCORING=1`，否则 PRM scoring 仍然运行) |
| `PRM_MODE` | `additive` | `additive` 或 `multiplicative`（公式见 [`algorithm.md`](algorithm.md)） |
| `SKIP_PRM_SCORING` | `0` | 设为 `1` 完全跳过 DeepSeek PRM judge 步骤（合成全零 per-turn 分数） |
| `POS_ONLY_CLIP` | `1` | 训练前把负 PRM turn 分数 clip 到 0；设为 `0` 保留原始 `{-1,0,+1}` |
| `VARIANCE_THRESHOLD` | `1e-8` | 丢弃 terminal-score variance 低于该值的 GRPO group（无信号） |
| `N_RESPONSES` | `2` | 每个 prompt 的 rollout 数（GRPO group size） |
| `NUM_WORKERS` | `4` | 并行 rollout worker 数 |
| `MAX_SEQ_LEN` | `81920` | 训练序列上限（必须与 vLLM `--max-model-len` 一致） |
| `MEETING_JUDGE_PROVIDER` | `deepseek` | 为未来 provider 预留；目前默认仅接通 `deepseek` |
| `TASKS_DIR` | `pinchbench_tasks/meeting_analysis` | 任务 `.md` 查找根 |
| `VLLM_BASE_URL` | `http://127.0.0.1:8021/v1` | vLLM endpoint |
| `SERVED_MODEL` | `Qwen3-4B` | vLLM 提供的 model id |

## 5. Bench / 评估

训练后，wrapper 把新 LoRA 热加载到 vLLM 并在 5 个测试任务上跑 3-run bench。
结果落在：

```text
$BASE_DIR/bench_<round_tag>_v2_<timestamp>/result.json
```

（默认 `$BASE_DIR=/workspace/$EXPERIMENT`；覆盖 `BASE_DIR` 可使用任意可写目录。）

要在已保存的 LoRA 上手动跑 bench：

```bash
curl -X POST http://127.0.0.1:8021/v1/load_lora_adapter \
    -H "Content-Type: application/json" \
    -d '{"lora_name":"meeting-r1","lora_path":"<lora_path>"}'

# DEEPSEEK_API_KEY must be in env — the judge resolution chain
# (scripts/lib_grading.py:resolve_judge_backend_from_env) reads it directly.
# `--api-key dummy` below is for the vLLM endpoint, NOT the judge.
DEEPSEEK_API_KEY=$DEEPSEEK_API_KEY \
python scripts/benchmark.py \
    --suite "task_meeting_advisory_stakeholders,task_meeting_council_votes,task_meeting_gov_speaker_summary,task_meeting_tech_action_items,task_meeting_sentiment_analysis" \
    --model "custom/meeting-r1" \
    --base-url "http://127.0.0.1:8021/v1" \
    --api-key "dummy" \
    --judge "deepseek-chat" \
    --output-dir /workspace/bench_meeting_r1 \
    --runs 3
```

> **常见 401 陷阱。** 仅设置 `OPENAI_API_KEY=$DEEPSEEK_API_KEY` 是
> **不**够的 — `lib_grading.py` 中的 judge 解析链不读 `OPENAI_API_KEY`。
> 它检查 `PINCHBENCH_GRADE_JUDGE_API_KEY` → `JUDGE_API_KEY` →
> `DEEPSEEK_API_KEY`。最简单的办法是确保导出 `DEEPSEEK_API_KEY`
> （像在 `~/.pinchbench_env` 里那样）。

## 6. 诊断本次运行

bench 之后运行 diagnostics 模块得到 per-task 分析（失败模式、输出预算分配、
跨 run 的自动 check 稳定性）：

```bash
python -m agent_loop.diagnostics analyze \
    --result-json $BASE_DIR/<bench_dir>/result.json \
    --transcripts-dirs results/0071_transcripts \
    --output $BASE_DIR/<bench_dir>/diagnosis.md
```

`results/<NNNN>_transcripts/` 在 bench 步骤中由 `scripts/benchmark.py`
自动创建 — `<NNNN>` 是零填充的递增 job id（例如 `0071`），每次 `benchmark.py`
调用一个 folder，每个被评估的任务一个 `<task_id>.jsonl`。bench 后用
`ls results/` 找到最新 folder；将其（或多个，用于多 run merge）传给
`--transcripts-dirs`。

报告内容见 [`diagnostics.md`](diagnostics.md)。

## 7. 预期结果

参考分数（3-run 平均，judge = `deepseek-chat`）：

| 配置 | 总分 | 备注 |
|---|---|---|
| Baseline (rope=2, no LoRA) | **50.6%** | apples-to-apples baseline |
| Terminal-only, R5 LoRA | 55.0% | ~5 轮收敛 |
| **Terminal + PRM (additive judge-gate, R1)** | **57.24%** | 1 轮收敛 |

完整的 per-task 拆分和消融历史见 [`experiment_report.md`](experiment_report.md)。

## 8. 常见陷阱

- **rope mismatch**：训练用 rope=2 但 bench 用 rope=1（或反过来）会得出
  不一致的分数。两者必须 `factor=2.0, max_model_len=81920`。
- **单次 run 验证有噪声**：测试集始终用 3-run。
- **不要用默认配方继续 R2 之后**：R2-additive 因 reward hacking 回退 -1.9pp
  （模型把输出字符从 `.md` 文件转移到 chat reply）。Diagnostics 模块能捕获
  这个；见 `experiment_report.md` §15。
- **Workspace 覆盖**：rollout 共享 `/tmp/pinchbench/<NNNN>/agent_workspace`。
  每个 rollout 在下一个任务覆盖之前先快照自己的 workspace。
- **不要关闭 thinking**：Qwen3-4B 的 tool calling 依赖 `<think>...</think>`。
  关闭 thinking 会无声地破坏 rollout。
