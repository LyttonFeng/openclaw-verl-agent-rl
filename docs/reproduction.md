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
| **veRL** | **不需要** | 当前 GRPO 训练器（`train_meeting_grpo_step.py`）是自包含 PyTorch + transformers + peft 循环，**不导入** veRL。`rl/legacy/` 下的 monkeypatch 仅作历史参考，**也无需安装 veRL**。本文档复现路径完全跳过 veRL。 |
| vLLM | 0.10.2 | 配合 `VLLM_ALLOW_RUNTIME_LORA_UPDATING=True` 实现热加载 |
| Transformers | 4.57.1 | |
| PEFT | latest compatible | LoRA 训练 |
| Torch | 2.8.0+cu128 | 在 A100 上测试 |
| OpenClaw CLI | `2026.4.5` (3e72c03) | 在本地运行 multi-turn agent |
| PinchBench | 1.2.1 | 任务定义 + grader（本仓库已嵌入所需子集） |
| GPU | 2 × A100-80GB | GPU 0 = 训练，GPU 1 = vLLM |

> **不需要 ECS / 外部 runtime。** OpenClaw 在与训练相同的 pod 上本地运行。
> 之前 task16 路径用的 SSH-to-OpenClaw 模式在这里不再使用。

### ⚠️ 训练 dtype：必须用 fp32，不能用 bf16（2026-05-09 发现）

**现象（NaN 复现 trace）**：在 **transformers 4.57 + peft 0.19 + 80K context** 组合下，
bf16 训练会出现完全可复现的 NaN：

```
sample 5/30:    loss=0.0313  ✓
sample 10/30:   loss=0.0290  ✓
sample 11-17:   <无 print，但单 sample 时间从 16s/sample 飙升到 51s/sample>
                <某次 backward 产生 inf gradient，optimizer.step() 把 NaN 写入 LoRA 参数>
sample 18/30:   loss is NaN/inf, skipping  ← 第一次 NaN
sample 19-30:   loss is NaN/inf, skipping  ← 后续全 NaN（参数已污染）
Training done: 7 optimizer steps, 16 skipped, avg_loss=0.0053
LoRA saved   ← 但保存的是 NaN 参数
```

**验证**：训练后用 safetensors 直接读 LoRA `adapter_model.safetensors`，
所有 layer 的 `lora_A.weight` / `lora_B.weight` **全部** `tensor([nan, nan, nan, ...])`。
vLLM hot-load 这个 NaN LoRA 后，bench 第一个 task 第一 run 即 0%。

**根因**：transformers 4.57 的 attention 实现 + peft 0.19 的 LoRA forward path
+ bf16 + 长 context 在某些 batch 上数值溢出。同代码（diff 仅是后加的 NaN guard）
在 2026-05 老 SOTA pod 上的 transformers/peft 旧版本不出 NaN。

**修复（必须）**：`rl/train/train_meeting_grpo_step.py` 里：

1. Model load 用 fp32：
   ```python
   model = AutoModelForCausalLM.from_pretrained(
       args.model_path,
       torch_dtype=torch.float32,    # 不是 torch.bfloat16
       ...
   )
   ```
2. 去掉 bf16 autocast：
   ```python
   # 旧：with torch.amp.autocast("cuda", dtype=torch.bfloat16):
   # 新：直接 forward，不要 autocast wrapper
   body_out = body(input_ids=input_ids)
   ...
   ```

代价：fp32 训练比 bf16 慢约 2x（每 sample 16s → 25s）。
显存：4B fp32 + LoRA + 80K context activation + grad checkpointing
在 80GB A100 上够用，不会 OOM。

实测 R1 fp32（terminal-only，30 sample，lr=1e-6）：
- avg_loss=0.0071（正常，无 NaN）
- bench overall **46.16%**，相比 baseline_v6 (44.68%) **+1.48pp** ↑

未来如果切回 bf16，需要先确认：
- 装老版本 `transformers==4.45.2 peft==0.13.0`（疑似 SOTA 时代版本）
- 或自己实现 attention scores cast fp32 的 numerical safety

### 安装 Python 依赖

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 从源码安装 veRL（已弃用，**默认跳过**）

> **当前 GRPO 训练器（`train_meeting_grpo_step.py`）不 import veRL。**
> 本文档的所有 SOTA 数字都跑的是 `train_meeting_grpo_step.py`。仓库名
> `openclaw-verl-agent-rl` 是历史遗留，`rl/legacy/` 下的 verl_*_patch.py
> 也仅作历史归档，**没有任何当前路径需要装 veRL**。
>
> 跳过本节直接到 §"安装 OpenClaw CLI"。

如果你出于历史考古目的想装 veRL：

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

### 进阶：质量过滤 + PPO（推荐 setting）

R3 实验（详见 `experiment_report.md`）发现 vanilla PG 在 N=2 GRPO 下会
**race-to-bottom 退化**（judge 噪声让 lazy 答案拿正 advantage，模型学偷懒）。
解决方案：**质量过滤 + PPO 三件套**（importance ratio + clip + KL）。

完整一轮命令（多卡 fp32 + rope=2 + 6h chain 验证 setting）：

```bash
ROUND_DIR=/workspace/grpo_runs/meeting_grpo_v2/round_3
PREV_LORA=/workspace/grpo_runs/meeting_grpo_v2/round_2/checkpoint/lora_adapter

# 1. Rollouts（同前；vLLM 在 GPU1 服务）
# 略 — 用 generate_meeting_rollouts.py 或 wrapper script

# 2. PRM-skip + variance filter（同前）
python3 rl/train/select_grpo_samples.py \
  --graded-file $ROUND_DIR/rollouts/graded_trajectories_prm.jsonl \
  --output-dir  $ROUND_DIR/selection \
  --variance-threshold 1e-08 --alpha 1.0

# 3. 质量过滤（race-to-bottom 防御，详见 algorithm.md）
python3 rl/train/apply_quality_filter.py \
  --input  $ROUND_DIR/selection/graded_trajectories_prm_valid.jsonl \
  --output $ROUND_DIR/selection/graded_trajectories_quality_filtered.jsonl \
  --report $ROUND_DIR/selection/quality_report.json

# 4. Kill vLLM 释放 GPU1 给训练
pkill -f vllm.entrypoints.openai.api_server
sleep 8

# 5. 计算 P_old log_probs（PPO 必需）
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 rl/train/compute_rollout_logprobs.py \
  --graded-file $ROUND_DIR/selection/graded_trajectories_quality_filtered.jsonl \
  --lora-path   $PREV_LORA \
  --output      $ROUND_DIR/rollout_logprobs.jsonl \
  --max-seq-length 65536 --rope-scaling-factor 2.0
# R1 从 base 起：去掉 --lora-path 参数（compute 脚本会用 base only）

# 6. PPO + KL 训练
CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
python3 rl/train/train_meeting_grpo_step.py \
  --graded-file $ROUND_DIR/selection/graded_trajectories_quality_filtered.jsonl \
  --model-path  Qwen/Qwen3-4B \
  --lora-path   $PREV_LORA \
  --output-dir  $ROUND_DIR/checkpoint \
  --logprobs-file $ROUND_DIR/rollout_logprobs.jsonl \
  --clip-eps 0.2 --kl-beta 0.02 \
  --lr 1e-6 --lora-rank 16 --grad-accum-steps 2 \
  --max-seq-length 65536 --rope-scaling-factor 2.0 \
  --prm-alpha 1.0 --prm-beta 0 --prm-mode additive

# 7. 重启 vLLM、hot-load、bench（同前）
```

**关键 flag**：
- `--logprobs-file`：传入 P_old 文件即启用 PPO；不传则退回 vanilla PG（向后兼容）
- `--clip-eps 0.2`：PPO clip ε，标准值
- `--kl-beta 0.02`：KL 惩罚系数（实测 KL 落在 0.001-0.003 健康区间）
- `CUDA_VISIBLE_DEVICES=0,1`：多卡 device_map="auto" 是 fp32 + 17k+ tokens 的必要条件（单卡 OOM）
- `--rope-scaling-factor 2.0`：必须，与 vLLM rollout 一致

参考一键 chain：[`experiments/clean_chain_filter_ppo/chain_script.sh`](../experiments/clean_chain_filter_ppo/chain_script.sh)
（6 轮自动化脚本，含 OOM 重试 + 断点续跑 + vLLM lifecycle 管理）。

### 进阶 ++：开启 PRM (with reward gate + per-turn loss)

把 PRM 直接叠加到 [filter + PPO] 上**会退化**（v1 实证 -1.2pp）。两个 fix 配合
PRM 才能拿到正向增量（+0.4pp）。详见 `algorithm.md` § "PRM 与 [filter + PPO]
组合的实证设计要点"。

```bash
# Step 2.5：在 PRM-skip 替换为真正 PRM scoring
PYTHONPATH=$REPO_ROOT python3 \
  $REPO_ROOT/agent_loop/roadmap_prm/scripts/score_trajectories.py \
  --graded-file $ROUND_DIR/rollouts/graded_trajectories.jsonl \
  --tasks-dir   $REPO_ROOT/pinchbench_tasks/meeting_analysis \
  --roadmaps-dir $REPO_ROOT/agent_loop/roadmap_prm/roadmaps \
  --output-suffix _prm \
  --max-workers 4
# → graded_trajectories_prm.jsonl with real {-1, 0, +1} per turn

# Step 3：select 同前

# Step 4：quality filter 加 reward gate
python3 rl/train/apply_quality_filter.py \
  --input  $ROUND_DIR/selection/graded_trajectories_prm_valid.jsonl \
  --output $ROUND_DIR/selection/graded_trajectories_quality_filtered.jsonl \
  --report $ROUND_DIR/selection/quality_report.json \
  --prm-reward-gate 0.5    # NEW: score >= 0.5 的 trajectory 清零 PRM

# Step 6：训练加 PRM + per-turn loss
python3 rl/train/train_meeting_grpo_step.py \
  ... (跟前面 PPO 一样) \
  --prm-alpha 1.0 --prm-beta 0.5 --prm-mode multiplicative \
  --per-turn-loss      # NEW: 消除 PRM "长 turn 多放大" 偏置
```

**新 flag 解读**：

| Flag | 文件 | 作用 |
|---|---|---|
| `--prm-reward-gate 0.5` | apply_quality_filter | score≥0.5 的 trajectory 直接清空 prm_turn_scores（避免 PRM 干扰已及格的） |
| `--per-turn-loss` | train_meeting_grpo_step | 每个 token 权重 = 1/n_tokens_in_its_turn，每个 turn 等权 |
| `--prm-mode multiplicative` | 同上 | 只放大正 advantage，避免 additive 翻转符号风险 |
| `--prm-beta 0.5` | 同上 | multiplicative 公式下，PRM=+1 turn ×1.5（不是 docs 默认 1.0；实测 0.5 在我们 baseline 下最稳） |

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

### 2026-05 复现 pod（推荐参考）

| 配置 | MEETING % | 关键设计 |
|---|---:|---|
| **base Qwen3-4B (rope=2 / 64K, fp32)** | **44.68** | apples-to-apples baseline |
| 老 R1 (vanilla GRPO) | 46.20 | terminal-only |
| R2 (vanilla GRPO，平台) | 46.40 | 已饱和 |
| R3 v1 (续训 vanilla) | **43.30** ↓ | **退化**（race-to-bottom） |
| R3 v2 (+ 质量过滤) | 46.20 | 止退化 |
| **R3 v3 (+ PPO + KL)** | **47.50** | 单轮破 R2 |
| **R4' (clean chain [filter + PPO])** | **47.80** 🏆 | 6 轮 chain 峰值 |
| R1' + PRM v1 (naive) | 45.69 ↓ | 失败：PRM 拖累强项 |
| **R1' + PRM v2 (+ reward gate + per-turn loss)** | **47.80** | 修复后单轮持平峰值 |

如果你用本文件 §4 "进阶 ++" 的 PRM with reward gate + per-turn loss 配方跑，
**单轮预期得到 47-48% 之间的总分**。

### 历史 SOTA pod 数据（仅作设计参考，不可比）

> ⚠️ **不要用这个表来验证你的复现**——它来自 2026-04 之前的旧 pod (bf16 + 80K
> + 旧 transformers/peft)。新 pod 必须 fp32 + 64K，base 重跑 = 44.68% 而不是 50.6%。

| 配置 | 老 pod 总分 |
|---|---|
| Baseline (rope=2, no LoRA) | 50.6% |
| Terminal-only, R5 LoRA | 55.0% |
| **Terminal + PRM (additive judge-gate, R1)** | **57.24%** |

完整 per-task 拆分和消融历史见 [`experiment_report.md`](experiment_report.md)。

## 8. 常见陷阱

- **不要用 bf16 训练**：新 transformers 4.57 + peft 0.19 + 长 context 必出 NaN
  （详见 §1 dtype 警告）。必须 fp32。
- **rope mismatch**：训练用 rope=2 但 bench 用 rope=1（或反过来）会得出
  不一致的分数。两者必须 `factor=2.0`。
- **单卡装不下 fp32 + 17k+ tokens 训练**：multi-GPU device_map="auto" 必需，
  CUDA_VISIBLE_DEVICES=0,1（详见进阶节）。
- **vanilla GRPO 在 R3 会 race-to-bottom 退化** —— 必须配合质量过滤
  ([`apply_quality_filter.py`](../rl/train/apply_quality_filter.py))。
- **质量过滤后单加 PRM 会退化**：v1 naive PRM 在 [filter + PPO] 上 -1.2pp。
  必须配合 `--prm-reward-gate 0.5` + `--per-turn-loss`（详见 §4 进阶 ++）。
- **clean chain 跑到 R4-R5 就停**，继续训会过训退化（见 experiment_report.md §3.4）。
- **单次 run 验证有噪声**：测试集始终用 3-run；5-task × 3-run = ±1pp 噪声。
- **Workspace 覆盖**：rollout 共享 `/tmp/pinchbench/<NNNN>/agent_workspace`。
- **不要关闭 thinking**：Qwen3-4B 的 tool calling 依赖 `<think>...</think>`。
