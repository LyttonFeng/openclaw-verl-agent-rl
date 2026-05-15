# openclaw-async v17 breakthrough (2026-05-15)

veRL fully-async-policy + OpenClaw GRPO 训练首次跑通，ckpt_4 (step 8 / 64 trajectories) bench mean=46.5% on 5 meeting test tasks × 3 runs，超 Qwen3-4B base 44.68% (+1.8pp)。

## 训练侧 4 个 env-overridable 改动（本 commit）

```bash
AGENT_NUM_WORKERS=4 \
OUTPUT_DIR=/workspace/verl_port/ckpt_openclaw_async_v17 \
RESUME_MODE=disable \
MAX_PROMPT_LENGTH=64000 \
MAX_RESPONSE_LENGTH=14000 \
VLLM_MAX_MODEL_LEN=81920 \
VLLM_MAX_NUM_SEQS=2 \
MAX_TURNS=8 \
STALENESS_THRESHOLD=0.3 \
TOTAL_EPOCHS=10 \
EXPERIMENT_NAME=openclaw_async_v17 \
bash experiments/verl_port_poc/launch_meeting_openclaw_async.sh
```

### 关键 env 作用

| Env | 作用 |
|---|---|
| **`AGENT_NUM_WORKERS=4`** | **真凶**: 默认 1 个 AgentLoopWorker 进程跑 16 并发 OC subproc，asyncio event loop 卡死。4 worker 分散后 HTTP path 不 deadlock。从 0 突破到 102 Got proxy / 22 Turn 1+。 |
| `OUTPUT_DIR` | env 化让多 run 用独立 ckpt dir，避免 step 序号冲突 |
| `RESUME_MODE=disable` | `PINCHBENCH_LORA_ONLY_CKPT=1` 只存 adapter；verl resume 需要 optimizer state 会 FileNotFoundError。fresh start 避免 |
| `PINCHBENCH_RL_VLLM_HTTP_MODEL=$MODEL` | agent_loop 的 HTTP tool-parser 路径默认请求 `Qwen/Qwen3-4B`，vLLM `served_model_name` 是完整 snapshot 路径，404 mismatch。设全路径修了 |
| `VLLM_MAX_MODEL_LEN=81920` | Qwen3-4B `config.json` 自带 `rope_scaling=yarn factor=2.0`（native 40960 → 81920），launcher 用满 |
| `MAX_PROMPT_LENGTH=64000` | 配合 80K 上下文，meeting transcript + OpenClaw system prompt 装得下 |
| `VLLM_MAX_NUM_SEQS=2` | 比默认 16 稳定，减少 KV cache fragment + vLLM scheduler 排队，避免 EngineCore crash |

## verl 内核 patch（仍在 pod-side /root/verl/，未 commit 到上游）

`fully_async_rollouter.py:643` 加一行 — empty-mask drop 不应该消耗 staleness 名额：

```python
if _valid == 0:
    print(f"[RolloutFilter] dropping empty trajectory ...", flush=True)
    self.dropped_stale_samples += 1
    self.processed_sample_count += 1
    # PATCH: empty trajectories shouldn't count toward max_required_samples
    self.staleness_samples = max(0, self.staleness_samples - 1)
    return
```

不打这个 patch：empty-mask drop 计入 staleness_samples，max_required=20 几个 OC timeout 后就满 → rollouter pause 等 trainer step → trainer 等 MQ 凑 8 sample → 死锁（v9 实测）。

## bench 套路（用 ckpt_4 复现 46.5%）

```bash
# 1) standalone vLLM exposing LoRA as separate model id
CUDA_VISIBLE_DEVICES=0 vllm serve /root/hf_cache/hub/models--Qwen--Qwen3-4B/snapshots/<sha> \
  --port 8021 --served-model-name Qwen3-4B-base \
  --enable-lora \
  --lora-modules meeting-ckpt4=/workspace/verl_port/ckpt_openclaw_async_v17/global_step_4/actor/lora_adapter \
  --max-model-len 81920 --gpu-memory-utilization 0.85 \
  --enable-auto-tool-choice --tool-call-parser hermes --max-num-seqs 8

# 2) benchmark.py with --base-url so it auto-generates custom provider
cd /workspace/openclaw-verl-agent-rl && source /root/.pinchbench_env
export PINCHBENCH_ALLOW_LOCAL_OPENCLAW=1 PINCHBENCH_SKIP_OPENCLAW_WEB_PREFLIGHT=1 \
       OPENCLAW_HOST=localhost PINCHBENCH_DIR=/workspace/openclaw-verl-agent-rl
SUITE='task_meeting_advisory_stakeholders,task_meeting_council_votes,task_meeting_gov_speaker_summary,task_meeting_tech_action_items,task_meeting_sentiment_analysis'
python3 scripts/benchmark.py \
  --model custom/meeting-ckpt4 \
  --base-url http://127.0.0.1:8021/v1 --api-key dummy \
  --suite $SUITE --runs 3 --judge deepseek-chat \
  --no-upload --no-fail-fast --output-dir /path/to/output
```

### bench 必坑
- 不传 `--base-url` → 自动 create 的 agent 只 deepseek provider，没 custom → OC 调不到 vLLM → 全 0 分
- 不传 `OPENCLAW_HOST=localhost` → benchmark.py 默认 SSH 到 ECS 公网 IP → permission denied
- 上一个 vLLM crash 不彻底，子进程 EngineCore 占 ~70GB GPU memory；SIGKILL specific PID 再起新 vLLM

## bench 完整分数 (ckpt_4 = step 8 = 64 trajectories trained)

| Task | Run 1 | Run 2 | Run 3 | Mean |
|---|---|---|---|---|
| advisory_stakeholders | 47% | 42% | 47% | 45.3% |
| council_votes | 21% | 26% | 20% | 22.3% |
| gov_speaker_summary | 41% | 44% | 41% | 42.0% |
| tech_action_items | 61% | 66% | 51% | 59.3% |
| sentiment_analysis | 60% | 62% | 69% | 63.7% |
| **mean** | | | | **46.5%** |

| 模型 | 5-task mean |
|---|---|
| Qwen3-4B base (no LoRA) | 44.68% |
| **ckpt_4 (async openclaw GRPO, step 8)** | **46.5%** |
| OpenClaw sync GRPO (24+ step baseline) | 47.80% |

## v1 → v17 的失败路径回顾

| 版本 | 关键改动/Bug | 状态 |
|---|---|---|
| v1-v3 | ModelProxy SSE bug + Hydra MFS EIO + OC timeout | 全卡 Turn 0 |
| v4 | `PINCHBENCH_RL_USE_VLLM_HTTP_TOOL_PARSER=1` + HTTP_MODEL=$MODEL | 22 Turn 1+/6 MQ samples，staleness 死锁 |
| v5-v8 | 调 staleness_threshold/prompt_length 试错 | 都没出 step |
| v9-v13 | verl staleness patch + max_num_seqs=2 + num_workers 试错 | v9 出 step 1 后 vLLM EngineCore crash |
| v14 | **`num_workers=4`** + 上述 patches | step 1 ✅ + ckpt_1 ✅ 后 ProcessLookupError 停 |
| v15 | 同 v14 retry | step 2 + ckpt_1 ✅ 后 EngineCore 不死，但 ProcessLookupError 中途停 |
| v16 | resume_mode=auto | 启动 crash（LoRA-only ckpt 缺 optimizer state） |
| **v17** | resume_mode=disable + 独立 OUTPUT_DIR | **step 8 + ckpt_4 = 46.5%** ✅ |
