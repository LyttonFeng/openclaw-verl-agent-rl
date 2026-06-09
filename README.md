# Naive Meeting Analysis RL

Minimal branch for running the meeting-analysis naive RL loop and isolated Val3 benchmark.

The branch intentionally keeps only the files needed for:

1. generating OpenClaw rollouts,
2. filtering rollouts with no useful training signal,
3. recomputing rollout-time `P_old` logprobs,
4. applying a PyTorch/PEFT PPO-style LoRA update,
5. reproducing the isolated Val3 temperature-0 benchmark.

## Reference Environment

The reference pod used to document this branch:

```bash
ssh root@154.54.102.37 -p 15877 -i ~/.ssh/id_ed25519
```

Use the pod virtualenv, not system Python:

```bash
source /root/openclaw-venv/bin/activate
source ~/.pinchbench_env
openclaw --version
```

Environment details and package pins:

- `env/training_environment.md`
- `env/benchmark_environment.md`
- `env/requirements.txt`

Optional preflight:

```bash
bash scripts/check_repro_env.sh
```

If vLLM is not started yet, skip only the endpoint check:

```bash
CHECK_VLLM=0 bash scripts/check_repro_env.sh
```

## Start Qwen3-4B Serving

This branch is centered on training `Qwen/Qwen3-4B`.

```bash
source /root/openclaw-venv/bin/activate
CUDA_VISIBLE_DEVICES=0 bash scripts/start_qwen3_vllm.sh
```

Check serving:

```bash
curl -s http://127.0.0.1:8021/v1/models
```

The serving wrapper uses vLLM `--tool-call-parser hermes` and applies `scripts/apply_oc_hermes_patch.sh` by default. That OpenClaw patch extracts `<tool_call>...</tool_call>` fallback text into executable OpenClaw tool calls for Qwen3 multi-turn sessions.

The captured pod also had a Qwen3-4B-family service on port `8767`; that is a pod-specific process, not the canonical default for this branch.

## Run One Naive PPO-Style Round

The training split intentionally uses all meeting-analysis samples: historical train23 plus the former Val5 tasks. This is for checking whether naive RL can overfit and converge, not for held-out generalization.

```bash
source /root/openclaw-venv/bin/activate
source ~/.pinchbench_env

RUN_ID=naive_overfit_r1 \
N_RESPONSES=2 \
NUM_WORKERS=1 \
VLLM_BASE_URL=http://127.0.0.1:8021/v1 \
ROLLOUT_MODEL=Qwen3-4B \
MODEL_PATH=Qwen/Qwen3-4B \
bash train/run_naive_ppo_round.sh
```

Main outputs:

- `results/train/<RUN_ID>/rollouts/graded_trajectories.jsonl`
- `results/train/<RUN_ID>/selection/graded_trajectories_prm_valid.jsonl`
- `results/train/<RUN_ID>/rollout_logprobs.jsonl`
- `results/train/<RUN_ID>/checkpoint/lora_adapter`

## Run Isolated Val3 Benchmark

Qwen3-4B local vLLM baseline:

```bash
source /root/openclaw-venv/bin/activate
source ~/.pinchbench_env

RUN_ID=qwen3_4b_base_val3_temp0 \
MODEL=Qwen3-4B-base \
BASE_URL=http://127.0.0.1:8021/v1 \
PINCHBENCH_MODEL_API_KEY=dummy \
PINCHBENCH_MODEL_TEMPERATURE=0 \
OUTPUT_DIR=results/val3_isolated/qwen3_4b_base_val3_temp0 \
bash scripts/run_val3_bench_isolated.sh
```

DeepSeek API baseline example:

```bash
RUN_ID=dsv4_pro_temp0 \
MODEL=deepseek-v4-pro \
BASE_URL=https://api.deepseek.com/v1 \
PINCHBENCH_MODEL_TEMPERATURE=0 \
OUTPUT_DIR=results/val3_isolated/dsv4_pro_temp0 \
bash scripts/run_val3_bench_isolated.sh
```

Tracked baseline table:

- `results/isolated_val3_temp0_baseline_results.md`

## Where The Logic Lives

- `train/generate_meeting_rollouts.py`: OpenClaw rollout sampling and grading.
- `train/select_grpo_samples.py`: dynamic filter for no-signal or bad trajectories.
- `train/compute_rollout_logprobs.py`: recomputes `P_old` logprobs.
- `train/train_meeting_grpo_step.py`: PyTorch/PEFT PPO-style LoRA update.
- `train/run_naive_ppo_round.sh`: end-to-end training round wrapper.
- `scripts/check_repro_env.sh`: quickstart preflight checker.
- `scripts/start_qwen3_vllm.sh`: canonical Qwen3-4B vLLM serving wrapper.
- `scripts/apply_oc_hermes_patch.sh`: OpenClaw fallback parser for Qwen3 `<tool_call>` text.
- `scripts/run_val3_bench_isolated.sh`: isolated Val3 benchmark wrapper.
- `scripts/lib_agent.py`: OpenClaw agent config and CLI execution against vLLM/OpenAI-compatible endpoints.
- `rewards/meeting_reward.py`: meeting-analysis terminal reward helper.
