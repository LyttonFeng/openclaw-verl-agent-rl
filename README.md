# Naive Meeting Analysis RL

Minimal branch for running the meeting-analysis naive RL loop and isolated Val5 benchmark.

The branch intentionally keeps only the files needed for:

1. generating OpenClaw rollouts,
2. filtering rollouts with no useful training signal,
3. recomputing rollout-time `P_old` logprobs,
4. applying a PyTorch/PEFT PPO-style LoRA update,
5. reproducing the isolated Val5 temperature-0 benchmark.

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

## Start Qwen3-4B Serving

This branch is centered on training `Qwen/Qwen3-4B`.

```bash
CUDA_VISIBLE_DEVICES=0 VLLM_ALLOW_RUNTIME_LORA_UPDATING=True \
/root/openclaw-venv/bin/python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-4B \
  --served-model-name Qwen3-4B \
  --host 127.0.0.1 \
  --port 8021 \
  --max-model-len 65536 \
  --enable-auto-tool-choice \
  --tool-call-parser hermes
```

Check serving:

```bash
curl -s http://127.0.0.1:8021/v1/models
```

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

## Run Isolated Val5 Benchmark

Qwen3-4B local vLLM baseline:

```bash
source /root/openclaw-venv/bin/activate
source ~/.pinchbench_env

RUN_ID=qwen3_4b_temp0 \
MODEL=Qwen3-4B \
BASE_URL=http://127.0.0.1:8021/v1 \
PINCHBENCH_MODEL_API_KEY=dummy \
PINCHBENCH_MODEL_TEMPERATURE=0 \
OUTPUT_DIR=results/val5_isolated/qwen3_4b_temp0 \
bash scripts/run_val5_bench_isolated.sh
```

DeepSeek API baseline example:

```bash
RUN_ID=dsv4_pro_temp0 \
MODEL=deepseek-v4-pro \
BASE_URL=https://api.deepseek.com/v1 \
PINCHBENCH_MODEL_TEMPERATURE=0 \
OUTPUT_DIR=results/val5_isolated/dsv4_pro_temp0 \
bash scripts/run_val5_bench_isolated.sh
```

Tracked baseline table:

- `results/isolated_val5_temp0_baseline_results.md`

## Where The Logic Lives

- `train/generate_meeting_rollouts.py`: OpenClaw rollout sampling and grading.
- `train/select_grpo_samples.py`: dynamic filter for no-signal or bad trajectories.
- `train/compute_rollout_logprobs.py`: recomputes `P_old` logprobs.
- `train/train_meeting_grpo_step.py`: PyTorch/PEFT PPO-style LoRA update.
- `train/run_naive_ppo_round.sh`: end-to-end training round wrapper.
- `scripts/run_val5_bench_isolated.sh`: isolated Val5 benchmark wrapper.
- `scripts/lib_agent.py`: OpenClaw agent config and CLI execution against vLLM/OpenAI-compatible endpoints.
- `rewards/meeting_reward.py`: meeting-analysis terminal reward helper.
