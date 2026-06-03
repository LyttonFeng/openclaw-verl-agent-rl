# Training Environment

This file documents the environment used for the naive meeting-analysis RL training loop.

The concrete reference environment is the current benchmark/training pod:

```text
ssh root@154.54.102.37 -p 15877 -i ~/.ssh/id_ed25519
```

## Hardware

Reference pod:

- GPU: NVIDIA A100-SXM4-80GB
- Driver: 580.126.16
- CUDA reported by `nvidia-smi`: 13.0
- Hostname observed during capture: `4438c54f6db2`

For Qwen3-4B long-context rollout and logprob recomputation, use an A100/H100 class GPU. The reference pod uses a single A100 80GB.

Recommended runtime env:

```bash
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

## Python

Reference pod virtualenv:

```bash
source /root/openclaw-venv/bin/activate
python --version
pip install -r env/requirements.txt
```

Observed versions:

- Python: 3.10.12
- `torch`: 2.11.0+cu130
- `transformers`: 5.9.0
- `peft`: 0.19.1
- `vllm`: 0.22.0
- `accelerate`: 1.13.0
- `pyyaml`: 6.0.3
- `huggingface_hub`: 1.17.0

The system `python3` on the pod is not the training environment. Use `/root/openclaw-venv/bin/python`.

## Model Serving

Training rollouts use OpenClaw as the agent runtime, and OpenClaw calls a vLLM OpenAI-compatible endpoint.

Canonical Qwen3-4B server for this branch:

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

The reference pod may have other vLLM processes running. During environment capture, one active process served `/workspace/qwen_models/qwen3-4b-instruct-2507` as `qwen3-4b-instruct-2507` on port `8767`; that is useful for debugging the pod, but it is not the canonical default for this branch.

Confirm the server:

```bash
curl -s http://127.0.0.1:8021/v1/models
```

`train/run_naive_ppo_round.sh` should be run with the same served model name and endpoint:

```bash
VLLM_BASE_URL=http://127.0.0.1:8021/v1
ROLLOUT_MODEL=Qwen3-4B
MODEL_PATH=Qwen/Qwen3-4B
```

## OpenClaw

OpenClaw CLI must be installed and available on `PATH`.

Reference pod version:

```text
OpenClaw 2026.4.5 (3e72c03)
```

Check locally:

```bash
openclaw --version
```

The rollout sampler creates one OpenClaw agent per worker and configures each agent with a custom OpenAI-compatible provider pointed at `VLLM_BASE_URL`.

Relevant implementation:

- `train/generate_meeting_rollouts.py`: creates worker agents and calls `execute_openclaw_task`.
- `scripts/lib_agent.py`: writes OpenClaw `models.json` and invokes `openclaw agent ... --local`.

## Secrets

The judge uses DeepSeek/OpenAI-compatible API credentials:

```bash
export DEEPSEEK_API_KEY=...
```

Optional env file:

```bash
cat > ~/.pinchbench_env <<'EOF'
export DEEPSEEK_API_KEY=REPLACE_ME
EOF
```

## End-To-End Round

Run one naive PPO-style round:

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

Main artifacts:

- `results/train/<RUN_ID>/rollouts/graded_trajectories.jsonl`
- `results/train/<RUN_ID>/selection/graded_trajectories_prm_valid.jsonl`
- `results/train/<RUN_ID>/rollout_logprobs.jsonl`
- `results/train/<RUN_ID>/checkpoint/lora_adapter`
