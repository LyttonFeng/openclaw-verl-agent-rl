# train

This directory contains the core training implementation.

It should contain code required to run the meeting-analysis policy training loop, including rollout generation, reward integration, trainer entrypoints, and model adaptation logic.

Current files:

- `train_meeting_grpo_step.py`: self-contained PyTorch/PEFT trainer for meeting-analysis LoRA updates. It computes GRPO-style advantages from graded rollouts and can run either vanilla policy-gradient updates or PPO-style ratio/clip/KL updates when old-policy logprobs are provided.
- `compute_rollout_logprobs.py`: offline utility for computing `P_old` trainable-token logprobs from existing rollout transcripts. This is required by `train_meeting_grpo_step.py` when PPO-style clipping is enabled.
- `generate_meeting_rollouts.py`: rollout sampler. It runs OpenClaw tasks against a vLLM OpenAI-compatible endpoint, snapshots workspace outputs/transcripts, and writes `graded_trajectories.jsonl`.
- `select_grpo_samples.py`: dynamic signal filter. It drops task groups with no score variance and can drop obviously unusable trajectories before training.
- `run_naive_ppo_round.sh`: minimal end-to-end round wrapper: rollout sampling, dynamic filtering, `P_old` logprob recomputation, and PPO-style LoRA update.

Pod slim12 example:

```bash
PYTHON_BIN=/root/openclaw-venv/bin/python \
MODEL_PATH=/workspace/qwen_models/qwen3-4b \
ROLLOUT_MODEL=Qwen3-4B-base \
VLLM_BASE_URL=http://127.0.0.1:8021/v1 \
TRAIN_SPLIT=data/train/meeting_analysis_slim12_split.json \
N_RESPONSES=4 \
NUM_WORKERS=1 \
bash train/run_naive_ppo_round.sh
```
