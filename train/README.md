# train

This directory contains the core training implementation.

It should contain code required to run the meeting-analysis policy training loop, including rollout generation, reward integration, trainer entrypoints, and model adaptation logic.

Current files:

- `train_meeting_grpo_step.py`: self-contained PyTorch/PEFT trainer for meeting-analysis LoRA updates. It computes GRPO-style advantages from graded rollouts and can run either vanilla policy-gradient updates or PPO-style ratio/clip/KL updates when old-policy logprobs are provided.
- `compute_rollout_logprobs.py`: offline utility for computing `P_old` trainable-token logprobs from existing rollout transcripts. This is required by `train_meeting_grpo_step.py` when PPO-style clipping is enabled.
