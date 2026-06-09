# env

This directory documents the training and benchmark environment.

It should contain only environment-level reproducibility information, such as dependency versions, model serving settings, OpenClaw configuration, vLLM configuration, hardware assumptions, and benchmark runtime variables.

Current files:

- `requirements.txt`: minimal Python package constraints for rollout generation, vLLM serving, logprob recomputation, and PyTorch/PEFT training.
- `training_environment.md`: training runtime requirements and the end-to-end naive PPO-style round entrypoint.
- `benchmark_environment.md`: isolated Val3 benchmark runtime requirements and example commands for API/vLLM baselines.
