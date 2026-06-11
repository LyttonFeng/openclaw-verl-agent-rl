# env

This directory documents the training and benchmark environment.

It should contain only environment-level reproducibility information, such as dependency versions, model serving settings, OpenClaw configuration, vLLM configuration, hardware assumptions, and benchmark runtime variables.

Current files:

- `requirements.txt`: minimal Python package constraints for rollout generation, vLLM serving, logprob recomputation, and PyTorch/PEFT training.
- `training_environment.md`: training runtime requirements and the end-to-end naive PPO-style round entrypoint.
- `benchmark_environment.md`: isolated Val3 benchmark runtime requirements and example commands for API/vLLM baselines.
- `qwen35_environment.md`: Qwen3.5-4B (non-think) serving contract, package versions (vLLM 0.22 / transformers 5.9 / peft 0.19 / torch 2.11), the local-disk-only weights rule, and the Val3 baseline (75.2%).
