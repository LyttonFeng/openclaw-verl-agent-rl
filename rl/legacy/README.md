# Legacy veRL 补丁（已弃用）

本目录的 `verl_*_patch.py` 和 `transformers_qwen3_5_patch.py` 是早期 veRL-based PPO
路径上的 monkeypatch 遗留物，**当前 SOTA 训练不依赖任何一个**：

- **当前 GRPO 训练器** = `rl/train/train_meeting_grpo_step.py`，自包含 PyTorch +
  transformers + peft 循环（~250 行），不 import `verl`。
- 这些 patch 仅当你**也想跑老的 `rl/train/launch_main_ppo.py` 或 `run_reinforce_lora.sh`**
  时才需要 — 但那些脚本不复现本仓库的 SOTA 结果（47.80% peak）。

如果你只想复现/移植当前算法（filter + PPO + KL，详见 `docs/algorithm.md`），
**完全可以无视本目录**。
