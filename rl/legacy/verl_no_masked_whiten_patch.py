"""Patch veRL RF++ advantage to skip masked_whiten for agent token rewards.

PinchBench agent rewards are already task-EMA centered before they enter veRL.
Applying veRL's default masked_whiten over every response token can distort
sparse or span-broadcast rewards by creating unrelated negative token signals.
"""

from __future__ import annotations

import os


def _enabled() -> bool:
    return os.environ.get("PINCHBENCH_NO_MASKED_WHITEN", "1").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def apply_patch() -> None:
    if not _enabled():
        print("[pinchbench_no_masked_whiten] disabled")
        return

    try:
        import torch
        import verl.trainer.ppo.core_algos as core_algos
    except Exception as exc:
        print(f"[pinchbench_no_masked_whiten] import skipped: {exc}")
        return

    if getattr(core_algos, "_pinchbench_no_masked_whiten_patched", False):
        return

    def compute_reinforce_plus_plus_no_whiten(
        token_level_rewards: torch.Tensor,
        response_mask: torch.Tensor,
        config=None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert config is not None
        gamma = config.gamma
        with torch.no_grad():
            returns = torch.zeros_like(token_level_rewards)
            running_return = 0

            for t in reversed(range(token_level_rewards.shape[1])):
                running_return = token_level_rewards[:, t] + gamma * running_return
                returns[:, t] = running_return
                running_return = running_return * response_mask[:, t]

            advantages = returns * response_mask

        return advantages, returns

    name = core_algos.AdvantageEstimator.REINFORCE_PLUS_PLUS.value
    core_algos.ADV_ESTIMATOR_REGISTRY[name] = compute_reinforce_plus_plus_no_whiten
    core_algos.compute_reinforce_plus_plus_outcome_advantage = (
        compute_reinforce_plus_plus_no_whiten
    )
    core_algos._pinchbench_no_masked_whiten_patched = True
    print("[pinchbench_no_masked_whiten] patched reinforce_plus_plus advantage")


apply_patch()
