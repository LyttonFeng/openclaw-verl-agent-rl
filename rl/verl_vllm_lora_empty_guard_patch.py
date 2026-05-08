"""Guard vLLM LoRA merge against empty adapter sets.

Some veRL/vLLM combinations can reach `LoRAModelManager._create_merged_loras_inplace`
with an empty `lora_model.loras` mapping during weight sync. Upstream code assumes
at least one LoRA tensor exists and crashes on `next(iter(...))`.

For our training loop, that crash is worse than a no-op because it stops the run
before we can inspect whether the rollout path is otherwise healthy.
"""

from __future__ import annotations


def apply_patch() -> None:
    _patch_verl_lora_collection_debug()
    _patch_verl_vllm_update_debug()

    try:
        import vllm.lora.model_manager as lora_model_manager
    except Exception as e:  # pragma: no cover
        print(f"[vllm_lora_empty_guard] import skipped: {e}")
        return

    original = lora_model_manager.LoRAModelManager._create_merged_loras_inplace
    if getattr(original, "_pinchbench_empty_guard_patched", False):
        return

    def _patched_create_merged_loras_inplace(self, lora_model):
        try:
            if not getattr(lora_model, "loras", None):
                # Empty LoRA is expected at training start (freshly initialized).
                # Just skip the merge; vLLM will use base weights for this rollout.
                return
            return original(self, lora_model)
        except StopIteration:
            raise RuntimeError(
                "vLLM LoRA merge hit StopIteration; upstream LoRA collection path is inconsistent"
            )

    _patched_create_merged_loras_inplace._pinchbench_empty_guard_patched = True
    lora_model_manager.LoRAModelManager._create_merged_loras_inplace = (
        _patched_create_merged_loras_inplace
    )
    print("[vllm_lora_empty_guard] applied")


def _patch_verl_lora_collection_debug() -> None:
    try:
        import verl.utils.fsdp_utils as fsdp_utils
        import verl.workers.fsdp_workers as fsdp_workers
    except Exception as e:  # pragma: no cover
        print(f"[vllm_lora_empty_guard] veRL fsdp debug skipped: {e}")
        return

    original = fsdp_utils.collect_lora_params
    if getattr(original, "_pinchbench_lora_debug_patched", False):
        return

    def _patched_collect_lora_params(module, layered_summon, base_sync_done):
        peft_model = getattr(module, "_fsdp_wrapped_module", module)
        peft_config = getattr(peft_model, "peft_config", None)
        default_cfg = peft_config.get("default") if isinstance(peft_config, dict) else None
        print(
            "[vllm_lora_debug] collect_lora_params start "
            f"layered_summon={layered_summon} base_sync_done={base_sync_done} "
            f"module={type(module).__name__} peft_model={type(peft_model).__name__} "
            f"has_peft_config={peft_config is not None} "
            f"r={getattr(default_cfg, 'r', None)} "
            f"target_modules={getattr(default_cfg, 'target_modules', None)}"
        )
        params = original(module, layered_summon, base_sync_done)
        keys = list(params.keys())
        sample = keys[:8]
        print(
            "[vllm_lora_debug] collect_lora_params done "
            f"count={len(keys)} sample={sample}"
        )
        if layered_summon and base_sync_done and not keys:
            names = [name for name, _ in list(module.named_modules())[:80]]
            print(f"[vllm_lora_debug] first_named_modules={names}")
        return params

    _patched_collect_lora_params._pinchbench_lora_debug_patched = True
    fsdp_utils.collect_lora_params = _patched_collect_lora_params
    # fsdp_workers imports collect_lora_params by value, so update that reference too.
    fsdp_workers.collect_lora_params = _patched_collect_lora_params
    print("[vllm_lora_debug] patched collect_lora_params")


def _patch_verl_vllm_update_debug() -> None:
    try:
        import os
        from verl.workers.rollout.vllm_rollout import utils as vllm_utils
    except Exception as e:  # pragma: no cover
        print(f"[vllm_lora_empty_guard] vLLM update debug skipped: {e}")
        return

    cls = vllm_utils.vLLMColocateWorkerExtension
    original = cls._update_weights
    if getattr(original, "_pinchbench_lora_debug_patched", False):
        return

    def _patched_update_weights(self, weights, peft_config, base_sync_done):
        materialized = list(weights)
        sample = [name for name, _ in materialized[:8]]
        print(
            "[vllm_lora_debug] vLLM _update_weights "
            f"count={len(materialized)} base_sync_done={base_sync_done} "
            f"has_peft_config={peft_config is not None} sample={sample}"
        )
        if os.environ.get("PINCHBENCH_SKIP_VLLM_UPDATE_WEIGHTS", "").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            print("[vllm_lora_debug] skipping vLLM _update_weights by PINCHBENCH_SKIP_VLLM_UPDATE_WEIGHTS")
            return None
        return original(self, materialized, peft_config, base_sync_done)

    _patched_update_weights._pinchbench_lora_debug_patched = True
    cls._update_weights = _patched_update_weights
    print("[vllm_lora_debug] patched vLLM _update_weights")


apply_patch()
