"""Patch veRL generation config loading for Qwen3.5 checkpoints.

Qwen3.5-2B on this stack can load the model config, but veRL's
get_generation_config() may fail because the checkpoint has no
generation_config.json and GenerationConfig.from_model_config() can trip over
the nested config layout.
"""

from __future__ import annotations


def apply_patch() -> None:
    try:
        import verl.utils.model as verl_model
        from transformers import GenerationConfig
    except Exception as e:  # pragma: no cover
        print(f"[qwen3_5_generation_patch] import skipped: {e}")
        return

    if getattr(verl_model.get_generation_config, "_pinchbench_qwen3_5_patched", False):
        return

    original = verl_model.get_generation_config
    original_from_model_config = GenerationConfig.from_model_config

    if not getattr(GenerationConfig.from_model_config, "_pinchbench_qwen3_5_patched", False):
        def _patched_from_model_config(cls, model_config):
            try:
                return original_from_model_config(model_config)
            except AttributeError as e:
                if "to_dict" not in str(e):
                    raise
                config_dict = model_config.to_dict() if hasattr(model_config, "to_dict") else {}
                marker = " ".join(
                    str(config_dict.get(k, ""))
                    for k in ("model_type", "architectures", "text_config", "vision_config")
                ).lower()
                if "qwen3_5" not in marker and "qwen3.5" not in marker:
                    raise
                print("[qwen3_5_generation_patch] from_model_config fallback to default GenerationConfig")
                return GenerationConfig()

        _patched_from_model_config._pinchbench_qwen3_5_patched = True
        GenerationConfig.from_model_config = classmethod(_patched_from_model_config)

    def _patched_get_generation_config(model: str, trust_remote_code: bool = False):
        if "qwen3.5" in model.lower() or "qwen3_5" in model.lower():
            print(f"[qwen3_5_generation_patch] using default GenerationConfig for {model}")
            return GenerationConfig()
        try:
            return original(model, trust_remote_code=trust_remote_code)
        except Exception as e:
            if "qwen3_5" not in str(e) and "generation_config.json" not in str(e):
                raise
            print(f"[qwen3_5_generation_patch] falling back to default GenerationConfig for {model}: {e}")
            return GenerationConfig()

    _patched_get_generation_config._pinchbench_qwen3_5_patched = True
    verl_model.get_generation_config = _patched_get_generation_config
    print("[qwen3_5_generation_patch] applied")


apply_patch()
