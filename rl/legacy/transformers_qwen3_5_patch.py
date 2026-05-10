"""Compatibility shim for Qwen3.5 checkpoints on Transformers 4.x.

Transformers 4.57.x does not ship `qwen3_5`, but it does ship `qwen3_next`,
which is the closest text-only implementation for the Qwen3.5-2B config used
by our veRL run. Qwen3.5 checkpoints store the language-model fields under
`text_config`; treating the top-level config as a plain Qwen3 config produces
an invalid vocab/pad-token combination and crashes during model init.
"""

from __future__ import annotations


def _is_vllm_server_process() -> bool:
    import inspect
    import os
    import sys

    if os.environ.get("PINCHBENCH_QWEN35_VLLM_SERVER") == "1":
        return True

    argv = " ".join(str(arg) for arg in sys.argv).lower()
    if "vllm" in argv and "serve" in argv:
        return True

    # veRL launches vLLM inside a Ray actor, so sys.argv may only look like a
    # generic worker process. In that path AutoConfig is still called from vLLM
    # modules; keep the original top-level Qwen3.5 config there because vLLM
    # expects its own Qwen3_5Config, not Transformers' Qwen3_5TextConfig.
    for frame in inspect.stack(context=0):
        filename = frame.filename.replace("\\", "/").lower()
        if "/site-packages/vllm/" in filename or filename.endswith("/vllm/__init__.py"):
            return True
    return False


def apply_patch() -> None:
    try:
        from transformers import AutoConfig
        from transformers.configuration_utils import PretrainedConfig
        from transformers.models.auto import modeling_auto
        from transformers.models.auto.configuration_auto import CONFIG_MAPPING
        from transformers.models.qwen3_next.configuration_qwen3_next import Qwen3NextConfig
        try:
            from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig
        except Exception:
            Qwen3_5TextConfig = None
    except Exception as e:  # pragma: no cover
        print(f"[qwen3_5_patch] import skipped: {e}")
        return

    try:
        CONFIG_MAPPING._extra_content["qwen3_5"] = Qwen3NextConfig
        if hasattr(CONFIG_MAPPING, "_mapping"):
            CONFIG_MAPPING._mapping["qwen3_5"] = "Qwen3NextConfig"
    except Exception as e:  # pragma: no cover
        print(f"[qwen3_5_patch] CONFIG_MAPPING patch skipped: {e}")

    # veRL 0.7.1 probes AutoModelForImageTextToText before AutoModelForCausalLM.
    # Qwen3.5-2B is text-only in this training path, so keep Qwen3NextConfig on the
    # causal LM route by removing the accidental multimodal registration.
    try:
        image_text_mapping = modeling_auto.AutoModelForImageTextToText._model_mapping
        if Qwen3NextConfig in image_text_mapping:
            image_text_mapping.pop(Qwen3NextConfig, None)
            print("[qwen3_5_patch] removed Qwen3NextConfig from AutoModelForImageTextToText mapping")
    except Exception as e:  # pragma: no cover
        print(f"[qwen3_5_patch] AutoModelForImageTextToText mapping patch skipped: {e}")

    try:
        import verl.workers.fsdp_workers as fsdp_workers
        from transformers import AutoModelForCausalLM

        fsdp_workers.AutoModelForImageTextToText = AutoModelForCausalLM
        print("[qwen3_5_patch] rebound veRL AutoModelForImageTextToText -> transformers.AutoModelForCausalLM")
    except Exception as e:  # pragma: no cover
        print(f"[qwen3_5_patch] veRL model-class rebinding skipped: {e}")

    try:
        from transformers import AutoModelForCausalLM
        from transformers.models.auto import modeling_auto

        image_text_cls = modeling_auto.AutoModelForImageTextToText
        original_image_text_from_pretrained = image_text_cls.from_pretrained
        if not getattr(original_image_text_from_pretrained, "_pinchbench_qwen3_5_patched", False):

            def _image_text_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
                config = kwargs.get("config")
                qwen_text_types = (Qwen3NextConfig,)
                if Qwen3_5TextConfig is not None:
                    qwen_text_types = (Qwen3NextConfig, Qwen3_5TextConfig)
                if isinstance(config, qwen_text_types):
                    return AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, *args, **kwargs)
                return original_image_text_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

            _image_text_from_pretrained._pinchbench_qwen3_5_patched = True
            image_text_cls.from_pretrained = classmethod(_image_text_from_pretrained)
            print("[qwen3_5_patch] patched AutoModelForImageTextToText.from_pretrained for Qwen3Next")
    except Exception as e:  # pragma: no cover
        print(f"[qwen3_5_patch] AutoModelForImageTextToText.from_pretrained patch skipped: {e}")

    try:
        import verl.workers.fsdp_workers as fsdp_workers
        from verl.single_controller.base.decorator import MAGIC_ATTR
        from transformers import AutoModelForCausalLM

        original_init_model = fsdp_workers.ActorRolloutRefWorker.init_model
        if getattr(original_init_model, "_pinchbench_qwen3_5_init_patched", False):
            raise RuntimeError("already patched")
        magic_attr = getattr(original_init_model, MAGIC_ATTR, None)

        def _wrapped_init_model(self, *args, **kwargs):
            fsdp_workers.AutoModelForImageTextToText = AutoModelForCausalLM
            return original_init_model(self, *args, **kwargs)

        if magic_attr is not None:
            setattr(_wrapped_init_model, MAGIC_ATTR, magic_attr)
        _wrapped_init_model._pinchbench_qwen3_5_init_patched = True
        _wrapped_init_model.__name__ = getattr(original_init_model, "__name__", "init_model")
        _wrapped_init_model.__doc__ = getattr(original_init_model, "__doc__", None)
        fsdp_workers.ActorRolloutRefWorker.init_model = _wrapped_init_model
        if hasattr(fsdp_workers, "AsyncActorRolloutRefWorker"):
            async_original = fsdp_workers.AsyncActorRolloutRefWorker.init_model
            async_magic_attr = getattr(async_original, MAGIC_ATTR, magic_attr)

            def _wrapped_async_init_model(self, *args, **kwargs):
                fsdp_workers.AutoModelForImageTextToText = AutoModelForCausalLM
                return async_original(self, *args, **kwargs)

            if async_magic_attr is not None:
                setattr(_wrapped_async_init_model, MAGIC_ATTR, async_magic_attr)
            _wrapped_async_init_model._pinchbench_qwen3_5_init_patched = True
            _wrapped_async_init_model.__name__ = getattr(async_original, "__name__", "init_model")
            _wrapped_async_init_model.__doc__ = getattr(async_original, "__doc__", None)
            fsdp_workers.AsyncActorRolloutRefWorker.init_model = _wrapped_async_init_model
        print("[qwen3_5_patch] patched ActorRolloutRefWorker.init_model for causal LM path")
    except RuntimeError as e:
        if str(e) != "already patched":
            raise
    except Exception as e:  # pragma: no cover
        print(f"[qwen3_5_patch] ActorRolloutRefWorker.init_model patch skipped: {e}")

    try:
        import os
        from verl.workers.rollout.vllm_rollout import vllm_async_server

        server_cls = vllm_async_server.vLLMHttpServer
        original_server_init = server_cls.__init__
        if not getattr(original_server_init, "_pinchbench_qwen3_5_vllm_env_patched", False):

            def _wrapped_server_init(self, *args, **kwargs):
                os.environ["PINCHBENCH_QWEN35_VLLM_SERVER"] = "1"
                return original_server_init(self, *args, **kwargs)

            _wrapped_server_init._pinchbench_qwen3_5_vllm_env_patched = True
            server_cls.__init__ = _wrapped_server_init

        original_run_server = server_cls.run_server
        if not getattr(original_run_server, "_pinchbench_qwen3_5_vllm_env_patched", False):

            async def _wrapped_run_server(self, *args, **kwargs):
                os.environ["PINCHBENCH_QWEN35_VLLM_SERVER"] = "1"
                return await original_run_server(self, *args, **kwargs)

            _wrapped_run_server._pinchbench_qwen3_5_vllm_env_patched = True
            _wrapped_run_server.__name__ = getattr(original_run_server, "__name__", "run_server")
            server_cls.run_server = _wrapped_run_server
            print("[qwen3_5_patch] patched vLLMHttpServer env guard")

        original_launch_server = server_cls.launch_server
        if not getattr(original_launch_server, "_pinchbench_qwen3_5_hf_overrides_patched", False):

            async def _wrapped_launch_server(self, *args, **kwargs):
                os.environ["PINCHBENCH_QWEN35_VLLM_SERVER"] = "1"
                # Try all possible attribute names for model path
                model_path = ""
                for attr in ("path", "local_path", "model", "model_name_or_path"):
                    val = getattr(self.model_config, attr, None)
                    if val:
                        model_path = str(val)
                        break
                if "qwen3.5" in model_path.lower() or "qwen3_5" in model_path.lower() or not model_path:
                    engine_kwargs = getattr(self.config, "engine_kwargs", None)
                    if engine_kwargs is None:
                        engine_kwargs = {}
                        object.__setattr__(self.config, "engine_kwargs", engine_kwargs)
                    vllm_kwargs = engine_kwargs.get("vllm", {}) or {}
                    hf_overrides = vllm_kwargs.get("hf_overrides", {}) or {}
                    # Use direct assignment (not setdefault) to force-override Qwen3.5's
                    # multimodal architectures field back to text-only CausalLM
                    hf_overrides["architectures"] = ["Qwen3_5ForCausalLM"]
                    vllm_kwargs["hf_overrides"] = hf_overrides
                    engine_kwargs["vllm"] = vllm_kwargs
                    print(f"[qwen3_5_patch] forcing vLLM Qwen3.5 architecture to Qwen3_5ForCausalLM (model={model_path!r})")
                return await original_launch_server(self, *args, **kwargs)

            _wrapped_launch_server._pinchbench_qwen3_5_hf_overrides_patched = True
            _wrapped_launch_server.__name__ = getattr(original_launch_server, "__name__", "launch_server")
            server_cls.launch_server = _wrapped_launch_server
    except Exception as e:  # pragma: no cover
        print(f"[qwen3_5_patch] vLLMHttpServer env guard skipped: {e}")

    original_from_pretrained = AutoConfig.from_pretrained

    if getattr(original_from_pretrained, "_pinchbench_qwen3_5_patched", False):
        return

    def _from_pretrained_qwen3_5_aware(pretrained_model_name_or_path, *args, **kwargs):
        if args:
            return original_from_pretrained(pretrained_model_name_or_path, *args, **kwargs)

        config_kwargs = dict(kwargs)
        config_kwargs["_from_auto"] = True
        config_kwargs["name_or_path"] = pretrained_model_name_or_path
        config_kwargs.pop("trust_remote_code", None)
        config_kwargs.pop("code_revision", None)
        try:
            config_dict, unused_kwargs = PretrainedConfig.get_config_dict(
                pretrained_model_name_or_path, **config_kwargs
            )
        except Exception:
            return original_from_pretrained(pretrained_model_name_or_path, **kwargs)

        if config_dict.get("model_type") != "qwen3_5" or not isinstance(config_dict.get("text_config"), dict):
            return original_from_pretrained(pretrained_model_name_or_path, **kwargs)
        if _is_vllm_server_process():
            return original_from_pretrained(pretrained_model_name_or_path, **kwargs)

        text_config = dict(config_dict["text_config"])
        if Qwen3_5TextConfig is not None:
            text_config["model_type"] = "qwen3_5_text"
            text_config["architectures"] = ["Qwen3_5ForCausalLM"]
        else:
            text_config["model_type"] = "qwen3_next"
            text_config["architectures"] = ["Qwen3NextForCausalLM"]
        text_config.setdefault("tie_word_embeddings", config_dict.get("tie_word_embeddings", True))
        text_config.setdefault("_name_or_path", str(pretrained_model_name_or_path))
        if Qwen3_5TextConfig is not None:
            print("[qwen3_5_patch] using text_config as Qwen3_5TextConfig")
            return Qwen3_5TextConfig.from_dict(text_config, **unused_kwargs)
        print("[qwen3_5_patch] using text_config as Qwen3NextConfig")
        return Qwen3NextConfig.from_dict(text_config, **unused_kwargs)

    _from_pretrained_qwen3_5_aware._pinchbench_qwen3_5_patched = True
    AutoConfig.from_pretrained = _from_pretrained_qwen3_5_aware
    print("[qwen3_5_patch] applied")


apply_patch()
