#!/usr/bin/env python3
"""Launch veRL main_ppo after applying local compatibility patches."""

from __future__ import annotations

import importlib.metadata
import runpy

import rl.transformers_qwen3_5_patch  # noqa: F401
import rl.verl_qwen3_5_generation_patch  # noqa: F401
import rl.verl_no_masked_whiten_patch  # noqa: F401


_orig_version = importlib.metadata.version


def _patched_version(dist_name: str) -> str:
    try:
        return _orig_version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        if dist_name == "vllm":
            import vllm

            return vllm.__version__
        raise


importlib.metadata.version = _patched_version

runpy.run_module("verl.trainer.main_ppo", run_name="__main__")
