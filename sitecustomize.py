"""Repo-local Python startup hooks for veRL Ray workers."""

from __future__ import annotations

import os
from pathlib import Path


root = Path(__file__).resolve().parent
os.environ.setdefault("PINCHBENCH_DIR", str(root))


def _optional_patch(env_name: str, module: str) -> None:
    value = os.environ.get(env_name, "").strip().lower()
    if value not in {"1", "true", "yes", "on"}:
        return
    try:
        mod = __import__(module, fromlist=["apply_patch"])
        mod.apply_patch()
    except Exception as exc:
        import sys

        print(f"[sitecustomize] {module} skipped: {exc}", file=sys.stderr)


_optional_patch("PINCHBENCH_BEST_CKPT", "rl.verl_best_ckpt_patch")
_optional_patch("PINCHBENCH_LORA_ONLY_CKPT", "rl.verl_lora_only_ckpt_patch")
