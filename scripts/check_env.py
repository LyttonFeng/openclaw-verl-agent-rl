#!/usr/bin/env python3
"""Preflight checks for the task16 RL reproduction environment."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from contextlib import contextmanager
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def require_module(name: str) -> None:
    try:
        importlib.import_module(name)
    except Exception as exc:
        raise SystemExit(f"Missing or broken Python module {name}: {exc}") from exc


@contextmanager
def prepend_sys_path(path: Path):
    value = str(path)
    added = value not in sys.path
    if added:
        sys.path.insert(0, value)
    try:
        yield
    finally:
        if added:
            try:
                sys.path.remove(value)
            except ValueError:
                pass


def check_judge(root: Path) -> None:
    if os.environ.get("PINCHBENCH_SKIP_JUDGE_PREFLIGHT", "").strip().lower() in {"1", "true", "yes", "on"}:
        print("WARN: judge preflight skipped by PINCHBENCH_SKIP_JUDGE_PREFLIGHT")
        return

    with prepend_sys_path(root / "scripts"):
        from lib_grading import preflight_judge_connection, resolve_judge_backend_from_env

        judge_cfg = resolve_judge_backend_from_env(
            default_backend="api",
            default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        backend = str(judge_cfg["judge_backend"])
        model = str(judge_cfg["judge_model"])
        base_url = str(judge_cfg["judge_base_url"] or "")
        has_key = bool(str(judge_cfg["judge_api_key"] or "").strip())
        print(
            "Judge config: "
            f"backend={backend} model={model} base_url={base_url or '<none>'} api_key_set={has_key}"
        )
        if backend != "api":
            raise SystemExit(
                "Hybrid task16 grading requires API judge for reproducible training. "
                "Set DASHSCOPE_API_KEY or PINCHBENCH_GRADE_JUDGE_API_KEY."
            )
        preflight_judge_connection(
            judge_model=model,
            judge_backend=backend,
            judge_base_url=base_url,
            judge_api_key=str(judge_cfg["judge_api_key"] or ""),
            timeout_seconds=float(os.environ.get("PINCHBENCH_JUDGE_PREFLIGHT_TIMEOUT", "30")),
        )
        print(f"OK: judge preflight succeeded ({backend}/{model})")


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    checks = [
        ("repo root", root.is_dir()),
        ("task16 markdown", (root / "pinchbench_tasks/task_16_email_triage.md").is_file()),
        ("agent loop config", (root / "agent_loop/config.yaml").is_file()),
    ]
    for label, ok in checks:
        if not ok:
            raise SystemExit(f"Missing {label}")

    if sys.version_info[:2] != (3, 12):
        print(f"WARN: Python 3.12 is expected, got {sys.version.split()[0]}")

    for mod in ("pandas", "pyarrow", "yaml", "aiohttp", "torch", "transformers", "vllm", "verl"):
        require_module(mod)

    try:
        verl_version = version("verl")
    except PackageNotFoundError as exc:
        raise SystemExit("veRL is not installed") from exc
    if verl_version != "0.7.1":
        print(f"WARN: verified veRL version is 0.7.1, installed {verl_version}")

    check_judge(root)

    host = os.environ.get("OPENCLAW_HOST", "")
    if not host:
        print("WARN: OPENCLAW_HOST is not set")
    elif shutil.which("ssh"):
        user = os.environ.get("OPENCLAW_USER", "root")
        port = os.environ.get("OPENCLAW_PORT", "22")
        key = os.environ.get("OPENCLAW_SSH_KEY", str(Path.home() / ".ssh/id_ed25519"))
        result = subprocess.run(
            [
                "ssh",
                "-o",
                "StrictHostKeyChecking=no",
                "-o",
                "ConnectTimeout=10",
                "-i",
                key,
                "-p",
                port,
                f"{user}@{host}",
                "command -v openclaw >/dev/null && openclaw --version",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise SystemExit(
                "OpenClaw SSH preflight failed: "
                + (result.stderr.strip() or result.stdout.strip())
            )
        print(result.stdout.strip())

    print("OK: environment preflight completed")


if __name__ == "__main__":
    main()
