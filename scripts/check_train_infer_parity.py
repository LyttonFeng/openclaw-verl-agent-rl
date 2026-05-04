#!/usr/bin/env python3
"""
Check train vs inference parity for the minimal task16 repo.

- Prompt text must match between ``prepare_prompts.py`` extraction and ``TaskLoader``.
- Prints workspace_files count (seed parity is via the same ``tasks/*.md`` in both paths).

Usage (from repo root)::

  python3 scripts/check_train_infer_parity.py

Exit 0 if all checks pass.

Called automatically by ``rl/train/run_reinforce_lora.sh`` before training
unless ``PINCHBENCH_SKIP_TRAIN_INFER_PARITY=1``.

Rollout alignment: ``rl/agent_loop/openclaw_agent_loop.py`` does **not** inject the legacy
``<tool_call>`` system suffix unless ``PINCHBENCH_RL_INJECT_TOOL_FORMAT_SUFFIX=1``,
so veRL rollout matches OpenClaw → vLLM used by the benchmark.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SCRIPTS = REPO / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from lib_tasks import TaskLoader, resolve_task_markdown_path  # noqa: E402

DEFAULT_TASK_IDS = ["task_16_email_triage"]


def parse_prompt_like_prepare_prompts(task_path: Path) -> str:
    """Same logic as rl/train/prepare_prompts.parse_task_file (## Prompt section)."""
    text = task_path.read_text("utf-8")
    if not text.startswith("---"):
        return ""
    parts = text.split("---", 2)
    if len(parts) < 3:
        return ""
    body = parts[2]
    prompt_lines: list[str] = []
    in_prompt = False
    for line in body.split("\n"):
        if line.strip() == "## Prompt":
            in_prompt = True
            continue
        if in_prompt and line.strip().startswith("## "):
            break
        if in_prompt:
            prompt_lines.append(line)
    return "\n".join(prompt_lines).strip()


def main() -> int:
    tasks_dir = REPO / "pinchbench_tasks"
    if not tasks_dir.is_dir():
        print(f"ERROR: tasks dir not found: {tasks_dir}", file=sys.stderr)
        return 2

    loader = TaskLoader(tasks_dir)
    failed = False
    for tid in DEFAULT_TASK_IDS:
        path = resolve_task_markdown_path(tasks_dir, tid)
        if not path.exists():
            print(f"FAIL {tid}: missing {path}", file=sys.stderr)
            failed = True
            continue
        task = loader.load_task(path)
        p_prep = parse_prompt_like_prepare_prompts(path)
        if task.prompt != p_prep:
            print(f"FAIL {tid}: TaskLoader prompt != prepare_prompts extraction", file=sys.stderr)
            failed = True
            continue
        h = hashlib.sha256(task.prompt.encode()).hexdigest()[:12]
        n_ws = len(task.workspace_files or [])
        print(f"OK  {tid}  prompt_sha12={h}  workspace_files={n_ws}")

    if failed:
        return 1
    print()
    print("task16 prompt extraction parity OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
