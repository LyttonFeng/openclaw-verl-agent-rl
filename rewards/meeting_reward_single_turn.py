"""
Single-turn variant of meeting_reward for veRL.

The agent-loop version expects extra_info to carry a workspace_path + tool-use
transcript. In single-turn training (no OpenClaw agent loop) the model just
emits the report body as its raw response, so we:

  1. Extract the expected output filename from the task's grading code
     (e.g. `report_path = workspace / "meeting_tldr.md"`).
  2. Write solution_str into a fresh temp workspace under that filename.
  3. Reuse meeting_reward._run_automated_checks and ._run_llm_judge.

This keeps the reward signal aligned with the bench harness without needing
the full tool-call loop.
"""

from __future__ import annotations

import logging
import os
import re
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import meeting_reward as mr

logger = logging.getLogger(__name__)


def _expected_output_filename(task) -> str:
    """Pull the output filename out of the task's automated_checks code."""
    code = mr._extract_grading_code(task)
    if code:
        m = re.search(r'workspace\s*/\s*[\'"]([^\'"]+\.md)[\'"]', code)
        if m:
            return m.group(1)
    m = re.search(r'(?:called|named)\s+`([^`]+\.md)`', task.prompt or "")
    if m:
        return m.group(1)
    return "output.md"


def _copy_inputs(task, workspace: Path) -> None:
    repo_root = mr._REPO_ROOT
    for wf in (task.workspace_files or []):
        dest = wf.get("dest") or wf.get("path")
        if not dest:
            continue
        target = workspace / dest
        target.parent.mkdir(parents=True, exist_ok=True)
        src = wf.get("source")
        if src:
            src_path = repo_root / "assets" / src
            if src_path.exists():
                shutil.copy(src_path, target)
        elif "content" in wf:
            target.write_text(wf["content"], encoding="utf-8")


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    extra_info = extra_info or {}
    task_id = ground_truth or extra_info.get("task_id", "")
    if not task_id:
        return {"score": 0.0, "error": "no task_id", "task_id": ""}

    try:
        task = mr._load_task(task_id)
    except FileNotFoundError as e:
        return {"score": 0.0, "error": str(e), "task_id": task_id}

    output_name = _expected_output_filename(task)
    synthetic_transcript = [
        {"role": "assistant", "content": solution_str or ""},
    ]

    judge_only = os.environ.get("MEETING_REWARD_JUDGE_ONLY", "0") == "1"
    auto_only = os.environ.get("MEETING_REWARD_AUTO_ONLY", "0") == "1"

    weights = extra_info.get(
        "grading_weights",
        task.grading_weights or {"automated": 0.5, "llm_judge": 0.5},
    )
    auto_weight = float(weights.get("automated", 0.5))
    judge_weight = float(weights.get("llm_judge", 0.5))

    auto_score = 0.0
    judge_score = 0.0

    with tempfile.TemporaryDirectory(prefix="meeting_rwd_") as tmp:
        workspace = Path(tmp)
        _copy_inputs(task, workspace)
        (workspace / output_name).write_text(solution_str or "", encoding="utf-8")

        if not judge_only:
            try:
                auto_score = mr._run_automated_checks(task, str(workspace), synthetic_transcript)
            except Exception as e:
                logger.warning(f"auto check failed for {task_id}: {e}")
                traceback.print_exc()

        if not auto_only:
            try:
                judge_score = mr._run_llm_judge(task, str(workspace), synthetic_transcript)
            except Exception as e:
                logger.warning(f"judge failed for {task_id}: {e}")
                traceback.print_exc()

    if judge_only:
        final = judge_score
    elif auto_only:
        final = auto_score
    else:
        denom = auto_weight + judge_weight
        final = (auto_score * auto_weight + judge_score * judge_weight) / denom if denom > 0 else 0.0

    return {
        "score": final,
        "automated_score": auto_score,
        "judge_score": judge_score,
        "task_id": task_id,
        "output_filename": output_name,
    }
