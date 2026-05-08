"""
Meeting Analysis terminal reward function for veRL.

veRL custom_reward_function interface:
    compute_score(data_source, solution_str, ground_truth, extra_info, **kwargs) -> float | dict

This module wraps PinchBench official grading (automated checks + LLM judge)
as a terminal reward for GRPO training.

Usage in veRL config:
    reward.custom_reward_function.path=rl/train/meeting_reward.py
    reward.custom_reward_function.name=compute_score

Environment variables:
    DEEPSEEK_API_KEY        - Required for the LLM judge (default provider: DeepSeek)
    MEETING_JUDGE_MODEL     - Judge model (default: deepseek-chat)
    MEETING_JUDGE_BASE_URL  - Judge API base URL (default: https://api.deepseek.com/v1)
    MEETING_JUDGE_PROVIDER  - "deepseek" (default) — kept as a switch for future providers
    MEETING_REWARD_JUDGE_ONLY - If "1", skip automated checks, only use judge (for debugging)
    MEETING_REWARD_AUTO_ONLY  - If "1", skip judge, only use automated checks (fast mode)
"""

from __future__ import annotations

import logging
import os
import re
import sys
import traceback
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Ensure scripts/ is importable
# Note: this file lives at <repo>/rewards/meeting_reward.py, so REPO_ROOT
# is parent.parent (NOT parent.parent.parent — that mistake left REPO_ROOT
# pointing at /workspace/ on pods, which silently broke _load_task and
# made every rollout grade to 0).
_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))


def _get_judge_config() -> tuple[str, str, str]:
    """Return (model, base_url, api_key) for the judge.

    Default provider is DeepSeek (deepseek-chat). Override via env vars if you
    point at a different OpenAI-compatible endpoint.
    """
    return (
        os.environ.get("MEETING_JUDGE_MODEL", "deepseek-chat"),
        os.environ.get("MEETING_JUDGE_BASE_URL", "https://api.deepseek.com/v1"),
        os.environ.get("DEEPSEEK_API_KEY", ""),
    )


def _load_task(task_id: str):
    """Load task definition from pinchbench_tasks/meeting_analysis/ directory.

    Honors MEETING_TASKS_DIR env var as override; useful if a caller wants
    to point at a different task family. Falls back to the canonical
    <repo>/pinchbench_tasks/meeting_analysis/ path used by this repo.
    """
    from lib_tasks import TaskLoader

    env_dir = os.environ.get("MEETING_TASKS_DIR")
    tasks_dir = Path(env_dir) if env_dir else (_REPO_ROOT / "pinchbench_tasks" / "meeting_analysis")
    task_file = tasks_dir / f"{task_id}.md"
    if not task_file.exists():
        raise FileNotFoundError(f"Task file not found: {task_file}")
    loader = TaskLoader(tasks_dir)
    return loader.load_task(task_file)


def _run_automated_checks(task, workspace_path: str, transcript: list) -> float:
    """Run the task's embedded grade() function and return average score."""
    grading_code = _extract_grading_code(task)
    if not grading_code:
        return 0.0

    namespace: Dict[str, Any] = {}
    exec(grading_code, namespace)
    grade_func = namespace.get("grade")
    if not callable(grade_func):
        return 0.0

    scores = grade_func(transcript, workspace_path)
    if not isinstance(scores, dict) or not scores:
        return 0.0

    values = [float(v) for v in scores.values() if isinstance(v, (int, float))]
    return sum(values) / len(values) if values else 0.0


def _extract_grading_code(task) -> str:
    """Extract Python grading code from task's automated_checks section."""
    if not task.automated_checks:
        return ""
    match = re.search(r"```python\s*(.*?)\s*```", task.automated_checks, re.DOTALL)
    return match.group(1) if match else ""


def _run_llm_judge(task, workspace_path: str, transcript: list) -> float:
    """Call LLM judge and return score 0.0-1.0.

    Uses direct HTTP call to avoid environment variable conflicts.
    """
    import json as json_mod
    import urllib.request

    transcript_summary = _summarize_transcript(transcript)
    workspace_content = _read_workspace_files(workspace_path)

    rubric = task.llm_judge_rubric or ""
    if not rubric and task.grading_criteria:
        rubric = "\n".join(f"- {c}" for c in task.grading_criteria)

    judge_prompt = (
        f"## Task\n{task.prompt}\n\n"
        f"## Expected Behavior\n{task.expected_behavior}\n\n"
        f"## Agent Transcript\n{transcript_summary}\n\n"
    )
    if workspace_content:
        judge_prompt += f"## Workspace Files Created\n{workspace_content}\n\n"
    judge_prompt += (
        f"## Grading Rubric\n{rubric}\n\n"
        "## Instructions\n"
        "Score the agent's output based on the rubric. "
        "Respond with ONLY a JSON object:\n"
        '{"scores": {"criterion_name": float_0_to_1, ...}, "total": float_0_to_1, "notes": "brief explanation"}'
    )

    judge_model, base_url, api_key = _get_judge_config()

    # Direct HTTP call — no environment variable dependency
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    payload = json_mod.dumps({
        "model": judge_model,
        "messages": [
            {"role": "system", "content": "You are a strict grading function. Respond with ONLY a JSON object, no prose, no markdown fences, no extra text."},
            {"role": "user", "content": judge_prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 2048,
    }).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    try:
        req = urllib.request.Request(endpoint, data=payload, headers=headers)
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json_mod.loads(resp.read())
            text = data["choices"][0]["message"]["content"]
            return _parse_judge_score(text)
    except Exception as e:
        logger.warning(f"Judge HTTP call failed: {e}")
        return 0.0


def _parse_judge_score(result: str) -> float:
    """Parse judge response JSON to extract total score."""
    import json as json_mod

    if not result:
        return 0.0

    # Try to extract JSON from response
    text = result.strip()
    # Remove markdown fences if present
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)

    try:
        data = json_mod.loads(text)
    except json_mod.JSONDecodeError:
        # Try to find JSON object in text
        match = re.search(r"\{[^{}]*\"total\"[^{}]*\}", text)
        if match:
            try:
                data = json_mod.loads(match.group())
            except json_mod.JSONDecodeError:
                return 0.0
        else:
            return 0.0

    # Extract score
    if isinstance(data, dict):
        if "total" in data:
            return float(data["total"])
        if "scores" in data and isinstance(data["scores"], dict):
            scores = data["scores"]
            values = [float(v) for v in scores.values() if isinstance(v, (int, float))]
            return sum(values) / len(values) if values else 0.0
    return 0.0


def _summarize_transcript(transcript: list) -> str:
    """Summarize agent transcript for judge prompt."""
    if not transcript:
        return "(no transcript)"

    lines = []
    for entry in transcript:
        if isinstance(entry, dict):
            role = entry.get("role", entry.get("type", "unknown"))
            content = entry.get("content", entry.get("text", ""))
            if isinstance(content, list):
                # Multi-part content
                parts = []
                for part in content:
                    if isinstance(part, dict):
                        if part.get("type") == "tool_use":
                            parts.append(f"[Tool: {part.get('name', '?')}({_truncate(str(part.get('input', '')), 200)})]")
                        elif part.get("type") == "tool_result":
                            parts.append(f"[Result: {_truncate(str(part.get('content', '')), 300)}]")
                        elif part.get("type") == "text":
                            parts.append(_truncate(part.get("text", ""), 500))
                        else:
                            parts.append(_truncate(str(part), 200))
                    else:
                        parts.append(_truncate(str(part), 200))
                content = "\n".join(parts)
            else:
                content = _truncate(str(content), 500)
            lines.append(f"**{role}**: {content}")
        elif isinstance(entry, str):
            lines.append(_truncate(entry, 300))

    return "\n\n".join(lines[-30:])  # Last 30 entries max


def _read_workspace_files(workspace_path: str) -> str:
    """Read text files in workspace for judge context."""
    if not workspace_path:
        return ""

    workspace = Path(workspace_path)
    if not workspace.is_dir():
        return ""

    content_parts = []
    for f in sorted(workspace.iterdir()):
        if f.is_file() and f.suffix in (".md", ".txt", ".json", ".csv"):
            # Skip the input transcript
            if f.name in ("transcript.md", "meeting_transcript.md"):
                continue
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
                content_parts.append(f"### {f.name}\n```\n{_truncate(text, 3000)}\n```")
            except Exception:
                continue

    return "\n\n".join(content_parts)


def _truncate(text: str, max_len: int) -> str:
    if len(text) <= max_len:
        return text
    return text[:max_len] + "..."


# ─── veRL Interface ──────────────────────────────────────────────────────────


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
    **kwargs,
) -> dict:
    """
    veRL custom_reward_function entry point.

    Args:
        data_source: "pinchbench/meeting_analysis"
        solution_str: model's full response text (from rollout)
        ground_truth: task_id (set in parquet reward_model.ground_truth)
        extra_info: dict with task_id, workspace_files, grading info, etc.

    Returns:
        dict with "score" key (float 0.0-1.0) for veRL reward placement.
    """
    extra_info = extra_info or {}
    task_id = ground_truth or extra_info.get("task_id", "")

    if not task_id:
        return {"score": 0.0, "error": "no task_id", "task_id": ""}

    try:
        task = _load_task(task_id)
    except FileNotFoundError as e:
        return {"score": 0.0, "error": str(e), "task_id": task_id}

    # Get workspace path and transcript from extra_info
    # These are populated by the agent loop after rollout completes
    workspace_path = extra_info.get("workspace_path", "")
    transcript = extra_info.get("transcript", [])

    # Determine grading mode
    judge_only = os.environ.get("MEETING_REWARD_JUDGE_ONLY", "0") == "1"
    auto_only = os.environ.get("MEETING_REWARD_AUTO_ONLY", "0") == "1"

    grading_weights = extra_info.get(
        "grading_weights",
        task.grading_weights or {"automated": 0.5, "llm_judge": 0.5},
    )
    auto_weight = float(grading_weights.get("automated", 0.5))
    judge_weight = float(grading_weights.get("llm_judge", 0.5))

    auto_score = 0.0
    judge_score = 0.0

    # Run automated checks
    if not judge_only:
        try:
            auto_score = _run_automated_checks(task, workspace_path, transcript)
        except Exception as e:
            logger.warning(f"Automated check failed for {task_id}: {e}")
            traceback.print_exc()
            auto_score = 0.0

    # Run LLM judge
    if not auto_only:
        try:
            judge_score = _run_llm_judge(task, workspace_path, transcript)
        except Exception as e:
            logger.warning(f"LLM judge failed for {task_id}: {e}")
            traceback.print_exc()
            judge_score = 0.0

    # Combine scores
    if judge_only:
        final_score = judge_score
    elif auto_only:
        final_score = auto_score
    else:
        total_weight = auto_weight + judge_weight
        if total_weight > 0:
            final_score = (auto_score * auto_weight + judge_score * judge_weight) / total_weight
        else:
            final_score = 0.0

    return {
        "score": final_score,
        "automated_score": auto_score,
        "judge_score": judge_score,
        "auto_weight": auto_weight,
        "judge_weight": judge_weight,
        "task_id": task_id,
        "grading_type": task.grading_type,
    }
