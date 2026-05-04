from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any


TERMINAL_REWARD_WEIGHT = float(
    os.environ.get("PINCHBENCH_TASK16_TERMINAL_REWARD_WEIGHT", "0.8")
)
TASK16_NO_REPORT_TERMINAL_PENALTY = float(
    os.environ.get("PINCHBENCH_TASK16_NO_REPORT_TERMINAL_PENALTY", "-1.2")
)

DEFAULT_TASK16_RUBRIC: dict[str, Any] = {
    "required_report_schema": "incident_groups_v1",
    "minimum_email_coverage": 13,
}


def _lower(s: str | None) -> str:
    return (s or "").lower()


def _tool_calls(turn: dict[str, Any]) -> list[dict[str, Any]]:
    tc = turn.get("tool_calls")
    return tc if isinstance(tc, list) else []


def _tool_name(tc: dict[str, Any]) -> str:
    fn = tc.get("function")
    if isinstance(fn, dict):
        return str(fn.get("name", ""))
    return str(tc.get("name", ""))


def _tool_args(tc: dict[str, Any]) -> str:
    fn = tc.get("function")
    if isinstance(fn, dict):
        args = fn.get("arguments", "")
    else:
        args = tc.get("arguments", "")
    if isinstance(args, str):
        return args
    try:
        return json.dumps(args, ensure_ascii=False)
    except Exception:
        return str(args)


def _parse_args_json(args: str) -> dict[str, Any]:
    try:
        obj = json.loads(args)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def is_task16(task_id: str | None) -> bool:
    return task_id == "task_16_email_triage"


def _write_call(turn: dict[str, Any]) -> dict[str, Any] | None:
    for tc in _tool_calls(turn):
        if _tool_name(tc) != "write":
            continue
        args = _parse_args_json(_tool_args(tc))
        path = _lower(str(args.get("path", "")))
        if path.endswith("triage_report.md"):
            return args
    return None


def _all_write_calls(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    writes: list[dict[str, Any]] = []
    for turn in trajectory:
        if turn.get("role") != "assistant":
            continue
        args = _write_call(turn)
        if args is not None:
            writes.append(args)
    return writes


def _writes_triage_report(turn: dict[str, Any]) -> bool:
    return _write_call(turn) is not None


def _read_calls(turn: dict[str, Any]) -> list[str]:
    out: list[str] = []
    for tc in _tool_calls(turn):
        if _tool_name(tc) != "read":
            continue
        args = _parse_args_json(_tool_args(tc))
        path = str(args.get("path", ""))
        if path:
            out.append(path)
    return out


def _count_inbox_reads(turn: dict[str, Any]) -> int:
    return sum(1 for p in _read_calls(turn) if p.startswith("inbox/email_"))


def _reads_many_inbox_emails(turn: dict[str, Any]) -> bool:
    return _count_inbox_reads(turn) >= 8


def _unique_inbox_reads(turns: list[dict[str, Any]]) -> set[str]:
    seen: set[str] = set()
    for turn in turns:
        for path in _read_calls(turn):
            if path.startswith("inbox/email_"):
                seen.add(path)
    return seen


def _bulk_read_count(prev_turns: list[dict[str, Any]]) -> int:
    return sum(1 for t in prev_turns if _reads_many_inbox_emails(t))


def _read_only_turn(turn: dict[str, Any]) -> bool:
    return bool(_read_calls(turn)) and not _writes_triage_report(turn)


def _consecutive_read_only_streak(prev_turns: list[dict[str, Any]], turn: dict[str, Any]) -> int:
    streak = 0
    for candidate in [*prev_turns, turn][::-1]:
        if _read_only_turn(candidate):
            streak += 1
        else:
            break
    return streak


def _tool_result_text(tool_result: str | None) -> str:
    return _lower(tool_result)


def _context_overflow(tool_result: str | None) -> bool:
    txt = _tool_result_text(tool_result)
    return "maximum context length" in txt or ("32768" in txt and "input tokens" in txt)


def _structured_complete_report(turn: dict[str, Any]) -> bool:
    args = _write_call(turn)
    if not args:
        return False
    content = _lower(str(args.get("content", "")))
    has_incident_groups = "## incident groups" in content
    has_standalone_items = "## standalone items" in content
    has_priority = "priority" in content
    has_category = "category" in content
    has_action = "recommended action" in content or re.search(r"\baction\b", content) is not None
    p_count = len(re.findall(r"\bp[0-4]\b", content))
    flat_ok = bool(has_priority and has_category and has_action and p_count >= 8)
    grouped_ok = bool(
        has_incident_groups
        and has_standalone_items
        and has_priority
        and has_category
        and has_action
        and p_count >= 4
    )
    return flat_ok or grouped_ok


def _incident_group_schema_report(turn: dict[str, Any]) -> bool:
    args = _write_call(turn)
    if not args:
        return False
    content = _lower(str(args.get("content", "")))
    return "## incident groups" in content and "## standalone items" in content


def _generic_or_partial_report(turn: dict[str, Any]) -> bool:
    args = _write_call(turn)
    if not args:
        return False
    content = _lower(str(args.get("content", "")))
    if _structured_complete_report(turn):
        return False
    if len(re.findall(r"\bp[0-4]\b", content)) < 5:
        return True
    if "bigclient" not in content and "incident" not in content and "security" not in content:
        return True
    return False


def _report_content_from_trajectory(trajectory: list[dict[str, Any]]) -> str:
    writes = _all_write_calls(trajectory)
    if not writes:
        return ""
    return str(writes[-1].get("content", ""))


def _report_content_from_workspace(workspace_path: str | None) -> str:
    if not workspace_path:
        return ""
    path = Path(workspace_path) / "triage_report.md"
    try:
        return path.read_text("utf-8", errors="replace")
    except Exception:
        return ""


def _final_report_content(
    trajectory: list[dict[str, Any]],
    workspace_path: str | None,
) -> str:
    return _report_content_from_trajectory(trajectory) or _report_content_from_workspace(workspace_path)


def _email_patterns(email_id: str) -> list[str]:
    suffix = email_id.split("_", 1)[-1]
    return [
        email_id.lower(),
        f"{email_id.lower()}.txt",
        f"email {suffix}",
        f"email-{suffix}",
    ]


def _mentions_email(content: str, email_id: str) -> bool:
    text = _lower(content)
    return any(pattern in text for pattern in _email_patterns(email_id))


def _window_around_email(content: str, email_id: str, radius: int = 500) -> str:
    text = _lower(content)
    positions = [text.find(pattern) for pattern in _email_patterns(email_id)]
    positions = [p for p in positions if p >= 0]
    if not positions:
        return ""
    pos = min(positions)
    return text[max(0, pos - radius) : pos + radius]


def _priority_near_email(content: str, email_id: str, priority: str | None = None) -> bool:
    window = _window_around_email(content, email_id)
    if not window:
        return False
    if priority:
        return re.search(rf"\b{re.escape(_lower(priority))}\b", window) is not None
    return re.search(r"\bp[0-4]\b", window) is not None


def _coverage_count(content: str) -> int:
    return sum(1 for i in range(1, 14) if _mentions_email(content, f"email_{i:02d}"))


def _required_email_ids() -> list[str]:
    return [f"email_{i:02d}" for i in range(1, 14)]


def _category_near_email(content: str, email_id: str) -> bool:
    window = _window_around_email(content, email_id)
    if not window:
        return False
    categories = (
        "incident",
        "client",
        "internal-request",
        "administrative",
        "code-review",
        "automated",
        "newsletter",
        "spam",
    )
    return any(category in window for category in categories)


def _action_near_email(content: str, email_id: str) -> bool:
    window = _window_around_email(content, email_id)
    if not window:
        return False
    return (
        "action" in window
        or "recommended" in window
        or "reply" in window
        or "respond" in window
        or "review" in window
        or "archive" in window
        or "schedule" in window
        or "join" in window
    )


def _per_email_field_count(content: str) -> int:
    count = 0
    for email_id in _required_email_ids():
        if (
            _mentions_email(content, email_id)
            and _priority_near_email(content, email_id)
            and _category_near_email(content, email_id)
            and _action_near_email(content, email_id)
        ):
            count += 1
    return count


def _placeholder_report(content: str) -> bool:
    text = _lower(content)
    placeholder_patterns = (
        r"\[email\s*\d+\]",
        r"\[action\]",
        r"\[category\]",
        r"\[priority\]",
        r"\[routine task\]",
        r"\[critical issue today\]",
    )
    return any(re.search(pattern, text) for pattern in placeholder_patterns)


def _summary_only_report(content: str) -> bool:
    text = _lower(content)
    has_summary = "summary" in text or "daily plan" in text or "priority summary" in text
    coverage = _coverage_count(content)
    field_count = _per_email_field_count(content)
    return bool(has_summary and coverage < 10 and field_count < 8)


def _rubric_from_extra_info(extra_info: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(extra_info, dict):
        return dict(DEFAULT_TASK16_RUBRIC)
    configured = extra_info.get("reward_rubric")
    if isinstance(configured, dict) and configured:
        rubric = dict(DEFAULT_TASK16_RUBRIC)
        rubric.update(configured)
        return rubric
    return dict(DEFAULT_TASK16_RUBRIC)


def _artifact_contract_reward(
    trajectory: list[dict[str, Any]],
    workspace_path: str,
    extra_info: dict[str, Any] | None,
) -> float:
    content = _final_report_content(trajectory, workspace_path)
    if not content:
        return 0.0

    content_l = _lower(content)
    rubric = _rubric_from_extra_info(extra_info)
    reward = 0.0

    coverage = _coverage_count(content)
    minimum_coverage = int(rubric.get("minimum_email_coverage", 10))
    if coverage >= minimum_coverage:
        reward += 0.6
    elif coverage >= 10:
        reward += 0.25
    elif coverage < 6:
        reward -= 0.4

    field_count = _per_email_field_count(content)
    if field_count >= minimum_coverage:
        reward += 0.5
    elif field_count >= 10:
        reward += 0.2
    elif coverage >= 8:
        reward -= 0.25

    if "## incident groups" in content_l and "## standalone items" in content_l:
        reward += 0.2

    if _placeholder_report(content):
        reward -= 1.0
    if _summary_only_report(content):
        reward -= 0.5

    return max(-1.5, min(1.5, reward))


def _early_switch_to_report(prev_turns: list[dict[str, Any]], turn: dict[str, Any]) -> bool:
    if not _writes_triage_report(turn):
        return False
    bulk_reads = _bulk_read_count(prev_turns)
    return bulk_reads <= 1


def _late_no_report(prev_turns: list[dict[str, Any]], turn: dict[str, Any]) -> bool:
    if _writes_triage_report(turn):
        return False
    bulk_reads = _bulk_read_count(prev_turns)
    current_bulk = _reads_many_inbox_emails(turn)
    return bulk_reads >= 1 and current_bulk


def _consecutive_read_only_turns_penalty(prev_turns: list[dict[str, Any]], turn: dict[str, Any]) -> float:
    streak = _consecutive_read_only_streak(prev_turns, turn)
    if streak < 2:
        return 0.0
    if streak == 2:
        return -0.3
    if streak == 3:
        return -0.6
    return -0.9


def _post_coverage_reread_penalty(
    prev_turns: list[dict[str, Any]],
    turn: dict[str, Any],
    extra_info: dict[str, Any] | None,
) -> float:
    if _writes_triage_report(turn):
        return 0.0
    current_reads = [p for p in _read_calls(turn) if p.startswith("inbox/email_")]
    if len(current_reads) < 4:
        return 0.0
    rubric = _rubric_from_extra_info(extra_info)
    minimum_coverage = int(rubric.get("minimum_email_coverage", 10))
    seen_before = _unique_inbox_reads(prev_turns)
    if len(seen_before) < minimum_coverage:
        return 0.0
    repeated = sum(1 for path in current_reads if path in seen_before)
    if repeated >= 4:
        return -0.7
    return -0.35


def _same_turn_bulk_read_and_write(turn: dict[str, Any]) -> bool:
    return _count_inbox_reads(turn) >= 4 and _writes_triage_report(turn)


def _write_before_sufficient_read_results(
    prev_turns: list[dict[str, Any]],
    turn: dict[str, Any],
    extra_info: dict[str, Any] | None,
) -> bool:
    if not _writes_triage_report(turn):
        return False
    rubric = _rubric_from_extra_info(extra_info)
    minimum_coverage = int(rubric.get("minimum_email_coverage", 10))
    return len(_unique_inbox_reads(prev_turns)) < minimum_coverage


def _report_created_in_workspace(workspace_dir: str | None) -> bool:
    if not workspace_dir:
        return False
    return (Path(workspace_dir) / "triage_report.md").exists()


def terminal_reward_raw(terminal_success: bool, workspace_dir: str | None) -> float:
    if terminal_success:
        return 1.0
    if _report_created_in_workspace(workspace_dir):
        return -0.25
    return TASK16_NO_REPORT_TERMINAL_PENALTY


def task16_event_reward(
    turn: dict[str, Any],
    prev_turns: list[dict[str, Any]],
    tool_result: str | None,
    extra_info: dict[str, Any] | None = None,
) -> float:
    r = 0.0

    if _writes_triage_report(turn):
        r += 0.2
    if _structured_complete_report(turn):
        r += 0.8
    if _incident_group_schema_report(turn):
        r += 0.2
    if _early_switch_to_report(prev_turns, turn):
        r += 0.5
    if _reads_many_inbox_emails(turn) and _bulk_read_count(prev_turns) >= 1:
        r -= 0.8
    if _late_no_report(prev_turns, turn):
        r -= 0.5
    r += _consecutive_read_only_turns_penalty(prev_turns, turn)
    r += _post_coverage_reread_penalty(prev_turns, turn, extra_info)
    if _same_turn_bulk_read_and_write(turn):
        r -= 0.8
    if _write_before_sufficient_read_results(prev_turns, turn, extra_info):
        r -= 0.8
    if _generic_or_partial_report(turn):
        r -= 0.6
    if _context_overflow(tool_result):
        r -= 1.0

    return r


def _assistant_turn_indices(trajectory: list[dict[str, Any]]) -> list[int]:
    return [i for i, t in enumerate(trajectory) if t.get("role") == "assistant"]


def _clip_turn_reward(x: float) -> float:
    return max(-1.5, min(1.5, x))


def compute_episode_rewards(
    trajectory: list[dict[str, Any]],
    terminal_success: bool,
    task_id: str,
    workspace_path: str = "",
    extra_info: dict[str, Any] | None = None,
) -> list[float]:
    assistant_indices = _assistant_turn_indices(trajectory)
    terminal = TERMINAL_REWARD_WEIGHT * terminal_reward_raw(
        terminal_success=terminal_success,
        workspace_dir=workspace_path or None,
    )

    if not assistant_indices:
        return [terminal]

    per_turn: list[float] = []
    for pos, turn_idx in enumerate(assistant_indices):
        turn = trajectory[turn_idx]
        prev_turns = [trajectory[i] for i in assistant_indices[:pos]]

        tool_result = None
        for t in trajectory[turn_idx + 1 :]:
            if t.get("role") == "tool":
                tool_result = str(t.get("content", ""))
                break
            if t.get("role") == "assistant":
                break

        r = 0.0
        if is_task16(task_id):
            r += task16_event_reward(turn, prev_turns, tool_result, extra_info)
        per_turn.append(_clip_turn_reward(r))

    if is_task16(task_id):
        artifact_reward = _artifact_contract_reward(trajectory, workspace_path, extra_info)
        if artifact_reward and per_turn:
            per_turn[-1] = _clip_turn_reward(per_turn[-1] + artifact_reward)

    return [x + terminal for x in per_turn]


async def compute_episode_rewards_async(
    trajectory: list[dict[str, Any]],
    terminal_success: bool,
    task_id: str,
    task_prompt: str = "",
    mode: str = "task16-event-only-v2",
    vllm_base_url: str = "",
    judge_model: str = "",
    judge_api_key: str = "",
    workspace_path: str = "",
    extra_info: dict[str, Any] | None = None,
) -> list[float]:
    return compute_episode_rewards(
        trajectory=trajectory,
        terminal_success=terminal_success,
        task_id=task_id,
        workspace_path=workspace_path,
        extra_info=extra_info,
    )
