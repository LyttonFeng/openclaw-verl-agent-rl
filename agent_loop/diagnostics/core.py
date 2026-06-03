"""Diagnostics core: DiagnosticsResult + diagnose().

Three layers of analysis, all assembled by `diagnose()`:

  Layer 1 — Structural (what the trajectory did)
      tool call counts, transcript-read flag, output-file-written flag,
      thinking_chars, read_loop, timeout, empty_response.

  Layer 2 — Output-budget allocation (where did the chars go)
      output_file_length vs final_chat_chars ratio, transcript_read_truncated.
      Catches the "model talks in chat instead of writing the file" failure
      mode that automated grading silently misses.

  Layer 3 — Consume already-computed grading signals
      automated_breakdown (per-check 0..1 from grading) → automated_failed_checks.
      prm_turn_scores → prm_negative_count.
      We do NOT re-grade; we surface what exists.

`fatal=True` is reserved for clearly-broken trajectories (timeout,
output_not_written, empty_response, transcript_not_read) and is the rollout
pipeline's signal to skip the expensive judge call. Layer 2/3 signals are
warnings only — they show up in `failure_tags` but do not flip `fatal`.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from .protocol import TaskPlugin, resolve_plugin


# Read tool returns are observed truncated at 39999/40000 chars (PinchBench
# default). Anything within 100 chars of that ceiling is treated as truncated.
READ_TRUNCATION_THRESHOLD = 39900

# Output budget ratio below this value is flagged as misallocation.
BUDGET_MISALLOC_THRESHOLD = 0.70

# Failure tags that mean "this trajectory is broken; skip the judge".
FATAL_TAGS = frozenset({
    "timeout",
    "output_not_written",
    "empty_response",
    "transcript_not_read",
})


@dataclass
class DiagnosticsResult:
    task_id: str
    family_id: Optional[str] = None

    # Layer 1 — structural
    assistant_turns: int = 0
    total_turns: int = 0
    tool_calls_read: int = 0
    tool_calls_write: int = 0
    tool_calls_other: int = 0
    transcript_read: bool = False
    output_file_written: bool = False
    output_file_exists: bool = False
    output_file_length: int = 0
    thinking_chars: int = 0
    read_loop: bool = False
    empty_response: bool = False
    timed_out: bool = False
    execution_time: float = 0.0

    # Layer 2 — output-budget allocation
    final_chat_chars: int = 0
    output_budget_ratio: Optional[float] = None
    transcript_read_truncated: bool = False
    max_read_size: int = 0

    # Layer 3 — consumed grading signals
    automated_breakdown: dict[str, float] = field(default_factory=dict)
    automated_failed_checks: list[str] = field(default_factory=list)
    prm_turn_scores: Optional[list[int]] = None
    prm_negative_count: int = 0
    prm_status: Optional[str] = None

    # Aggregate
    failure_tags: list[str] = field(default_factory=list)
    fatal: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "family_id": self.family_id,
            "assistant_turns": self.assistant_turns,
            "total_turns": self.total_turns,
            "tool_calls_read": self.tool_calls_read,
            "tool_calls_write": self.tool_calls_write,
            "tool_calls_other": self.tool_calls_other,
            "transcript_read": self.transcript_read,
            "output_file_written": self.output_file_written,
            "output_file_exists": self.output_file_exists,
            "output_file_length": self.output_file_length,
            "thinking_chars": self.thinking_chars,
            "read_loop": self.read_loop,
            "empty_response": self.empty_response,
            "timed_out": self.timed_out,
            "execution_time": self.execution_time,
            "final_chat_chars": self.final_chat_chars,
            "output_budget_ratio": self.output_budget_ratio,
            "transcript_read_truncated": self.transcript_read_truncated,
            "max_read_size": self.max_read_size,
            "automated_breakdown": self.automated_breakdown,
            "automated_failed_checks": self.automated_failed_checks,
            "prm_turn_scores": self.prm_turn_scores,
            "prm_negative_count": self.prm_negative_count,
            "prm_status": self.prm_status,
            "failure_tags": self.failure_tags,
            "fatal": self.fatal,
            "notes": self.notes,
        }


def diagnose(
    *,
    trajectory: list[dict[str, Any]],
    workspace_path: Optional[str],
    task_id: str,
    plugin: Optional[TaskPlugin] = None,
    execution_time: float = 0.0,
    timed_out: bool = False,
    automated_breakdown: Optional[dict[str, float]] = None,
    prm_turn_scores: Optional[list[int]] = None,
    prm_status: Optional[str] = None,
) -> DiagnosticsResult:
    if plugin is None:
        plugin = resolve_plugin(task_id)

    diag = DiagnosticsResult(
        task_id=task_id,
        family_id=plugin.family_id if plugin else None,
        timed_out=timed_out,
        execution_time=execution_time,
    )

    expected_output_file = plugin.expected_output_file.get(task_id, "") if plugin else ""
    expected_input_files = set(plugin.expected_input_files) if plugin else set()
    min_output_chars = plugin.min_output_chars.get(task_id, 0) if plugin else 0

    read_paths, write_paths, read_sizes, write_records = _walk_trajectory(trajectory, diag)

    # Swarm/relay mode: the Lead dispatches a sub-agent (exec subagent.sh /
    # run_subagent.py) which reads the transcript and does the analysis. The
    # Lead's own reads=0 is correct-by-design, so transcript_not_read must NOT
    # be treated as fatal here. Non-swarm flows never match this, so behavior
    # is unchanged for them.
    delegated_to_subagent = _used_subagent(trajectory)

    # Layer 1: transcript_read
    for path in read_paths:
        leaf = _path_leaf(path)
        if leaf in expected_input_files or (expected_input_files and leaf in {f.lower() for f in expected_input_files}):
            diag.transcript_read = True
            break

    # Layer 1: output file
    # Source of truth priority:
    #   1. transcript-attested write content length (always available, immune to
    #      workspace overwrite by later bench tasks)
    #   2. workspace filesystem (more accurate if file still exists post-bench)
    if expected_output_file:
        target_leaf = expected_output_file.lower()
        write_content_chars = 0
        for path, content_chars in write_records:
            if _path_leaf(path) == target_leaf:
                diag.output_file_written = True
                # Latest write to expected output wins (model may overwrite).
                write_content_chars = content_chars
        if write_content_chars > 0:
            diag.output_file_length = write_content_chars

        if workspace_path:
            ws = Path(workspace_path)
            if ws.is_dir():
                target = ws / expected_output_file
                found = target if target.exists() else None
                if not found:
                    for f in ws.iterdir():
                        if f.name.lower() == expected_output_file.lower():
                            found = f
                            break
                if found:
                    diag.output_file_exists = True
                    try:
                        fs_len = len(found.read_text(encoding="utf-8", errors="replace"))
                    except Exception:
                        fs_len = 0
                    # Filesystem trumps transcript only if transcript said nothing
                    if write_content_chars == 0 and fs_len > 0:
                        diag.output_file_length = fs_len

    # Layer 1: read loop
    if read_paths:
        most = Counter(read_paths).most_common(1)[0][1]
        if most >= 3 and not write_paths:
            diag.read_loop = True

    # Layer 1: empty response
    if diag.assistant_turns == 0 or (diag.tool_calls_read == 0 and diag.tool_calls_write == 0):
        diag.empty_response = True

    # Layer 2: budget allocation + read truncation
    if read_sizes:
        diag.max_read_size = max(read_sizes)
        if diag.max_read_size >= READ_TRUNCATION_THRESHOLD:
            diag.transcript_read_truncated = True
    # Budget ratio only meaningful when the model actually wrote something.
    # If output_file_length is 0, we leave ratio as None — the "output not
    # written" failure tag covers that case directly.
    if diag.output_file_length > 0:
        denom = diag.output_file_length + diag.final_chat_chars
        if denom > 0:
            diag.output_budget_ratio = diag.output_file_length / denom

    # Layer 3: automated grading + PRM
    if automated_breakdown:
        diag.automated_breakdown = dict(automated_breakdown)
        diag.automated_failed_checks = sorted(
            k for k, v in automated_breakdown.items() if v < 0.5
        )
    if prm_turn_scores is not None:
        diag.prm_turn_scores = list(prm_turn_scores)
        diag.prm_negative_count = sum(1 for s in prm_turn_scores if s < 0)
    diag.prm_status = prm_status

    # Compile failure tags
    tags: list[str] = []
    if diag.timed_out:
        tags.append("timeout")
    if expected_input_files and not diag.transcript_read and not delegated_to_subagent:
        tags.append("transcript_not_read")
    if expected_output_file and not (diag.output_file_written or diag.output_file_exists):
        tags.append("output_not_written")
    if diag.output_file_exists and diag.output_file_length < 50:
        tags.append("output_too_short")
    if diag.thinking_chars > 5000:
        tags.append("excessive_thinking")
    if diag.read_loop:
        tags.append("read_loop")
    if diag.empty_response:
        tags.append("empty_response")
    if (
        diag.output_budget_ratio is not None
        and diag.output_file_length > 0
        and diag.output_budget_ratio < BUDGET_MISALLOC_THRESHOLD
    ):
        tags.append("output_budget_misallocated")
    if diag.transcript_read_truncated and diag.tool_calls_read <= 1:
        tags.append("transcript_read_truncated")
    if min_output_chars and diag.output_file_exists and diag.output_file_length < min_output_chars:
        tags.append("output_below_min")

    if plugin and plugin.custom_checks:
        try:
            extra = plugin.custom_checks(trajectory, workspace_path) or []
            tags.extend(extra)
        except Exception as e:  # pragma: no cover - defensive
            tags.append(f"custom_check_error:{type(e).__name__}")

    diag.failure_tags = tags
    diag.fatal = bool(set(tags) & FATAL_TAGS)
    diag.notes = _make_notes(diag)
    return diag


def _used_subagent(trajectory: list[dict[str, Any]]) -> bool:
    """True if the Lead dispatched a sub-agent (exec subagent.sh/run_subagent.py).

    In swarm/relay mode the sub-agent reads the transcript on the Lead's behalf,
    so the Lead reading nothing is expected rather than a fatal failure.
    """
    for event in trajectory:
        if event.get("type") != "message":
            continue
        msg = event.get("message", {})
        if msg.get("role") != "assistant":
            continue
        for item in msg.get("content", []):
            if not isinstance(item, dict) or item.get("type") != "toolCall":
                continue
            if str(item.get("name", "")).lower() != "exec":
                continue
            args = item.get("arguments", {}) or {}
            cmd = str(args.get("command", ""))
            joined = cmd + " " + " ".join(str(a) for a in (args.get("args") or []))
            if "subagent.sh" in joined or "run_subagent.py" in joined:
                return True
    return False


def _walk_trajectory(
    trajectory: list[dict[str, Any]],
    diag: DiagnosticsResult,
) -> tuple[list[str], list[str], list[int], list[tuple[str, int]]]:
    """Single pass: counts turns/tools/thinking, captures read/write paths and
    read result sizes, computes final_chat_chars (last assistant text turn).

    Returns:
        read_paths: paths read by the model
        write_paths: paths written by the model (legacy; same info as write_records)
        read_sizes: tool-result lengths for each read
        write_records: (path, content_chars) per write call, in chronological order
    """
    read_paths: list[str] = []
    write_paths: list[str] = []
    read_sizes: list[int] = []
    write_records: list[tuple[str, int]] = []
    last_assistant_text_chars = 0

    for raw in trajectory:
        if not isinstance(raw, dict):
            continue

        # Unwrap OpenClaw JSONL: {"type": "message", "message": {...}}
        if raw.get("type") == "message" and "message" in raw:
            entry = raw["message"]
        elif "role" in raw:
            entry = raw
        else:
            continue

        role = entry.get("role", "")
        diag.total_turns += 1
        content = entry.get("content", "")

        if role == "assistant":
            diag.assistant_turns += 1
            this_turn_text_chars = 0

            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    ptype = part.get("type", "")
                    if ptype == "thinking":
                        diag.thinking_chars += len(str(part.get("thinking", "")))
                    elif ptype == "text":
                        this_turn_text_chars += len(str(part.get("text", "")))
                    elif ptype == "toolCall":
                        name, path, content_chars = _name_path_content(part)
                        if name == "read":
                            diag.tool_calls_read += 1
                            if path:
                                read_paths.append(path)
                        elif name == "write":
                            diag.tool_calls_write += 1
                            if path:
                                write_paths.append(path)
                                write_records.append((path, content_chars))
                        elif name:
                            diag.tool_calls_other += 1
            elif isinstance(content, str):
                if "<think>" in content:
                    for m in re.findall(r"<think>(.*?)</think>", content, re.DOTALL):
                        diag.thinking_chars += len(m)
                this_turn_text_chars += len(content)

            # Flat tool_calls field (format A: chat-completion-style)
            for tc in entry.get("tool_calls", []) or []:
                name, path, content_chars = _flat_tool_call(tc)
                if name == "read":
                    diag.tool_calls_read += 1
                    if path:
                        read_paths.append(path)
                elif name == "write":
                    diag.tool_calls_write += 1
                    if path:
                        write_paths.append(path)
                        write_records.append((path, content_chars))
                elif name:
                    diag.tool_calls_other += 1

            if this_turn_text_chars > 0:
                last_assistant_text_chars = this_turn_text_chars

        elif role == "toolResult":
            tool_name = entry.get("toolName", "")
            if tool_name == "read":
                size = 0
                if isinstance(content, list):
                    for c in content:
                        if isinstance(c, dict) and c.get("type") == "text":
                            size += len(str(c.get("text", "")))
                read_sizes.append(size)

    diag.final_chat_chars = last_assistant_text_chars
    return read_paths, write_paths, read_sizes, write_records


def _name_path_content(part: dict) -> tuple[str, str, int]:
    """OpenClaw-style toolCall part: (name, path, content_chars).
    content_chars only meaningful for `write` calls; 0 otherwise."""
    name = part.get("name", "")
    args = part.get("arguments", "")
    args_dict = args if isinstance(args, dict) else _parse_json(str(args))
    path = ""
    content_chars = 0
    if isinstance(args_dict, dict):
        path = str(args_dict.get("path", args_dict.get("file_path", "")))
        if name == "write":
            content = args_dict.get("content") or args_dict.get("file_text") or args_dict.get("text") or ""
            content_chars = len(str(content))
    return name, path, content_chars


def _flat_tool_call(tc: Any) -> tuple[str, str, int]:
    """OpenAI chat-completion-style tool_call: (name, path, content_chars)."""
    if not isinstance(tc, dict):
        return "", "", 0
    fn = tc.get("function")
    if isinstance(fn, dict):
        name = str(fn.get("name", ""))
        args = fn.get("arguments", "")
    else:
        name = str(tc.get("name", ""))
        args = tc.get("arguments", "")
    args_dict = args if isinstance(args, dict) else _parse_json(str(args))
    path = ""
    content_chars = 0
    if isinstance(args_dict, dict):
        path = str(args_dict.get("path", args_dict.get("file_path", "")))
        if name == "write":
            content = args_dict.get("content") or args_dict.get("file_text") or args_dict.get("text") or ""
            content_chars = len(str(content))
    return name, path, content_chars


def _parse_json(s: str) -> Any:
    if not s:
        return None
    try:
        return json.loads(s)
    except (json.JSONDecodeError, TypeError):
        return None


def _path_leaf(path: str) -> str:
    return path.lower().replace("\\", "/").split("/")[-1]


def _make_notes(diag: DiagnosticsResult) -> str:
    parts: list[str] = []
    fatal_active = sorted(set(diag.failure_tags) & FATAL_TAGS)
    if fatal_active:
        parts.append(f"FATAL: {','.join(fatal_active)}")
    warn = [t for t in diag.failure_tags if t not in FATAL_TAGS]
    if warn:
        parts.append(f"warn=[{','.join(warn)}]")
    parts.append(f"reads={diag.tool_calls_read} writes={diag.tool_calls_write} turns={diag.assistant_turns}")
    if diag.output_file_exists:
        parts.append(f"out_len={diag.output_file_length}")
    if diag.output_budget_ratio is not None:
        parts.append(f"budget={diag.output_budget_ratio:.2f}")
    if diag.automated_failed_checks:
        parts.append(f"auto_fail={len(diag.automated_failed_checks)}")
    return " | ".join(parts)
