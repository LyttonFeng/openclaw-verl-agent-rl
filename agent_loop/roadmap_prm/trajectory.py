"""
Trajectory parser for OpenClaw JSONL transcripts.

A "turn" = one assistant message with its associated tool calls and tool results.
Each turn carries enough context for a judge to score it:
  - thinking content
  - tool calls made (with truncated args)
  - tool results received (truncated)
  - final text output (if any)
  - turn index in the trajectory
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


TOOL_RESULT_TRUNC = 600  # chars per tool result shown to judge


@dataclass
class Turn:
    index: int  # 0-based turn index
    thinking: str = ""
    tool_calls: list[dict[str, Any]] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    text: str = ""  # final assistant text (non-tool)

    def to_judge_text(self) -> str:
        parts = [f"--- Turn {self.index} ---"]
        if self.thinking:
            think_preview = self.thinking[:1500]
            if len(self.thinking) > 1500:
                think_preview += f"\n[... truncated, {len(self.thinking)} chars total]"
            parts.append(f"[thinking]\n{think_preview}")
        for tc in self.tool_calls:
            name = tc.get("name", "?")
            inp = json.dumps(tc.get("input", {}), ensure_ascii=False)
            if len(inp) > 400:
                inp = inp[:400] + "..."
            parts.append(f"[tool_call:{name}] {inp}")
        for tr in self.tool_results:
            name = tr.get("tool_name", "?")
            content = tr.get("content", "")
            if len(content) > TOOL_RESULT_TRUNC:
                content = content[:TOOL_RESULT_TRUNC] + f"\n[... truncated, {len(content)} chars total]"
            parts.append(f"[tool_result:{name}]\n{content}")
        if self.text:
            parts.append(f"[assistant_text]\n{self.text}")
        return "\n".join(parts)


def _extract_text_from_content(content: Any) -> tuple[str, list[dict], str]:
    """
    Parse the `content` field of an assistant message.

    OpenClaw assistant content is a list with parts of type:
      - "text" -> text
      - "thinking" / "reasoning" -> reasoning
      - "toolCall" with keys {id, name, arguments}
    """
    thinking = ""
    tool_calls = []
    text = ""

    if isinstance(content, str):
        text = content
        return thinking, tool_calls, text

    if not isinstance(content, list):
        return thinking, tool_calls, text

    for part in content:
        if not isinstance(part, dict):
            continue
        ptype = part.get("type", "")
        if ptype in ("thinking", "reasoning"):
            # DSv4 uses {"text": ...}; Qwen3 uses {"thinking": ...}
            thinking += (
                part.get("thinking")
                or part.get("text")
                or part.get("content")
                or part.get("reasoning")
                or ""
            )
        elif ptype == "text":
            text += part.get("text", "") or ""
        elif ptype in ("toolCall", "tool_use", "tool_call"):
            args = part.get("arguments")
            if args is None:
                args = part.get("input") or part.get("args") or {}
            tool_calls.append({
                "name": part.get("name") or part.get("toolName", "?"),
                "input": args,
                "id": part.get("id") or part.get("toolCallId", "") or "",
            })
    return thinking, tool_calls, text


def parse_openclaw_jsonl(path: str | Path) -> list[Turn]:
    """
    Parse an OpenClaw v3 transcript JSONL into a list of Turn objects.

    OpenClaw v3 format:
      Each line: {"type": "message", "id": ..., "message": {"role": ..., "content": ...}}
      role can be: "user", "assistant", "toolResult"
      For toolResult: message has top-level keys {role, toolCallId, toolName, content}.
      content for toolResult is a list of {"type":"text","text":"..."}
    """
    lines = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                lines.append(json.loads(line))

    # Pass 1: collect tool results indexed by toolCallId
    tool_results_by_id: dict[str, dict] = {}
    for entry in lines:
        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {})
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role != "toolResult":
            continue
        tcid = msg.get("toolCallId", "")
        if not tcid:
            continue
        tool_results_by_id[tcid] = {
            "tool_name": msg.get("toolName", "?"),
            "content": _stringify_tool_result(msg.get("content")),
            "is_error": bool(msg.get("isError", False)),
        }

    # Pass 2: build turns from assistant messages
    turns: list[Turn] = []
    for entry in lines:
        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {})
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role != "assistant":
            continue
        thinking, tool_calls, text = _extract_text_from_content(msg.get("content"))
        # Attach tool results for this turn's tool calls
        attached_results = []
        for tc in tool_calls:
            tr = tool_results_by_id.get(tc.get("id", ""))
            if tr:
                attached_results.append(tr)
        # Skip empty assistant messages (no content at all)
        if not thinking and not tool_calls and not text:
            continue
        turns.append(Turn(
            index=len(turns),
            thinking=thinking,
            tool_calls=tool_calls,
            tool_results=attached_results,
            text=text,
        ))

    return turns


def _stringify_tool_result(content: Any) -> str:
    """Convert a tool_result content (list-of-parts or str) to a flat string."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        out = []
        for sub in content:
            if isinstance(sub, dict):
                out.append(sub.get("text") or sub.get("content") or json.dumps(sub, ensure_ascii=False))
            else:
                out.append(str(sub))
        return "\n".join(out)
    if content is None:
        return ""
    return json.dumps(content, ensure_ascii=False)


def render_history(turns: list[Turn], up_to: int) -> str:
    """Render turns 0..up_to-1 as compact history for the judge prompt."""
    if up_to == 0:
        return "(no prior turns)"
    return "\n\n".join(t.to_judge_text() for t in turns[:up_to])
