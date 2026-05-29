#!/usr/bin/env python3
"""Analyze OpenClaw trajectory JSONL files into capability reports.

This is intentionally heuristic and judge-free. It answers a narrower question
than benchmark grading: what process capability did this trajectory demonstrate
or miss? The output is meant to drive RL data construction from failure modes.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


MEETING_OUTPUT_FILES = {
    "task_meeting_advisory_stakeholders": "stakeholder_analysis.md",
    "task_meeting_council_votes": "votes_report.md",
    "task_meeting_gov_speaker_summary": "speaker_summary.md",
    "task_meeting_tech_action_items": "action_items.md",
    "task_meeting_sentiment_analysis": "sentiment_analysis.md",
}

MEETING_INPUT_FILES = {
    "transcript.md",
    "meeting_transcript.md",
    "meeting-transcript.md",
}

TASK_KEYWORDS = {
    "task_meeting_council_votes": {
        "motion",
        "vote",
        "voted",
        "roll call",
        "ayes",
        "nays",
        "opposed",
        "recuse",
        "recusal",
        "abstain",
        "dissent",
        "unanimous",
        "second",
    },
    "task_meeting_tech_action_items": {
        "action",
        "owner",
        "deadline",
        "follow up",
        "todo",
        "task",
        "commitment",
        "next step",
    },
    "task_meeting_gov_speaker_summary": {
        "speaker",
        "presenter",
        "quote",
        "key point",
        "testimony",
        "question",
    },
    "task_meeting_advisory_stakeholders": {
        "stakeholder",
        "interest",
        "concern",
        "position",
        "influence",
        "tension",
        "agreement",
    },
    "task_meeting_sentiment_analysis": {
        "sentiment",
        "tone",
        "emotion",
        "concern",
        "energy",
        "confidence",
        "friction",
    },
}

CHECK_KEYWORDS = {
    "verify",
    "verified",
    "check",
    "cross-check",
    "recheck",
    "validate",
    "confirm",
    "count",
    "coverage",
    "missing",
    "audit",
}

LEDGER_KEYWORDS = {
    "ledger",
    "table",
    "checklist",
    "chronological",
    "item",
    "owner",
    "deadline",
    "count",
    "mover",
    "seconder",
    "speaker",
}


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return "" if content is None else json.dumps(content, ensure_ascii=False)
    parts: list[str] = []
    for item in content:
        if not isinstance(item, dict):
            continue
        for key in ("text", "thinking", "content"):
            val = item.get(key)
            if isinstance(val, str):
                parts.append(val)
    return "\n".join(parts)


def _leaf(path: Any) -> str:
    return Path(str(path or "")).name.lower()


def _norm_path(path: Any) -> str:
    return re.sub(r"/tmp/(pinchbench|pinchbench_[^/]+)/[^/]+/agent_workspace/", "", str(path or ""))


def _score(value: float) -> float:
    return round(max(0.0, min(1.0, value)), 3)


def _level(score: float) -> str:
    if score >= 0.8:
        return "strong"
    if score >= 0.55:
        return "partial"
    if score >= 0.3:
        return "weak"
    return "missing"


@dataclass
class ToolCall:
    index: int
    name: str
    arguments: dict[str, Any]


@dataclass
class ParsedTrajectory:
    path: str
    task_id: str
    model_id: str | None = None
    messages: list[dict[str, Any]] = field(default_factory=list)
    tool_calls: list[ToolCall] = field(default_factory=list)
    tool_results: list[dict[str, Any]] = field(default_factory=list)
    assistant_text: str = ""
    assistant_thinking: str = ""
    final_text: str = ""
    write_records: list[dict[str, Any]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def infer_task_id(path: Path, explicit: str | None) -> str:
    if explicit:
        return explicit
    name = path.name
    if name.endswith(".jsonl"):
        name = name[:-6]
    if "__" in name:
        name = name.split("__", 1)[0]
    return name


def parse_trajectory(path: Path, task_id: str) -> ParsedTrajectory:
    parsed = ParsedTrajectory(path=str(path), task_id=task_id)
    for line_no, line in enumerate(path.read_text("utf-8", errors="replace").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            parsed.errors.append(f"json_parse_error:{line_no}:{exc}")
            continue

        if row.get("type") == "model_change":
            parsed.model_id = row.get("modelId")
            continue
        if row.get("type") != "message":
            continue

        msg = row.get("message") or {}
        role = msg.get("role")
        content = msg.get("content")
        text = _content_text(content)
        record = {
            "role": role,
            "text": text,
            "is_error": bool(msg.get("isError")),
            "tool_name": msg.get("toolName"),
        }
        parsed.messages.append(record)

        if role == "assistant":
            last_text = ""
            for item in content or []:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "text":
                    val = str(item.get("text") or "")
                    parsed.assistant_text += "\n" + val
                    last_text = val
                elif item_type == "thinking":
                    parsed.assistant_thinking += "\n" + str(item.get("thinking") or "")
                elif item_type == "toolCall":
                    name = str(item.get("name") or "")
                    args = item.get("arguments") or {}
                    if not isinstance(args, dict):
                        args = {"raw": args}
                    call = ToolCall(index=len(parsed.tool_calls), name=name, arguments=args)
                    parsed.tool_calls.append(call)
                    if name.lower() == "write":
                        parsed.write_records.append(
                            {
                                "path": _norm_path(args.get("path")),
                                "content": str(args.get("content") or ""),
                                "chars": len(str(args.get("content") or "")),
                                "turn_index": len(parsed.messages) - 1,
                            }
                        )
            if last_text:
                parsed.final_text = last_text
        elif role == "toolResult":
            parsed.tool_results.append(record)
            if record["is_error"] or re.search(r"\berror\b|not found|no such file|failed", text, re.I):
                parsed.errors.append(text[:300])
    return parsed


def _tool_paths(parsed: ParsedTrajectory, name: str) -> list[str]:
    out = []
    for call in parsed.tool_calls:
        if call.name.lower() == name.lower():
            out.append(_norm_path(call.arguments.get("path") or call.arguments.get("file") or ""))
    return out


def _command_texts(parsed: ParsedTrajectory) -> list[str]:
    texts = []
    for call in parsed.tool_calls:
        if call.name.lower() in {"exec", "bash", "shell"}:
            texts.append(str(call.arguments.get("command") or call.arguments))
    return texts


def _result_texts(parsed: ParsedTrajectory, tool_name: str | None = None) -> list[str]:
    rows = []
    for result in parsed.tool_results:
        if tool_name is None or str(result.get("tool_name") or "").lower() == tool_name.lower():
            rows.append(str(result.get("text") or ""))
    return rows


def _read_offsets(parsed: ParsedTrajectory) -> list[int]:
    offsets = []
    for call in parsed.tool_calls:
        if call.name.lower() == "read":
            raw = call.arguments.get("offset", 0)
            try:
                offsets.append(int(raw or 0))
            except (TypeError, ValueError):
                offsets.append(0)
    return offsets


def capability_report(parsed: ParsedTrajectory) -> dict[str, Any]:
    task_id = parsed.task_id
    expected_output = MEETING_OUTPUT_FILES.get(task_id, "")
    task_keywords = TASK_KEYWORDS.get(task_id, set())

    read_paths = _tool_paths(parsed, "read")
    write_paths = _tool_paths(parsed, "write")
    command_text = "\n".join(_command_texts(parsed))
    all_assistant = "\n".join([parsed.assistant_thinking, parsed.assistant_text])
    all_output = "\n".join([r["content"] for r in parsed.write_records] + [parsed.final_text])
    read_results = _result_texts(parsed, "read")
    max_read_chars = max((len(t) for t in read_results), default=0)
    offsets = _read_offsets(parsed)
    distinct_offsets = sorted(set(offsets))
    transcript_reads = [p for p in read_paths if _leaf(p) in MEETING_INPUT_FILES or "transcript" in _leaf(p)]

    grep_hits = sum(1 for cmd in _command_texts(parsed) if re.search(r"\b(grep|rg|awk|sed)\b", cmd))
    keyword_search_hits = sum(1 for kw in task_keywords if re.search(re.escape(kw), command_text, re.I))
    keyword_output_hits = sum(1 for kw in task_keywords if re.search(re.escape(kw), all_output, re.I))

    output_writes = [
        rec for rec in parsed.write_records
        if not expected_output or _leaf(rec["path"]) == expected_output.lower()
    ]
    output_chars = max((rec["chars"] for rec in output_writes), default=0)
    any_write_chars = sum(rec["chars"] for rec in parsed.write_records)

    last_write_turn = max((rec["turn_index"] for rec in parsed.write_records), default=-1)
    reads_after_write = 0
    checks_after_write = 0
    for idx, msg in enumerate(parsed.messages):
        if idx <= last_write_turn:
            continue
        if msg.get("role") == "assistant" and re.search("|".join(map(re.escape, CHECK_KEYWORDS)), msg.get("text") or "", re.I):
            checks_after_write += 1
        if msg.get("role") == "toolResult" and str(msg.get("tool_name") or "").lower() == "read":
            reads_after_write += 1

    all_check_text = "\n".join([all_assistant, all_output, command_text])
    check_hits = sum(1 for kw in CHECK_KEYWORDS if re.search(re.escape(kw), all_check_text, re.I))
    ledger_hits = sum(1 for kw in LEDGER_KEYWORDS if re.search(re.escape(kw), all_output, re.I))
    table_like = all_output.count("|") >= 8 or len(re.findall(r"^\s*[-*]\s+", all_output, flags=re.M)) >= 8
    tool_error_count = len(parsed.errors)
    tool_calls_by_name = Counter(call.name.lower() for call in parsed.tool_calls)

    caps: dict[str, dict[str, Any]] = {}

    score = 0.0
    evidence = []
    gaps = []
    if transcript_reads:
        score += 0.45
        evidence.append(f"read transcript {len(transcript_reads)} time(s)")
    else:
        gaps.append("did not read a transcript-like source file")
    if len(distinct_offsets) >= 3:
        score += 0.3
        evidence.append(f"covered {len(distinct_offsets)} distinct read offsets")
    elif len(transcript_reads) >= 2:
        score += 0.15
        evidence.append("used repeated reads but limited offset diversity")
    else:
        gaps.append("limited pagination over long transcript")
    if grep_hits:
        score += 0.15
        evidence.append(f"used search/slicing command(s): {grep_hits}")
    if max_read_chars >= 39900 and len(distinct_offsets) <= 1:
        score -= 0.25
        gaps.append("read result likely truncated without pagination")
    if len(transcript_reads) <= 1 and not grep_hits:
        gaps.append("single-pass read/search strategy")
    caps["long_context_coverage"] = _cap(score, evidence, gaps)

    score = 0.0
    evidence = []
    gaps = []
    if task_keywords:
        if keyword_search_hits:
            score += min(0.45, 0.08 * keyword_search_hits)
            evidence.append(f"searched task keywords: {keyword_search_hits}/{len(task_keywords)}")
        else:
            gaps.append("no task-specific search query observed")
        if keyword_output_hits:
            score += min(0.35, 0.05 * keyword_output_hits)
            evidence.append(f"final output mentions task concepts: {keyword_output_hits}/{len(task_keywords)}")
        else:
            gaps.append("final output lacks task-specific anchors")
    else:
        score += 0.4
    if grep_hits or len(distinct_offsets) >= 2:
        score += 0.2
    caps["targeted_evidence_extraction"] = _cap(score, evidence, gaps)

    score = 0.0
    evidence = []
    gaps = []
    if ledger_hits:
        score += min(0.45, 0.06 * ledger_hits)
        evidence.append(f"ledger/table/checklist terms in output: {ledger_hits}")
    else:
        gaps.append("no explicit ledger/checklist structure found")
    if table_like:
        score += 0.25
        evidence.append("table or substantial bullet structure found")
    if task_id == "task_meeting_council_votes" and re.search(r"\btotal\b|\bsummary\b|\bcount\b", all_output, re.I):
        score += 0.15
        evidence.append("vote summary/count language present")
    if output_chars >= 1200 or any_write_chars >= 1200:
        score += 0.15
    elif output_chars < 500:
        gaps.append("written artifact is short for this task")
    caps["intermediate_ledger_and_structure"] = _cap(score, evidence, gaps)

    score = 0.0
    evidence = []
    gaps = []
    if check_hits:
        score += min(0.45, 0.06 * check_hits)
        evidence.append(f"verification/check terms found: {check_hits}")
    else:
        gaps.append("no explicit verification/checking language")
    if reads_after_write:
        score += 0.25
        evidence.append(f"read after writing draft: {reads_after_write}")
    if checks_after_write:
        score += 0.2
        evidence.append(f"post-write assistant check turn(s): {checks_after_write}")
    if task_id == "task_meeting_council_votes" and not re.search(r"\bcount\b|\btotal\b", all_output, re.I):
        gaps.append("vote task lacks final count consistency check")
    caps["verification_and_self_correction"] = _cap(score, evidence, gaps)

    score = 0.0
    evidence = []
    gaps = []
    if output_writes:
        score += 0.45
        evidence.append(f"wrote expected output file {expected_output or output_writes[-1]['path']}")
    else:
        gaps.append(f"did not write expected output file {expected_output}" if expected_output else "no output write")
    if output_chars >= 1000:
        score += 0.3
        evidence.append(f"expected output length {output_chars} chars")
    elif output_chars >= 300:
        score += 0.15
        evidence.append(f"expected output length {output_chars} chars")
    else:
        gaps.append("expected output too short or missing")
    if parsed.final_text and not re.search(r"not found|cannot|unable|error", parsed.final_text, re.I):
        score += 0.15
    if tool_error_count == 0:
        score += 0.1
    caps["output_delivery"] = _cap(score, evidence, gaps)

    score = 0.8
    evidence = []
    gaps = []
    if tool_error_count:
        score -= min(0.5, 0.15 * tool_error_count)
        gaps.append(f"tool/file errors observed: {tool_error_count}")
    else:
        evidence.append("no obvious tool/file errors")
    if re.search(r"file (was )?not found|no such file|cannot find", parsed.final_text, re.I):
        score -= 0.35
        gaps.append("final answer reports file-not-found instead of recovering")
    if tool_error_count and len(parsed.tool_calls) > tool_error_count + 2:
        score += 0.15
        evidence.append("continued with additional tool calls after error")
    caps["tool_error_recovery"] = _cap(score, evidence, gaps)

    failure_modes = []
    for name, cap in caps.items():
        if cap["score"] < 0.55:
            failure_modes.append(
                {
                    "capability": name,
                    "severity": "high" if cap["score"] < 0.3 else "medium",
                    "gaps": cap["gaps"],
                }
            )

    training_targets = _training_targets(task_id, caps)

    return {
        "path": parsed.path,
        "task_id": task_id,
        "model_id": parsed.model_id,
        "metrics": {
            "assistant_turns": sum(1 for m in parsed.messages if m.get("role") == "assistant"),
            "tool_calls": len(parsed.tool_calls),
            "tool_calls_by_name": dict(tool_calls_by_name),
            "tool_results": len(parsed.tool_results),
            "transcript_reads": len(transcript_reads),
            "distinct_read_offsets": distinct_offsets,
            "max_read_result_chars": max_read_chars,
            "grep_or_slice_commands": grep_hits,
            "write_records": [
                {"path": rec["path"], "chars": rec["chars"]} for rec in parsed.write_records
            ],
            "expected_output_chars": output_chars,
            "tool_error_count": tool_error_count,
            "thinking_chars": len(parsed.assistant_thinking),
            "final_text_chars": len(parsed.final_text),
        },
        "capabilities": caps,
        "failure_modes": failure_modes,
        "rl_training_targets": training_targets,
    }


def _cap(score: float, evidence: list[str], gaps: list[str]) -> dict[str, Any]:
    score = _score(score)
    if score < 0.55 and not gaps:
        gaps = ["weak evidence for this capability in the trajectory"]
    return {
        "score": score,
        "level": _level(score),
        "evidence": evidence,
        "gaps": gaps,
    }


def _training_targets(task_id: str, caps: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    targets = []
    if caps["long_context_coverage"]["score"] < 0.65:
        targets.append(
            {
                "name": "long_transcript_coverage_policy",
                "objective": "learn to page/search the full meeting transcript before final synthesis",
                "positive_trace_features": ["multiple read offsets", "grep/rg/sed over task keywords", "coverage checklist"],
            }
        )
    if caps["intermediate_ledger_and_structure"]["score"] < 0.65:
        targets.append(
            {
                "name": "evidence_ledger_construction",
                "objective": "build a durable intermediate table/ledger before final answer",
                "positive_trace_features": ["write or maintain structured rows", "item/owner/result/count columns", "final derived from ledger"],
            }
        )
    if caps["verification_and_self_correction"]["score"] < 0.65:
        targets.append(
            {
                "name": "pre_final_verification_policy",
                "objective": "verify coverage and consistency before final output",
                "positive_trace_features": ["read back critical spans", "count consistency check", "explicit missing-item audit"],
            }
        )
    if caps["tool_error_recovery"]["score"] < 0.65:
        targets.append(
            {
                "name": "tool_error_recovery_policy",
                "objective": "recover from missing file/path/tool errors instead of finalizing failure",
                "positive_trace_features": ["list workspace", "try alternate transcript names", "continue after failed read"],
            }
        )
    if task_id == "task_meeting_council_votes":
        targets.insert(
            0,
            {
                "name": "vote_ledger_policy",
                "objective": "extract every motion/vote, including late substantive votes, dissents, abstentions, recusals, and reconsiderations",
                "positive_trace_features": [
                    "search motion/vote/roll call/opposed/recuse/dissent",
                    "read context around every hit",
                    "ledger with item, mover, seconder, result, exceptions",
                    "total count check",
                ],
            },
        )
    return targets


def render_markdown(reports: list[dict[str, Any]]) -> str:
    lines = ["# Trajectory Capability Report", ""]
    if len(reports) > 1:
        avg = Counter()
        counts = Counter()
        for report in reports:
            for name, cap in report["capabilities"].items():
                avg[name] += cap["score"]
                counts[name] += 1
        lines.extend(["## Aggregate", ""])
        for name in sorted(avg):
            score = avg[name] / counts[name]
            lines.append(f"- **{name}**: {score:.3f} ({_level(score)})")
        lines.append("")

    for report in reports:
        lines.extend([
            f"## {report['task_id']}",
            "",
            f"- Path: `{report['path']}`",
            f"- Model: `{report.get('model_id') or 'unknown'}`",
            f"- Tool calls: `{report['metrics']['tool_calls']}` {report['metrics']['tool_calls_by_name']}",
            f"- Transcript reads: `{report['metrics']['transcript_reads']}` offsets `{report['metrics']['distinct_read_offsets']}`",
            f"- Expected output chars: `{report['metrics']['expected_output_chars']}`",
            "",
            "| Capability | Score | Level | Main gaps |",
            "|---|---:|---|---|",
        ])
        for name, cap in report["capabilities"].items():
            gaps = "; ".join(cap["gaps"][:3]) if cap["gaps"] else "-"
            lines.append(f"| `{name}` | {cap['score']:.3f} | {cap['level']} | {gaps} |")
        if report["failure_modes"]:
            lines.extend(["", "### Failure Modes", ""])
            for item in report["failure_modes"]:
                lines.append(f"- `{item['severity']}` `{item['capability']}`: {'; '.join(item['gaps'])}")
        if report["rl_training_targets"]:
            lines.extend(["", "### RL Training Targets", ""])
            for item in report["rl_training_targets"]:
                lines.append(f"- `{item['name']}`: {item['objective']}")
        lines.append("")
    return "\n".join(lines)


def collect_paths(args: argparse.Namespace) -> list[Path]:
    paths = [Path(p) for p in args.transcript]
    for raw_dir in args.transcripts_dir:
        d = Path(raw_dir)
        paths.extend(sorted(d.glob("*.jsonl")))
    seen = set()
    out = []
    for path in paths:
        key = str(path)
        if key not in seen:
            seen.add(key)
            out.append(path)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Analyze OpenClaw trajectory capabilities.")
    parser.add_argument("--transcript", action="append", default=[], help="OpenClaw transcript JSONL. Repeatable.")
    parser.add_argument("--transcripts-dir", action="append", default=[], help="Directory of transcript JSONL files. Repeatable.")
    parser.add_argument("--task-id", help="Override task id for single transcript input.")
    parser.add_argument("--output-json", help="Write JSON report.")
    parser.add_argument("--output-md", help="Write Markdown report.")
    args = parser.parse_args()

    paths = collect_paths(args)
    if not paths:
        raise SystemExit("No --transcript or --transcripts-dir input provided")

    reports = []
    for path in paths:
        if not path.is_file():
            reports.append({"path": str(path), "error": "missing transcript"})
            continue
        task_id = infer_task_id(path, args.task_id if len(paths) == 1 else None)
        reports.append(capability_report(parse_trajectory(path, task_id)))

    payload = {
        "reports": reports,
        "summary": {
            "num_trajectories": len(reports),
            "num_errors": sum(1 for r in reports if "error" in r),
        },
    }

    if args.output_json:
        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", "utf-8")
        print(f"wrote json: {out}")
    if args.output_md:
        out = Path(args.output_md)
        out.parent.mkdir(parents=True, exist_ok=True)
        valid_reports = [r for r in reports if "capabilities" in r]
        out.write_text(render_markdown(valid_reports), "utf-8")
        print(f"wrote markdown: {out}")
    if not args.output_json and not args.output_md:
        print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
