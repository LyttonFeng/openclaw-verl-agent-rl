"""Markdown + JSON output for batches of DiagnosticsResult."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from .core import FATAL_TAGS, DiagnosticsResult


def render_markdown_report(
    diags: Iterable[DiagnosticsResult],
    *,
    source: str = "",
    overall_score: Optional[float] = None,
    per_task_means: Optional[dict[str, float]] = None,
    per_task_failed_checks: Optional[dict[str, list[list[str]]]] = None,
) -> str:
    """Render a markdown report from a list of DiagnosticsResult.

    Args:
        diags: one DiagnosticsResult per (task, run) — typically deduplicated to
            one per task at bench time, or one per (task, response) at rollout.
        source: free-form provenance string for the report header.
        overall_score: optional 3-run overall mean from result.json.
        per_task_means: optional task_id → 3-run mean score.
        per_task_failed_checks: optional task_id → list of failed-check name
            lists (one per grading run). Used to surface "stable failure across
            runs" pattern.
    """
    diags = list(diags)
    n_total = len(diags)
    n_fatal = sum(1 for d in diags if d.fatal)
    n_warn = sum(1 for d in diags if d.failure_tags and not d.fatal)

    lines: list[str] = []
    lines.append(f"# Diagnosis report")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    if source:
        lines.append(f"Source: {source}")
    lines.append("")

    # Overall
    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Diagnosed entries: {n_total}")
    lines.append(f"- Fatal: {n_fatal} / Warnings: {n_warn} / Healthy: {n_total - n_fatal - n_warn}")
    if overall_score is not None:
        lines.append(f"- Overall mean score: {overall_score:.4f}")
    lines.append("")

    # Failure-tag distribution
    tag_counter: Counter[str] = Counter()
    tag_to_tasks: dict[str, list[str]] = defaultdict(list)
    for d in diags:
        for t in d.failure_tags:
            tag_counter[t] += 1
            tag_to_tasks[t].append(d.task_id)
    if tag_counter:
        lines.append("## Failure-tag distribution")
        lines.append("")
        lines.append("| tag | count | level | tasks |")
        lines.append("|---|---|---|---|")
        for tag, count in tag_counter.most_common():
            level = "FATAL" if tag in FATAL_TAGS else "warn"
            tasks = ", ".join(sorted(set(tag_to_tasks[tag])))
            lines.append(f"| `{tag}` | {count} | {level} | {tasks} |")
        lines.append("")
    else:
        lines.append("## Failure-tag distribution")
        lines.append("")
        lines.append("_No tags raised._")
        lines.append("")

    # Per-task table
    lines.append("## Per-task")
    lines.append("")
    lines.append("| task | mean | turns | reads | writes | thinking | out_len | budget | tags |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for d in diags:
        mean_str = (
            f"{per_task_means[d.task_id]:.3f}"
            if per_task_means and d.task_id in per_task_means
            else "—"
        )
        budget_str = (
            f"{d.output_budget_ratio:.2f}"
            if d.output_budget_ratio is not None
            else "—"
        )
        tags_str = ", ".join(f"`{t}`" for t in d.failure_tags) if d.failure_tags else "—"
        lines.append(
            f"| {d.task_id} | {mean_str} | {d.assistant_turns} | "
            f"{d.tool_calls_read} | {d.tool_calls_write} | {d.thinking_chars} | "
            f"{d.output_file_length} | {budget_str} | {tags_str} |"
        )
    lines.append("")

    # Failed automated checks (cross-run pattern)
    if per_task_failed_checks:
        lines.append("## Failed automated checks (across runs)")
        lines.append("")
        for task_id, runs_of_failed in per_task_failed_checks.items():
            if not any(runs_of_failed):
                continue
            lines.append(f"### {task_id}")
            lines.append("")
            lines.append("| run | failed checks |")
            lines.append("|---|---|")
            for i, failed in enumerate(runs_of_failed, 1):
                if failed:
                    lines.append(f"| {i} | {', '.join(f'`{c}`' for c in failed)} |")
                else:
                    lines.append(f"| {i} | — |")
            sets = [set(r) for r in runs_of_failed if r]
            if len(sets) >= 2:
                stable = set.intersection(*sets) if sets else set()
                if stable:
                    lines.append("")
                    lines.append(
                        f"**Stable across all runs:** {', '.join(f'`{c}`' for c in sorted(stable))} "
                        f"— this is a deterministic failure pattern, not stochastic noise."
                    )
            lines.append("")

    # Notable trajectories — only those with at least one tag
    notable = [d for d in diags if d.failure_tags]
    if notable:
        lines.append("## Notable trajectories")
        lines.append("")
        for d in notable:
            lines.append(f"### {d.task_id}")
            lines.append("")
            lines.append(f"- {d.notes}")
            if d.output_budget_ratio is not None and d.output_file_length > 0:
                lines.append(
                    f"- Output budget: file {d.output_file_length} chars, "
                    f"final chat {d.final_chat_chars} chars, ratio {d.output_budget_ratio:.2f}"
                )
            if d.transcript_read_truncated:
                lines.append(
                    f"- Transcript read returned {d.max_read_size} chars (≥{39900}) "
                    f"with only {d.tool_calls_read} read call(s) — likely truncated."
                )
            if d.automated_failed_checks:
                lines.append(
                    f"- Automated checks failed ({len(d.automated_failed_checks)}): "
                    f"{', '.join(f'`{c}`' for c in d.automated_failed_checks)}"
                )
            lines.append("")

    return "\n".join(lines)


def dump_json(diags: Iterable[DiagnosticsResult], path: Path) -> None:
    Path(path).write_text(
        json.dumps([d.to_dict() for d in diags], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
