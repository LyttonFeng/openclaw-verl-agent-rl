"""CLI: `python -m agent_loop.diagnostics analyze ...`

Reads a benchmark `result.json`, locates per-task transcripts under one or
more transcript directories, runs `diagnose()` on each, and writes markdown +
json reports.

Each `tasks[]` entry in result.json typically has its workspace at
`/tmp/pinchbench/<NNNN>/agent_workspace` and a transcript stored under
`<results-dir>/<NNNN>_transcripts/<task_id>.jsonl`. We accept the transcript
parent dirs explicitly (`--transcripts-dirs`) so this works for any
benchmarking layout.

Note: result.json often contains the same task_id repeated `runs` times with
identical grading payloads (a quirk of how PinchBench writes its result).
We dedup by task_id so each task gets exactly one diagnosis row, even though
the grading still reflects all runs (mean / per-run breakdowns).
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Optional

from .core import diagnose
from .reporters import dump_json, render_markdown_report


def _load_transcript(path: Path) -> list[dict]:
    out = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def _find_transcript(
    task_id: str,
    transcripts_dirs: list[Path],
) -> Optional[Path]:
    """Look for `<dir>/<task_id>.jsonl` in each given dir."""
    for d in transcripts_dirs:
        candidate = d / f"{task_id}.jsonl"
        if candidate.exists():
            return candidate
    return None


def _aggregate_breakdown(runs: list[dict]) -> dict[str, float]:
    """Mean each breakdown key across grading runs."""
    sums: dict[str, float] = defaultdict(float)
    counts: dict[str, int] = defaultdict(int)
    for r in runs:
        bd = r.get("breakdown") or {}
        for k, v in bd.items():
            try:
                sums[k] += float(v)
                counts[k] += 1
            except (TypeError, ValueError):
                continue
    return {k: sums[k] / counts[k] for k in sums if counts[k] > 0}


def cmd_analyze(args: argparse.Namespace) -> int:
    result_json = Path(args.result_json)
    if not result_json.exists():
        print(f"ERROR: result.json not found: {result_json}", file=sys.stderr)
        return 1

    with open(result_json) as f:
        result = json.load(f)

    transcripts_dirs = [Path(d) for d in args.transcripts_dirs]
    for d in transcripts_dirs:
        if not d.exists():
            print(f"WARN: transcripts dir not found: {d}", file=sys.stderr)

    # Dedup tasks[] by task_id (keep first)
    seen: dict[str, dict] = {}
    for t in result.get("tasks") or []:
        tid = t.get("task_id")
        if tid and tid not in seen:
            seen[tid] = t

    diags = []
    per_task_means: dict[str, float] = {}
    per_task_failed_checks: dict[str, list[list[str]]] = {}

    for tid, t in seen.items():
        grading = t.get("grading") or {}
        runs = grading.get("runs") or []
        per_task_means[tid] = grading.get("mean", 0.0) if runs else 0.0

        # Per-run failed-check lists (for cross-run stability analysis)
        per_run_failed: list[list[str]] = []
        for r in runs:
            bd = r.get("breakdown") or {}
            failed = sorted(k for k, v in bd.items() if isinstance(v, (int, float)) and v < 0.5)
            per_run_failed.append(failed)
        per_task_failed_checks[tid] = per_run_failed

        # Aggregate breakdown across grading runs (mean per check)
        agg_breakdown = _aggregate_breakdown(runs)

        # Locate transcript + workspace
        transcript_path = _find_transcript(tid, transcripts_dirs)
        if transcript_path is None:
            print(f"WARN: no transcript for {tid} in any --transcripts-dirs", file=sys.stderr)
            trajectory: list[dict] = []
        else:
            trajectory = _load_transcript(transcript_path)

        workspace_path = t.get("workspace") or None

        diag = diagnose(
            trajectory=trajectory,
            workspace_path=workspace_path,
            task_id=tid,
            execution_time=t.get("execution_time", 0.0),
            timed_out=t.get("timed_out", False),
            automated_breakdown=agg_breakdown,
        )
        diags.append(diag)

    # Overall score: prefer category_scores.<category>.pct/100 (PinchBench format
    # stores absolute score+max_score+pct); else mean of per_task_means.
    overall = None
    cs = result.get("category_scores") or {}
    if cs:
        first_val = next(iter(cs.values()), None)
        if isinstance(first_val, dict):
            if "pct" in first_val:
                overall = float(first_val["pct"]) / 100.0
            elif "score" in first_val and "max_score" in first_val and first_val["max_score"]:
                overall = float(first_val["score"]) / float(first_val["max_score"])
            elif "mean" in first_val:
                overall = float(first_val["mean"])
        elif isinstance(first_val, (int, float)):
            overall = float(first_val)
    if overall is None and per_task_means:
        vals = list(per_task_means.values())
        overall = sum(vals) / len(vals)

    md = render_markdown_report(
        diags,
        source=str(result_json),
        overall_score=overall,
        per_task_means=per_task_means,
        per_task_failed_checks=per_task_failed_checks,
    )

    output_md = Path(args.output) if args.output else result_json.parent / "diagnosis.md"
    output_md.write_text(md, encoding="utf-8")
    print(f"wrote markdown: {output_md}")

    output_json = (
        Path(args.output_json)
        if args.output_json
        else result_json.parent / "diagnosis.json"
    )
    dump_json(diags, output_json)
    print(f"wrote json:     {output_json}")

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m agent_loop.diagnostics")
    sub = p.add_subparsers(dest="command", required=True)

    a = sub.add_parser("analyze", help="Analyze a benchmark result.json + transcripts")
    a.add_argument("--result-json", required=True, help="Path to bench result.json")
    a.add_argument(
        "--transcripts-dirs",
        nargs="+",
        required=True,
        help="One or more dirs containing <task_id>.jsonl transcripts",
    )
    a.add_argument("--output", default=None, help="Markdown output path (default: <result.json dir>/diagnosis.md)")
    a.add_argument("--output-json", default=None, help="JSON output path (default: <result.json dir>/diagnosis.json)")
    a.set_defaults(func=cmd_analyze)
    return p


def main(argv: Optional[list[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)
