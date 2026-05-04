#!/usr/bin/env python3
"""Analyze task16 rollout transcripts with the official PinchBench grader."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any


TASK_ID = "task_16_email_triage"


def _load_env_file(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text("utf-8", errors="replace").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and key not in os.environ:
            os.environ[key] = value


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                out.append(obj)
    return out


def _workspace_from_transcript(transcript: list[dict[str, Any]]) -> str:
    for entry in transcript:
        if entry.get("type") == "session":
            cwd = entry.get("cwd")
            if isinstance(cwd, str):
                return cwd
    return ""


def _tool_events(transcript: list[dict[str, Any]]) -> tuple[list[str], list[str]]:
    reads: list[str] = []
    writes: list[str] = []
    for entry in transcript:
        msg = entry.get("message")
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        if role == "assistant":
            for part in msg.get("content") or []:
                if not isinstance(part, dict) or part.get("type") != "toolCall":
                    continue
                args = part.get("arguments")
                if not isinstance(args, dict):
                    args = {}
                path = str(args.get("path", ""))
                if part.get("name") == "read":
                    reads.append(path)
                elif part.get("name") == "write":
                    writes.append(path)
        elif role == "toolResult" and msg.get("toolName") == "write":
            writes.append("toolResult")
    return reads, writes


def _bad_read_paths(paths: list[str]) -> list[str]:
    bad: list[str] = []
    for path in paths:
        p = path.strip()
        if not p:
            continue
        if p.startswith("/"):
            bad.append(path)
        elif "inboxes/" in p or "inabox/" in p or "/ inbox/" in p:
            bad.append(path)
        elif p.endswith("triage_report.md"):
            bad.append(path)
    return bad


def _bad_report_paths(paths: list[str]) -> list[str]:
    bad: list[str] = []
    for path in paths:
        if path in {"toolResult", "triage_report.md"}:
            continue
        if path:
            bad.append(path)
    return bad


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("transcripts_dir", type=Path)
    ap.add_argument("--repo", type=Path, default=Path.cwd())
    ap.add_argument("--env-file", type=Path, default=Path.home() / ".pinchbench_env")
    ap.add_argument("--task-id", default=TASK_ID)
    args = ap.parse_args()

    _load_env_file(args.env_file)
    repo = args.repo.resolve()
    sys.path.insert(0, str(repo / "scripts"))

    from lib_grading import (  # type: ignore
        grade_task,
        preflight_judge_connection,
        resolve_judge_backend_from_env,
    )
    from lib_tasks import TaskLoader, resolve_task_markdown_path  # type: ignore

    task_file = resolve_task_markdown_path(repo / "tasks", args.task_id)
    task = TaskLoader(repo / "tasks").load_task(task_file)
    judge_cfg = resolve_judge_backend_from_env(
        default_backend="api",
        default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
    )
    judge_status = "not_called"
    if judge_cfg["judge_backend"] == "api" and judge_cfg["judge_api_key"]:
        preflight_judge_connection(
            judge_model=str(judge_cfg["judge_model"]),
            judge_backend=str(judge_cfg["judge_backend"]),
            judge_base_url=str(judge_cfg["judge_base_url"]),
            judge_api_key=str(judge_cfg["judge_api_key"]),
            timeout_seconds=30.0,
        )
        judge_status = "ok_preflight"

    paths = sorted(args.transcripts_dir.glob(f"{args.task_id}__*.jsonl"))
    if not paths:
        paths = sorted(args.transcripts_dir.glob("*.jsonl"))

    rows: list[dict[str, Any]] = []
    for path in paths:
        transcript = _load_jsonl(path)
        workspace = _workspace_from_transcript(transcript)
        reads, writes = _tool_events(transcript)
        report_exists = bool(workspace and (Path(workspace) / "triage_report.md").is_file())
        result = grade_task(
            task=task,
            execution_result={
                "transcript": transcript,
                "workspace": workspace,
                "status": "success",
            },
            skill_dir=repo,
            judge_model=str(judge_cfg["judge_model"]),
            judge_backend=str(judge_cfg["judge_backend"]),
            judge_base_url=judge_cfg["judge_base_url"],
            judge_api_key=judge_cfg["judge_api_key"],
        )
        breakdown = result.breakdown or {}
        rows.append(
            {
                "file": path.name,
                "workspace": workspace,
                "score": float(result.score),
                "report_exists": report_exists,
                "judge_skipped": bool(breakdown.get("llm_judge.skipped_missing_required_report")),
                "bad_read_path": _bad_read_paths(reads),
                "bad_report_path": _bad_report_paths(writes),
                "notes": result.notes,
            }
        )

    scores = [r["score"] for r in rows]
    print(
        json.dumps(
            {
                "task_id": args.task_id,
                "judge_status.grade_call": judge_status,
                "judge_backend": judge_cfg["judge_backend"],
                "judge_model": judge_cfg["judge_model"],
                "n": len(rows),
                "mean_score": sum(scores) / len(scores) if scores else 0.0,
                "min_score": min(scores) if scores else 0.0,
                "max_score": max(scores) if scores else 0.0,
                "score_ge_0.6": sum(1 for s in scores if s >= 0.6),
                "report_exists_rate": (
                    sum(1 for r in rows if r["report_exists"]) / len(rows) if rows else 0.0
                ),
                "bad_read_path_count": sum(len(r["bad_read_path"]) for r in rows),
                "bad_report_path_count": sum(len(r["bad_report_path"]) for r in rows),
                "skipped_judge_count": sum(1 for r in rows if r["judge_skipped"]),
                "episodes": rows,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
