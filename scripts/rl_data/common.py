#!/usr/bin/env python3
"""Shared helpers for meeting_analysis RL data construction."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TASKS_DIR = REPO_ROOT / "pinchbench_tasks" / "meeting_analysis"
DEFAULT_ASSETS_DIR = REPO_ROOT / "assets"
DEFAULT_SPLIT_FILE = REPO_ROOT / "rl" / "train" / "meeting_analysis_split.json"

SCAFFOLD_BEGIN = "TRAINING_SCAFFOLD_START"
SCAFFOLD_END = "TRAINING_SCAFFOLD_END"


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def expected_output_files(prompt: str) -> list[str]:
    files = []
    for match in re.finditer(r"`([^`]+?\.(?:md|txt|json|csv))`", prompt):
        name = match.group(1).strip()
        if name and name not in files:
            files.append(name)
    return files


def workspace_sources(workspace_files: list[dict[str, Any]]) -> list[str]:
    sources = []
    for wf in workspace_files:
        source = wf.get("source")
        if source and source not in sources:
            sources.append(source)
    return sources


def stable_slug(text: str) -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", text.lower()).strip("_")
    return re.sub(r"_+", "_", slug)[:80] or "item"


def normalize_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def token_set(text: str) -> set[str]:
    stop = {
        "the", "and", "for", "with", "that", "this", "from", "into", "about",
        "include", "report", "meeting", "transcript", "please", "should",
    }
    return {t for t in normalize_text(text).split() if len(t) > 2 and t not in stop}


def jaccard(a: str, b: str) -> float:
    ta = token_set(a)
    tb = token_set(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)

