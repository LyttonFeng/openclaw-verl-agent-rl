#!/usr/bin/env python3
"""Remove rollout-only scaffold text from collected OpenClaw transcripts."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

from common import SCAFFOLD_BEGIN, SCAFFOLD_END


SCAFFOLD_RE = re.compile(
    rf"{re.escape(SCAFFOLD_BEGIN)}.*?{re.escape(SCAFFOLD_END)}",
    flags=re.DOTALL,
)


def scrub_value(value: Any) -> Any:
    if isinstance(value, str):
        return SCAFFOLD_RE.sub("", value).strip()
    if isinstance(value, list):
        return [scrub_value(v) for v in value]
    if isinstance(value, dict):
        return {k: scrub_value(v) for k, v in value.items()}
    return value


def should_drop(entry: dict[str, Any]) -> bool:
    text = json.dumps(entry, ensure_ascii=False)
    if SCAFFOLD_BEGIN not in text:
        return False
    role = str(entry.get("role") or entry.get("type") or "").lower()
    # Drop pure scaffold system/user events. Assistant/tool events are scrubbed
    # rather than dropped so model actions are preserved.
    return role in {"system", "user", "instruction"} or "assistant" not in role


def sanitize_entries(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean = []
    for entry in entries:
        if should_drop(entry):
            continue
        scrubbed = scrub_value(entry)
        if isinstance(scrubbed, dict):
            clean.append(scrubbed)
    return clean


def read_transcript(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    stripped = text.strip()
    if not stripped:
        return []
    if stripped.startswith("["):
        return json.loads(stripped)
    rows = []
    for line in stripped.splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def write_transcript(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Sanitize rollout-only scaffold from a transcript JSON/JSONL.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    rows = read_transcript(Path(args.input))
    clean = sanitize_entries(rows)
    write_transcript(Path(args.output), clean)
    print(f"Wrote sanitized transcript: {args.output} ({len(rows)} -> {len(clean)} events)")


if __name__ == "__main__":
    main()

