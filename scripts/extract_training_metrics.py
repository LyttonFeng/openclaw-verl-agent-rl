#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path


ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
STEP_RE = re.compile(r"step:(\d+) - (.*)")

FIELDS = [
    "critic/score/mean",
    "critic/rewards/mean",
    "critic/advantages/mean",
    "critic/advantages/max",
    "critic/advantages/min",
    "critic/returns/mean",
    "actor/pg_loss",
    "actor/kl_loss",
    "actor/grad_norm",
    "actor/entropy",
    "response_length/mean",
    "response_length/clip_ratio",
    "prompt_length/mean",
    "prompt_length/clip_ratio",
    "num_turns/mean",
    "timing_s/agent_loop/tool_calls/mean",
    "val-core/pinchbench/reward/mean@1",
    "val-aux/num_turns/mean",
]


def parse_value(rest: str, key: str) -> str:
    match = re.search(re.escape(key) + r":([-+0-9.eE]+|nan)", rest)
    return match.group(1) if match else ""


def extract(log_path: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    if not log_path.exists():
        return rows
    for raw in log_path.read_text(errors="replace").splitlines():
        line = ANSI_RE.sub("", raw)
        match = STEP_RE.search(line)
        if not match:
            continue
        rest = match.group(2)
        row = {"step": match.group(1)}
        for field in FIELDS:
            row[field] = parse_value(rest, field)
        rows.append(row)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("log_path", type=Path)
    parser.add_argument("--output", "-o", type=Path, required=True)
    args = parser.parse_args()

    rows = extract(args.log_path)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["step", *FIELDS])
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
