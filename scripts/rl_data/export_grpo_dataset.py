#!/usr/bin/env python3
"""Export meeting_analysis task specs plus consensus gold into GRPO-ready JSONL."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from common import REPO_ROOT, read_jsonl, write_jsonl


def main() -> None:
    parser = argparse.ArgumentParser(description="Export RL task specs to GRPO JSONL.")
    parser.add_argument("--tasks", default=str(REPO_ROOT / "rl" / "data" / "meeting_rl_tasks.jsonl"))
    parser.add_argument("--gold", default=str(REPO_ROOT / "rl" / "data" / "gold_ledgers" / "consensus_gold.jsonl"))
    parser.add_argument("--output", default=str(REPO_ROOT / "rl" / "data" / "meeting_grpo_dataset.jsonl"))
    parser.add_argument("--require-gold", action="store_true")
    args = parser.parse_args()

    tasks = read_jsonl(Path(args.tasks))
    gold_by_id: dict[str, dict[str, Any]] = {}
    gold_path = Path(args.gold)
    if gold_path.exists():
        gold_by_id = {row["task_id"]: row for row in read_jsonl(gold_path)}

    rows = []
    for task in tasks:
        gold = gold_by_id.get(task["id"])
        if args.require_gold and not gold:
            continue
        rows.append(
            {
                "data_source": task["data_source"],
                "prompt": task["prompt"],
                "reward_model": {
                    "style": "meeting_consensus_gold",
                    "ground_truth": task["id"],
                    "gold": gold or {},
                },
                "extra_info": {
                    "task_id": task["id"],
                    "source_task_id": task["source_task_id"],
                    "variant": task["variant"],
                    "target_capabilities": task.get("target_capabilities", []),
                    "workspace_files": task.get("workspace_files", []),
                    "expected_output_files": task.get("expected_output_files", []),
                    "training_scaffold": task.get("training_scaffold", ""),
                    "scaffold_injection": task.get("scaffold_injection", {}),
                    "metadata": task.get("metadata", {}),
                },
            }
        )
    count = write_jsonl(Path(args.output), rows)
    print(f"Wrote GRPO dataset: {args.output} ({count} rows)")


if __name__ == "__main__":
    main()

