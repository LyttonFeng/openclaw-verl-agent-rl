"""
Prepare PinchBench task prompts as veRL parquet dataset for Online RL.

Regenerate parquet when ``pinchbench_tasks/*.md`` prompts change so
train/benchmark prompts stay aligned.

For Online RL, we only need task prompts (not pre-collected trajectories).
veRL's agent loop will generate new trajectories using the current policy.

Usage:
    python rl/train/prepare_prompts.py \
        --tasks-dir pinchbench_tasks/ \
        --output-dir rl/data/prompts/ \
        --task-ids task_16_email_triage
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import yaml

_REPO_SCRIPTS = Path(__file__).resolve().parent.parent.parent / "scripts"
if str(_REPO_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_REPO_SCRIPTS))
from lib_tasks import resolve_task_markdown_path  # noqa: E402


DEFAULT_TASK_IDS = [
    "task_16_email_triage",
]


def parse_task_file(task_path: Path) -> dict:
    """Parse a PinchBench task markdown file."""
    text = task_path.read_text("utf-8")

    frontmatter = {}
    if text.startswith("---"):
        parts = text.split("---", 2)
        if len(parts) >= 3:
            frontmatter = yaml.safe_load(parts[1]) or {}
            text = parts[2]

    prompt = ""
    in_prompt = False
    prompt_lines = []
    for line in text.split("\n"):
        if line.strip() == "## Prompt":
            in_prompt = True
            continue
        if in_prompt and line.strip().startswith("## "):
            break
        if in_prompt:
            prompt_lines.append(line)

    prompt = "\n".join(prompt_lines).strip()

    return {
        "task_id": frontmatter.get("id", task_path.stem),
        "name": frontmatter.get("name", ""),
        "category": frontmatter.get("category", ""),
        "grading_type": frontmatter.get("grading_type", "automated"),
        "timeout_seconds": frontmatter.get("timeout_seconds", 300),
        "prompt": prompt,
        "workspace_files": frontmatter.get("workspace_files", []),
    }


def build_verl_row(task: dict, repeat_idx: int = 0) -> dict:
    """Build one veRL parquet row from a task prompt."""
    prompt_messages = [
        {"role": "user", "content": task["prompt"]},
    ]

    return {
        "data_source": "pinchbench",
        "prompt": prompt_messages,
        "ability": "tool_use",
        "reward_model": {
            "style": "rule",
            "ground_truth": task["task_id"],
        },
        "extra_info": {
            "task_id": task["task_id"],
            "task_name": task["name"],
            "category": task["category"],
            "grading_type": task["grading_type"],
            "timeout_seconds": task["timeout_seconds"],
            "repeat_idx": repeat_idx,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Prepare PinchBench task prompts for veRL Online RL"
    )
    parser.add_argument(
        "--tasks-dir", type=Path, default=Path("tasks"),
        help="Directory containing task_*.md files",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("rl/data/prompts"),
        help="Output directory for parquet files",
    )
    parser.add_argument(
        "--task-ids", nargs="+", default=DEFAULT_TASK_IDS,
        help="Task IDs to include (default: 8 selected tasks)",
    )
    parser.add_argument(
        "--repeats", type=int, default=1,
        help="How many times to repeat each task in the dataset",
    )
    parser.add_argument(
        "--round-robin", action="store_true", default=False,
        help=(
            "Interleave tasks in round-robin order: [t0,t1,...,t7] x repeats. "
            "Ensures each consecutive batch_size window contains diverse tasks "
            "instead of clustering the same task together. "
            "Combine with data.shuffle=False in veRL for strict round-robin."
        ),
    )
    args = parser.parse_args()

    tasks_dir = args.tasks_dir
    if not tasks_dir.exists():
        print(f"Tasks directory not found: {tasks_dir}")
        sys.exit(1)

    tasks = []
    for task_id in args.task_ids:
        task_path = resolve_task_markdown_path(tasks_dir, task_id)
        if not task_path.exists():
            print(f"Warning: task file not found: {task_path}")
            continue
        task = parse_task_file(task_path)
        tasks.append(task)
        canon = task.get("task_id", task_id)
        print(f"  Loaded {task_id} -> {canon}: {task['name']} ({task['category']}) [{task_path.name}]")

    if not tasks:
        print("No valid tasks found")
        sys.exit(1)

    rows = []
    if args.round_robin:
        # Interleave: [t0,t1,...,tN] x repeats so consecutive batch windows are diverse.
        for i in range(args.repeats):
            for task in tasks:
                rows.append(build_verl_row(task, repeat_idx=i))
    else:
        for task in tasks:
            for i in range(args.repeats):
                rows.append(build_verl_row(task, repeat_idx=i))

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # For Online RL, all data is "train" (prompts to sample from)
    train_path = args.output_dir / "train.parquet"
    pd.DataFrame(rows).to_parquet(train_path, index=False)
    print(f"\nWrote {len(rows)} prompts to {train_path}")

    # Also write a small val set (same tasks, for eval)
    val_path = args.output_dir / "val.parquet"
    val_rows = [build_verl_row(task, repeat_idx=999) for task in tasks]
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)
    print(f"Wrote {len(val_rows)} val prompts to {val_path}")

    # Summary
    print(f"\n{'='*50}")
    print(f"Tasks: {len(tasks)}")
    print(f"Repeats per task: {args.repeats}")
    print(f"Total train prompts: {len(rows)}")
    for task in tasks:
        print(f"  {task['task_id']}: {task['prompt'][:80]}...")


if __name__ == "__main__":
    main()
