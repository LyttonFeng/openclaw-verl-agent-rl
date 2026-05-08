#!/usr/bin/env python3
"""
Build veRL-compatible parquet files for meeting_analysis RL training.

Usage:
    python rl/train/build_meeting_analysis_prompts.py

Output:
    rl/data/meeting_analysis_train.parquet  (22 tasks)
    rl/data/meeting_analysis_val.parquet    (5 tasks)
"""

import json
import sys
from pathlib import Path

import pandas as pd

# Project paths
REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TASKS_DIR = REPO_ROOT / "tasks"
ASSETS_DIR = REPO_ROOT / "assets"
SPLIT_FILE = REPO_ROOT / "rl" / "train" / "meeting_analysis_split.json"
OUTPUT_DIR = REPO_ROOT / "rl" / "data"

# Add scripts to path for lib_tasks
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from lib_tasks import TaskLoader


SYSTEM_PROMPT = (
    "You are a helpful assistant. You have access to tools for reading and writing files. "
    "When given a task, read the relevant files in your workspace, analyze the content, "
    "and write your output to the specified file."
)


def load_split() -> dict:
    """Load train/test split definition."""
    with open(SPLIT_FILE) as f:
        return json.load(f)


def task_to_record(task, split: str, index: int) -> dict:
    """Convert a Task object to a veRL parquet record."""
    # Build prompt messages
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": task.prompt},
    ]

    # Workspace files info for agent loop to seed
    workspace_files = []
    for wf in task.workspace_files:
        source = wf.get("source", "")
        dest = wf.get("dest", wf.get("path", ""))
        if source:
            # source is relative to assets/
            workspace_files.append({"source": source, "dest": dest})
        elif "content" in wf:
            # inline content
            workspace_files.append({"content": wf["content"], "dest": dest})

    record = {
        "data_source": "pinchbench/meeting_analysis",
        "prompt": prompt,
        "reward_model": {"style": "rule", "ground_truth": task.task_id},
        "extra_info": {
            "split": split,
            "index": index,
            "task_id": task.task_id,
            "task_name": task.name,
            "grading_type": task.grading_type,
            "grading_weights": task.grading_weights or {"automated": 0.5, "llm_judge": 0.5},
            "timeout_seconds": task.timeout_seconds,
            "workspace_files": workspace_files,
        },
    }
    return record


def build_parquet(task_ids: list, split: str, loader: TaskLoader) -> pd.DataFrame:
    """Build a DataFrame from task IDs."""
    records = []
    for i, task_id in enumerate(task_ids):
        task_file = TASKS_DIR / f"{task_id}.md"
        if not task_file.exists():
            print(f"  WARNING: {task_file} not found, skipping")
            continue
        task = loader.load_task(task_file)
        record = task_to_record(task, split, i)
        records.append(record)
        print(f"  [{split}] {i:02d} {task_id} -> {len(task.prompt)} chars prompt")

    df = pd.DataFrame(records)
    return df


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    split = load_split()
    loader = TaskLoader(TASKS_DIR)

    print(f"Building meeting_analysis parquet files...")
    print(f"  Train tasks: {len(split['train'])}")
    print(f"  Val tasks: {len(split['test'])}")
    print()

    # Build train
    print("=== Train ===")
    train_df = build_parquet(split["train"], "train", loader)
    train_path = OUTPUT_DIR / "meeting_analysis_train.parquet"
    train_df.to_parquet(train_path, index=False)
    print(f"\n  Saved: {train_path} ({len(train_df)} rows)")

    # Build val
    print("\n=== Val ===")
    val_df = build_parquet(split["test"], "test", loader)
    val_path = OUTPUT_DIR / "meeting_analysis_val.parquet"
    val_df.to_parquet(val_path, index=False)
    print(f"\n  Saved: {val_path} ({len(val_df)} rows)")

    # Verify
    print("\n=== Verification ===")
    check_df = pd.read_parquet(train_path)
    sample = check_df.iloc[0]
    print(f"  Sample data_source: {sample['data_source']}")
    print(f"  Sample prompt[0]: {sample['prompt'][0]['role']}")
    print(f"  Sample prompt[1] content[:80]: {sample['prompt'][1]['content'][:80]}...")
    print(f"  Sample extra_info task_id: {sample['extra_info']['task_id']}")
    print(f"  Sample workspace_files: {len(sample['extra_info']['workspace_files'])} files")
    print("\nDone.")


if __name__ == "__main__":
    main()
