#!/usr/bin/env python3
"""
Build veRL parquet for *single-turn, transcript-inlined* meeting training.

Compared to build_meeting_analysis_prompts.py (which keeps the original prompt
and expects an agent loop to read files from a workspace), this script:

  - filters to tasks whose source transcript fits in MAX_PROMPT_CHARS
    (default: NTIA + GitLab transcripts; NASA/Tampa are too large)
  - inlines the transcript into the user message so a single-turn rollout has
    everything it needs
  - prepends instructions telling the model to emit the report body directly,
    matching the single-turn reward function (rewards/meeting_reward_single_turn).

Output:
  rl/data/meeting_inline_train.parquet
  rl/data/meeting_inline_val.parquet
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
TASKS_DIR = REPO_ROOT / "pinchbench_tasks" / "meeting_analysis"
ASSETS_DIR = REPO_ROOT / "assets"
SPLIT_FILE = REPO_ROOT / "rl" / "train" / "meeting_analysis_split.json"
OUTPUT_DIR = REPO_ROOT / "rl" / "data"

sys.path.insert(0, str(REPO_ROOT / "scripts"))
from lib_tasks import TaskLoader  # noqa: E402


SYSTEM_PROMPT = (
    "You are a careful analyst. Read the meeting transcript provided in the user "
    "message, then produce the requested report body. Output ONLY the contents of "
    "the target file — no preamble, no XML tags, no markdown fence around the "
    "whole output."
)


def _extract_output_filename(task) -> str:
    code = task.automated_checks or ""
    m = re.search(r'workspace\s*/\s*[\'"]([^\'"]+\.md)[\'"]', code)
    if m:
        return m.group(1)
    m = re.search(r'(?:called|named)\s+`([^`]+\.md)`', task.prompt or "")
    if m:
        return m.group(1)
    return "output.md"


def _load_transcript(task) -> tuple[str, int]:
    for wf in (task.workspace_files or []):
        src = wf.get("source", "")
        if "meetings/" in src or src.endswith(".md"):
            p = ASSETS_DIR / src if not src.startswith("assets/") else REPO_ROOT / src
            if p.exists():
                txt = p.read_text(encoding="utf-8")
                return txt, len(txt)
    return "", 0


def build_record(task, split: str, index: int, max_chars: int) -> dict | None:
    transcript, n_chars = _load_transcript(task)
    if not transcript:
        return None
    if n_chars > max_chars:
        return None

    output_name = _extract_output_filename(task)
    user_msg = (
        f"{task.prompt}\n\n"
        f"---\n\n"
        f"Below is the full transcript for context. After reading, write the body "
        f"of `{output_name}` directly as your response (do not echo this prompt, "
        f"do not add tags, do not narrate).\n\n"
        f"```\n{transcript}\n```"
    )
    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]
    return {
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
            "transcript_chars": n_chars,
            "output_filename": output_name,
        },
    }


def build_split(task_ids, split, loader, max_chars):
    rows = []
    skipped = []
    for i, tid in enumerate(task_ids):
        task_file = TASKS_DIR / f"{tid}.md"
        if not task_file.exists():
            skipped.append((tid, "task file missing"))
            continue
        task = loader.load_task(task_file)
        rec = build_record(task, split, i, max_chars)
        if rec is None:
            t, n = _load_transcript(task)
            skipped.append((tid, f"transcript {n} chars > {max_chars}" if n else "no transcript"))
            continue
        rows.append(rec)
        print(f"  [{split}] {tid}  -> {rec['extra_info']['transcript_chars']} chars, output={rec['extra_info']['output_filename']}")
    if skipped:
        for tid, why in skipped:
            print(f"  [skip] {tid}  ({why})")
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-chars", type=int, default=80000,
                    help="skip tasks whose transcript exceeds this size (chars)")
    ap.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    split = json.loads(SPLIT_FILE.read_text())
    loader = TaskLoader(TASKS_DIR)

    print(f"max_chars={args.max_chars}  out={args.out_dir}")
    print("\n=== train ===")
    train_df = build_split(split["train"], "train", loader, args.max_chars)
    train_path = args.out_dir / "meeting_inline_train.parquet"
    train_df.to_parquet(train_path, index=False)
    print(f"  wrote {train_path} ({len(train_df)} rows)")

    print("\n=== val ===")
    val_df = build_split(split["test"], "test", loader, args.max_chars)
    val_path = args.out_dir / "meeting_inline_val.parquet"
    val_df.to_parquet(val_path, index=False)
    print(f"  wrote {val_path} ({len(val_df)} rows)")


if __name__ == "__main__":
    main()
