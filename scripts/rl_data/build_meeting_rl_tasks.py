#!/usr/bin/env python3
"""Build non-Val5 meeting_analysis RL task specs.

The output intentionally separates ``prompt`` from ``training_scaffold``. During
rollout we may inject the scaffold to elicit usable trajectories, but the
sanitized training sample should only keep the task prompt and model actions.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_ASSETS_DIR,
    DEFAULT_SPLIT_FILE,
    DEFAULT_TASKS_DIR,
    SCAFFOLD_BEGIN,
    SCAFFOLD_END,
    expected_output_files,
    stable_slug,
    workspace_sources,
    write_jsonl,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
from lib_tasks import TaskLoader, resolve_task_markdown_path  # noqa: E402


SYSTEM_PROMPT = (
    "You are an OpenClaw agent. Read the workspace files, use tools when needed, "
    "and write the requested deliverable to the required output file."
)


CAPABILITY_BY_HINT = [
    ("council", ["long_context_coverage", "targeted_evidence_extraction", "verification_and_self_correction"]),
    ("vote", ["vote_ledger_policy", "targeted_evidence_extraction", "verification_and_self_correction"]),
    ("budget", ["fact_ledger_policy", "targeted_evidence_extraction"]),
    ("stakeholder", ["entity_relation_ledger_policy", "long_context_coverage"]),
    ("attendee", ["entity_relation_ledger_policy", "long_context_coverage"]),
    ("action", ["action_item_policy", "verification_and_self_correction"]),
    ("tech", ["action_item_policy", "intermediate_ledger_and_structure"]),
    ("summary", ["long_context_coverage", "intermediate_ledger_and_structure"]),
]


VARIANT_TEMPLATES = [
    {
        "suffix": "evidence_ledger",
        "capabilities": ["targeted_evidence_extraction", "intermediate_ledger_and_structure"],
        "prompt_addition": (
            "\n\nAdditional requirement: before writing the final report, build an internal evidence ledger "
            "covering every major section of the transcript. The final file must include only verified facts, "
            "each with enough context to audit where it came from."
        ),
    },
    {
        "suffix": "pre_final_audit",
        "capabilities": ["verification_and_self_correction", "long_context_coverage"],
        "prompt_addition": (
            "\n\nAdditional requirement: do a pre-final consistency audit against the transcript. Check for "
            "missed late-meeting items, wrong entity bindings, wrong dates or amounts, and unsupported claims "
            "before producing the final file."
        ),
    },
]


TRAINING_SCAFFOLD = f"""{SCAFFOLD_BEGIN}
This scaffold is for training rollout only. Do not include it in supervised or RL
training prompts after trajectory collection.

Policy checklist:
1. Read the transcript in chunks or targeted passes until early, middle, and late
   sections have all been inspected.
2. Build a private ledger before final output: item, speaker/entity, evidence,
   required action/status, uncertainty.
3. For voting, budget, stakeholder, action-item, and decision tasks, verify every
   extracted row against the source before writing the deliverable.
4. If a tool call returns partial context, continue with targeted reads instead of
   guessing.
5. Write the requested output file only after the ledger is internally consistent.
{SCAFFOLD_END}"""


def infer_capabilities(task_id: str, prompt: str) -> list[str]:
    text = f"{task_id} {prompt}".lower()
    caps: list[str] = []
    for hint, names in CAPABILITY_BY_HINT:
        if re.search(rf"(?<![a-z0-9]){re.escape(hint)}(?![a-z0-9])", text):
            for name in names:
                if name not in caps:
                    caps.append(name)
    if not caps:
        caps = ["long_context_coverage", "targeted_evidence_extraction"]
    return caps


def task_to_record(task: Any, variant: str, prompt: str, index: int, extra_caps: list[str] | None = None) -> dict[str, Any]:
    caps = infer_capabilities(task.task_id, prompt)
    for cap in extra_caps or []:
        if cap not in caps:
            caps.append(cap)
    sources = workspace_sources(task.workspace_files)
    workspace_dests = {
        wf.get("dest") or wf.get("path")
        for wf in task.workspace_files
        if wf.get("dest") or wf.get("path")
    }
    output_files = [name for name in expected_output_files(prompt) if name not in workspace_dests]
    return {
        "id": f"{task.task_id}__{variant}",
        "source_task_id": task.task_id,
        "variant": variant,
        "data_source": "pinchbench/meeting_analysis/train_policy",
        "index": index,
        "prompt": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "base_prompt": task.prompt,
        "workspace_files": task.workspace_files,
        "transcript_sources": sources,
        "expected_output_files": output_files,
        "grading_criteria": task.grading_criteria,
        "target_capabilities": caps,
        "training_scaffold": TRAINING_SCAFFOLD,
        "scaffold_injection": {
            "mode": "rollout_only",
            "sanitize_before_training": True,
            "markers": [SCAFFOLD_BEGIN, SCAFFOLD_END],
        },
        "metadata": {
            "task_name": task.name,
            "grading_type": task.grading_type,
            "timeout_seconds": task.timeout_seconds,
            "grading_weights": task.grading_weights or {"automated": 0.5, "llm_judge": 0.5},
            "file_path": str(task.file_path) if task.file_path else None,
        },
    }


def build_records(args: argparse.Namespace) -> list[dict[str, Any]]:
    split = json.loads(Path(args.split_file).read_text(encoding="utf-8"))
    test_ids = set(split.get("test", []))
    train_ids = list(split.get("train", []))
    if args.include:
        train_ids = [tid for tid in train_ids if tid in set(args.include)]

    loader = TaskLoader(Path(args.tasks_dir))
    records: list[dict[str, Any]] = []
    idx = 0
    for task_id in train_ids:
        if task_id in test_ids:
            raise RuntimeError(f"Refusing to build RL data from Val5/test task: {task_id}")
        task_path = resolve_task_markdown_path(Path(args.tasks_dir), task_id)
        task = loader.load_task(task_path)
        records.append(task_to_record(task, "base", task.prompt, idx))
        idx += 1
        if args.variants_per_task:
            for tpl in VARIANT_TEMPLATES[: args.variants_per_task]:
                output_files = expected_output_files(task.prompt)
                prompt = task.prompt + tpl["prompt_addition"]
                if output_files:
                    stem = stable_slug(tpl["suffix"])
                    prompt += f"\n\nWrite the final answer to `{output_files[0]}` as requested; do not create a substitute filename."
                records.append(task_to_record(task, tpl["suffix"], prompt, idx, tpl["capabilities"]))
                idx += 1
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Build meeting_analysis RL task specs from non-Val5 train tasks.")
    parser.add_argument("--tasks-dir", default=str(DEFAULT_TASKS_DIR))
    parser.add_argument("--assets-dir", default=str(DEFAULT_ASSETS_DIR))
    parser.add_argument("--split-file", default=str(DEFAULT_SPLIT_FILE))
    parser.add_argument("--output", default=str(REPO_ROOT / "rl" / "data" / "meeting_rl_tasks.jsonl"))
    parser.add_argument("--variants-per-task", type=int, default=2, choices=[0, 1, 2])
    parser.add_argument("--include", action="append", default=[], help="Optional train task id filter. Repeatable.")
    args = parser.parse_args()

    records = build_records(args)
    count = write_jsonl(Path(args.output), records)
    print(f"Wrote {count} RL task specs to {args.output}")
    print(f"Train tasks used: {len({r['source_task_id'] for r in records})}; variants: {count}")


if __name__ == "__main__":
    main()
