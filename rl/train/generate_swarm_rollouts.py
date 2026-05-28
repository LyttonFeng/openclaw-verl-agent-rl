#!/usr/bin/env python3
"""Generate template-diverse rollouts for swarm-policy GRPO training.

For each task, runs K rollouts where each uses a different prompt template
that biases the agent toward a different swarm/team policy. After the rollout
completes, DSv4-Pro judges the policy quality. The output record carries both
terminal and swarm_policy scores plus a composite reward.

This is a standalone wrapper around `generate_meeting_rollouts`'s helpers —
it does NOT modify codex's running pipeline files. Output goes to its own
swarm-policy directory so nothing else is touched.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("swarm_rollouts")

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "swarm_policy"))
sys.path.insert(0, str(REPO_ROOT / "agent_loop"))
sys.path.insert(0, str(REPO_ROOT / "rl" / "train"))
sys.path.insert(0, str(REPO_ROOT / "rewards"))

from lib_tasks import TaskLoader  # noqa: E402
from templates import SWARM_TEMPLATES, apply_template, extract_plan, extract_plan_from_transcript, get_template  # noqa: E402
from judge import judge_swarm_policy, composite_reward  # noqa: E402

# reuse run_single_rollout + grade_rollout from generate_meeting_rollouts.py
import generate_meeting_rollouts as gmr  # noqa: E402


def _save_jsonl_line(path: Path, record: dict, lock: threading.Lock):
    with lock:
        with path.open("a") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _extract_exec_stats(transcript: list[dict]) -> dict:
    """Extract observable behavior stats from transcript.

    Returns dict with: n_tool_calls, n_reads, coverage_hint, n_files_written,
    has_reread, has_intermediate_files.
    """
    n_tool = 0
    n_read = 0
    read_paths_offsets: list[tuple[str, int]] = []
    files_written: set[str] = set()
    for event in transcript:
        if event.get("type") != "message":
            continue
        msg = event.get("message", {})
        if msg.get("role") != "assistant":
            continue
        for item in msg.get("content", []):
            if not isinstance(item, dict):
                continue
            if item.get("type") != "toolCall":
                continue
            n_tool += 1
            args = item.get("arguments", {}) or {}
            name = str(item.get("name", "")).lower()
            if name in ("read",):
                n_read += 1
                path = str(args.get("path") or args.get("file") or "")
                off = args.get("offset", 0)
                try:
                    off = int(off) if off is not None else 0
                except (TypeError, ValueError):
                    off = 0
                read_paths_offsets.append((path, off))
            elif name in ("write", "edit"):
                path = str(args.get("path") or args.get("file") or "")
                if path:
                    files_written.add(path)

    offsets = [o for _, o in read_paths_offsets]
    coverage = f"L{min(offsets)}-{max(offsets)}" if offsets else ""

    # has_reread: any single file got read more than once with overlapping
    # offsets, OR same path read after a write happened.
    per_path_reads: dict[str, list[int]] = {}
    for p, o in read_paths_offsets:
        per_path_reads.setdefault(p, []).append(o)
    has_reread = any(len(v) > 1 for v in per_path_reads.values())

    # has_intermediate_files: more than one file written, OR files with names
    # suggesting notes/drafts (e.g., notes_, draft_, sub_, table_).
    has_intermediate = len(files_written) > 1 or any(
        any(tok in f.lower() for tok in ("notes", "draft", "sub_", "table", "evidence", "intermediate"))
        for f in files_written
    )

    return {
        "n_tool_calls": n_tool,
        "n_reads": n_read,
        "coverage_hint": coverage,
        "n_files_written": len(files_written),
        "has_reread": has_reread,
        "has_intermediate_files": has_intermediate,
        "files_written": sorted(files_written),
    }


def _process_one(
    *,
    task_id: str,
    task,
    template_id: str,
    resp_idx: int,
    worker_idx: int,
    assets_dir: Path,
    vllm_base_url: str,
    model: str,
    timeout: int,
    output_dir: Path,
    judge_kwargs: dict,
    write_lock: threading.Lock,
    graded_file: Path,
) -> dict:
    """Run one rollout with template applied; grade terminal; judge swarm policy."""
    # Mutate task.prompt to inject template (we work on a deep-copy of the loaded task)
    task_copy = copy.copy(task)  # shallow is fine — Task fields are scalars/dicts
    original_prompt = task_copy.prompt
    task_copy.prompt = apply_template(original_prompt, template_id)

    rollout = gmr.run_single_rollout(
        task=task_copy,
        assets_dir=assets_dir,
        vllm_base_url=vllm_base_url,
        model=model,
        timeout=timeout,
        worker_idx=worker_idx,
    )

    # Snapshot workspace
    snap_dir = output_dir / "workspaces" / f"{task_id}__{template_id}__r{resp_idx}"
    snap_dir.mkdir(parents=True, exist_ok=True)
    gmr._snapshot_workspace(rollout.get("workspace_path", ""), str(snap_dir))
    rollout["workspace_path"] = str(snap_dir)

    grading = gmr.grade_rollout(task_copy, rollout)
    terminal_score = grading.get("score", 0.0)

    # Save transcript
    ts_path = output_dir / "transcripts" / f"{task_id}__{template_id}__r{resp_idx}.jsonl"
    ts_path.parent.mkdir(parents=True, exist_ok=True)
    gmr._save_transcript(rollout.get("transcript", []), str(ts_path))

    # Extract plan ONLY from assistant message text (avoid matching the
    # template example that's embedded in the user prompt).
    plan_text = extract_plan_from_transcript(rollout.get("transcript", [])) or ""

    stats = _extract_exec_stats(rollout.get("transcript", []))
    template_obj = get_template(template_id)

    # Swarm policy judge — evaluates OBSERVED behavior, not just plan text
    judge_result = judge_swarm_policy(
        task_brief=original_prompt[:1500],
        plan_text=plan_text,
        terminal_score=terminal_score,
        template_id=template_id,
        template_desc=template_obj.description,
        n_tool_calls=stats["n_tool_calls"],
        n_reads=stats["n_reads"],
        coverage_hint=stats["coverage_hint"],
        n_files_written=stats["n_files_written"],
        has_reread=stats["has_reread"],
        has_intermediate_files=stats["has_intermediate_files"],
        **judge_kwargs,
    )
    swarm_score = judge_result.score

    composite = composite_reward(terminal_score, swarm_score, gamma=0.4)

    record = {
        "task_id": task_id,
        "template_id": template_id,
        "template_desc": get_template(template_id).description,
        "resp_idx": resp_idx,
        "prompt": task_copy.prompt,
        "policy_text": plan_text,
        "terminal_score": float(terminal_score),
        "swarm_policy_score": float(swarm_score),
        "swarm_breakdown": judge_result.breakdown,
        "swarm_notes": judge_result.notes,
        "swarm_judge_error": judge_result.error,
        "composite_reward": float(composite),
        "diagnostics": grading.get("diagnostics", {}),
        "automated_score": grading.get("automated_score", 0.0),
        "judge_score": grading.get("judge_score", 0.0),
        "n_tool_calls": stats["n_tool_calls"],
        "n_reads": stats["n_reads"],
        "coverage_hint": stats["coverage_hint"],
        "n_files_written": stats["n_files_written"],
        "files_written": stats["files_written"],
        "has_reread": stats["has_reread"],
        "has_intermediate_files": stats["has_intermediate_files"],
        "transcript_path": str(ts_path),
        "workspace_path": str(snap_dir),
        "execution_status": rollout.get("status", "unknown"),
        "execution_time": rollout.get("execution_time", 0),
    }
    _save_jsonl_line(graded_file, record, write_lock)
    logger.info(
        "DONE %s | %s | term=%.3f swarm=%.3f comp=%.3f plan=%s",
        task_id, template_id, terminal_score, swarm_score, composite,
        "✓" if plan_text else "MISSING"
    )
    return record


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tasks", nargs="+", required=True, help="Task IDs (one or more).")
    p.add_argument("--templates", nargs="+", default=None,
                   help="Template IDs to use. Default: all 4.")
    p.add_argument("--resp-per-template", type=int, default=1,
                   help="Rollouts per (task, template) pair. Default 1.")
    p.add_argument("--tasks-dir", required=True)
    p.add_argument("--assets-dir", required=True)
    p.add_argument("--vllm-base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument("--judge-model", default="deepseek-v4-pro")
    p.add_argument("--judge-base-url", default="https://api.deepseek.com/v1")
    p.add_argument("--timeout", type=int, default=600)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--terminal-judge-model", default="deepseek-chat",
                   help="Terminal grader model. Default deepseek-chat.")
    args = p.parse_args()

    os.environ["MEETING_JUDGE_MODEL"] = args.terminal_judge_model

    tasks_dir = Path(args.tasks_dir)
    assets_dir = Path(args.assets_dir)
    loader = TaskLoader(tasks_dir)

    template_ids = args.templates or [t.template_id for t in SWARM_TEMPLATES]
    logger.info("Templates: %s", template_ids)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    graded_file = output_dir / "graded_swarm_trajectories.jsonl"
    # Append mode: do not clobber existing data, but freshen header line.
    if not graded_file.exists():
        graded_file.touch()

    # Build job list
    jobs: list[tuple] = []
    for tid in args.tasks:
        tf = tasks_dir / f"{tid}.md"
        if not tf.exists():
            logger.warning("Task file missing: %s", tf)
            continue
        task = loader.load_task(tf)
        for template_id in template_ids:
            for resp_idx in range(args.resp_per_template):
                jobs.append((tid, task, template_id, resp_idx))
    logger.info("Total jobs: %d  (tasks=%d × templates=%d × resp=%d)",
                len(jobs), len(args.tasks), len(template_ids), args.resp_per_template)

    judge_kwargs = {
        "model": args.judge_model,
        "base_url": args.judge_base_url,
        "api_key": os.environ.get("DEEPSEEK_API_KEY"),
        "timeout": 120.0,
    }
    write_lock = threading.Lock()

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.num_workers) as ex:
        futures = []
        for i, (tid, task, template_id, resp_idx) in enumerate(jobs):
            worker_idx = i % args.num_workers
            futures.append(ex.submit(
                _process_one,
                task_id=tid, task=task, template_id=template_id,
                resp_idx=resp_idx, worker_idx=worker_idx,
                assets_dir=assets_dir, vllm_base_url=args.vllm_base_url,
                model=args.model, timeout=args.timeout, output_dir=output_dir,
                judge_kwargs=judge_kwargs, write_lock=write_lock,
                graded_file=graded_file,
            ))
        for f in as_completed(futures):
            try:
                f.result()
            except Exception as e:
                logger.error("Job failed: %s", e)
    elapsed = time.time() - t0
    logger.info("All %d jobs done in %.1fs. Graded file: %s", len(jobs), elapsed, graded_file)


if __name__ == "__main__":
    main()
