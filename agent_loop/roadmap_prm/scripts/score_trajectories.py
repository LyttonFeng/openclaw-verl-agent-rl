"""
Attach Roadmap PRM per-turn scores to a graded_trajectories.jsonl.

Reads each record, parses its OpenClaw transcript, runs DSv4-flash judge against
the matching roadmap, and writes a new JSONL with extra fields:
    prm_turn_scores:  list[int]  ∈ {-1, 0, +1}, length = num assistant turns
    prm_milestones:   list[str]  milestone id (or "none") per turn
    prm_pitfalls:     list[str]  pitfall id (or "none") per turn
    prm_reasons:      list[str]  one-sentence reason per turn
    prm_n_turns:      int
    prm_pos:          int        count of +1 turns
    prm_neg:          int        count of -1 turns
    prm_zero:         int        count of  0 turns

Tasks without a roadmap (e.g. test-set tasks) are passed through unchanged
with prm_turn_scores=[] (training will fall back to terminal-only).

Usage:
    DEEPSEEK_API_KEY=sk-... \
    PYTHONPATH=. python3 -m agent_loop.roadmap_prm.scripts.score_trajectories \
        --graded-file /workspace/meeting_grpo_prm_v1/round_1/rollouts/graded_trajectories.jsonl \
        --tasks-dir tasks \
        --roadmaps-dir agent_loop/roadmap_prm/roadmaps \
        --output-suffix _prm \
        --max-workers 4

Idempotency: skips records that already have prm_turn_scores (re-run safe).
Use --force to rescore everything.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from agent_loop.roadmap_prm.judge import judge_terminal_completion, judge_trajectory
from agent_loop.roadmap_prm.schema import Roadmap
from agent_loop.roadmap_prm.trajectory import parse_openclaw_jsonl


def _extract_prompt_from_task_md(task_md_path: Path) -> str:
    """Same regex used by batch_calibration.py — keeps prompt extraction
    consistent across calibration and scoring."""
    text = task_md_path.read_text()
    m = re.search(r"##\s*Prompt\s*\n(.+?)(?:\n---|\n##\s)", text, re.DOTALL)
    if not m:
        raise ValueError(f"Could not find ## Prompt section in {task_md_path}")
    return m.group(1).strip()


def _score_one_record(
    record: dict,
    roadmaps_dir: Path,
    tasks_dir: Path,
    prompt_cache: dict[str, str],
    roadmap_cache: dict[str, Roadmap | None],
    max_turns: int | None,
) -> dict:
    """Score one record. Returns the record with PRM fields added (or empty
    PRM fields if no roadmap / no transcript / judge error)."""
    task_id = record.get("task_id", "")
    transcript_path = record.get("transcript_path", "")
    out = dict(record)  # shallow copy

    # No roadmap → empty PRM (terminal-only fallback at training time)
    if task_id not in roadmap_cache:
        rmp = roadmaps_dir / f"{task_id}.yaml"
        if rmp.exists():
            try:
                roadmap_cache[task_id] = Roadmap.from_yaml(str(rmp))
            except Exception as e:
                print(f"  WARN  {task_id}: roadmap parse failed: {e}", file=sys.stderr)
                roadmap_cache[task_id] = None
        else:
            roadmap_cache[task_id] = None
    roadmap = roadmap_cache[task_id]
    if roadmap is None:
        out.update(prm_turn_scores=[], prm_milestones=[], prm_pitfalls=[],
                   prm_reasons=[], prm_n_turns=0, prm_pos=0, prm_neg=0,
                   prm_zero=0, prm_status="no_roadmap")
        return out

    # Resolve prompt
    if task_id not in prompt_cache:
        task_md = tasks_dir / f"{task_id}.md"
        if task_md.exists():
            try:
                prompt_cache[task_id] = _extract_prompt_from_task_md(task_md)
            except Exception as e:
                print(f"  WARN  {task_id}: task md prompt extract failed: {e}", file=sys.stderr)
                prompt_cache[task_id] = record.get("prompt", "") or ""
        else:
            prompt_cache[task_id] = record.get("prompt", "") or ""
    prompt = prompt_cache[task_id]

    # Parse transcript
    if not transcript_path or not Path(transcript_path).exists():
        out.update(prm_turn_scores=[], prm_milestones=[], prm_pitfalls=[],
                   prm_reasons=[], prm_n_turns=0, prm_pos=0, prm_neg=0,
                   prm_zero=0, prm_status="no_transcript")
        return out
    try:
        turns = parse_openclaw_jsonl(transcript_path)
    except Exception as e:
        out.update(prm_turn_scores=[], prm_milestones=[], prm_pitfalls=[],
                   prm_reasons=[], prm_n_turns=0, prm_pos=0, prm_neg=0,
                   prm_zero=0, prm_status=f"parse_error:{type(e).__name__}")
        return out

    if not turns:
        out.update(prm_turn_scores=[], prm_milestones=[], prm_pitfalls=[],
                   prm_reasons=[], prm_n_turns=0, prm_pos=0, prm_neg=0,
                   prm_zero=0, prm_status="empty_trajectory")
        return out

    # Terminal-completion gate (design pivot 2026-05-06):
    # Ask the judge directly "is the agent's terminal goal mostly accomplished?"
    # If yes → suppress per-turn PRM (all zeros). Only "lost" trajectories get
    # process supervision. Judge decides the binary, no hardcoded threshold —
    # but the terminal grader's score is passed in as one signal for the judge
    # to sanity-check against.
    n_turns_total = len(turns) if max_turns is None else min(max_turns, len(turns))
    final_text = turns[n_turns_total - 1].to_judge_text() if n_turns_total > 0 else ""
    terminal_score = record.get("score")
    try:
        terminal_score = float(terminal_score) if terminal_score is not None else None
    except (TypeError, ValueError):
        terminal_score = None
    mostly_done, gate_reason = judge_terminal_completion(prompt, final_text, terminal_score)
    if mostly_done:
        out.update(
            prm_turn_scores=[0] * n_turns_total,
            prm_milestones=["none"] * n_turns_total,
            prm_pitfalls=["none"] * n_turns_total,
            prm_reasons=[gate_reason] * n_turns_total,
            prm_n_turns=n_turns_total,
            prm_pos=0, prm_neg=0, prm_zero=n_turns_total,
            prm_status="terminal_satisfied",
            prm_gate_reason=gate_reason,
        )
        return out

    # Run judge over all turns
    try:
        results = judge_trajectory(roadmap, prompt, turns, max_turns=max_turns)
    except Exception as e:
        out.update(prm_turn_scores=[], prm_milestones=[], prm_pitfalls=[],
                   prm_reasons=[], prm_n_turns=0, prm_pos=0, prm_neg=0,
                   prm_zero=0, prm_status=f"judge_error:{type(e).__name__}:{str(e)[:80]}")
        return out

    scores = [r.score for r in results]
    out.update(
        prm_turn_scores=scores,
        prm_milestones=[r.milestone for r in results],
        prm_pitfalls=[r.pitfall for r in results],
        prm_reasons=[r.reason for r in results],
        prm_n_turns=len(scores),
        prm_pos=sum(1 for s in scores if s > 0),
        prm_neg=sum(1 for s in scores if s < 0),
        prm_zero=sum(1 for s in scores if s == 0),
        prm_status="ok",
    )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graded-file", required=True,
                    help="graded_trajectories.jsonl from rollout pipeline")
    ap.add_argument("--tasks-dir", default="tasks")
    ap.add_argument("--roadmaps-dir", default="agent_loop/roadmap_prm/roadmaps")
    ap.add_argument("--output-suffix", default="_prm",
                    help="Output file = <input>{suffix}.jsonl. Default _prm.")
    ap.add_argument("--max-turns-per-trajectory", type=int, default=None,
                    help="Optional cap to limit judge API cost on long trajectories.")
    ap.add_argument("--max-workers", type=int, default=4,
                    help="Concurrent judge calls. DeepSeek API rate-limit friendly.")
    ap.add_argument("--force", action="store_true",
                    help="Rescore even records that already have prm_turn_scores.")
    args = ap.parse_args()

    in_path = Path(args.graded_file)
    if not in_path.exists():
        print(f"ERROR: graded file not found: {in_path}", file=sys.stderr)
        return 1
    out_path = in_path.parent / f"{in_path.stem}{args.output_suffix}.jsonl"

    records = []
    with open(in_path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records)} records from {in_path}")
    print(f"Output: {out_path}")

    # Filter records that already have PRM scores (unless --force)
    todo_idx = []
    for i, r in enumerate(records):
        if not args.force and "prm_turn_scores" in r and r.get("prm_status") == "ok":
            continue
        todo_idx.append(i)
    print(f"To score: {len(todo_idx)} ({len(records) - len(todo_idx)} already done, "
          f"{'force re-score' if args.force else 'skipped'})")

    roadmaps_dir = Path(args.roadmaps_dir)
    tasks_dir = Path(args.tasks_dir)
    prompt_cache: dict[str, str] = {}
    roadmap_cache: dict[str, Roadmap | None] = {}

    # Pre-warm caches single-threaded so concurrent workers don't all parse
    # the same yaml/md files redundantly.
    unique_tasks = {records[i]["task_id"] for i in todo_idx if records[i].get("task_id")}
    for tid in unique_tasks:
        rmp = roadmaps_dir / f"{tid}.yaml"
        if rmp.exists():
            try:
                roadmap_cache[tid] = Roadmap.from_yaml(str(rmp))
            except Exception as e:
                print(f"  WARN  {tid}: roadmap parse failed: {e}", file=sys.stderr)
                roadmap_cache[tid] = None
        else:
            roadmap_cache[tid] = None
        task_md = tasks_dir / f"{tid}.md"
        if task_md.exists():
            try:
                prompt_cache[tid] = _extract_prompt_from_task_md(task_md)
            except Exception as e:
                print(f"  WARN  {tid}: prompt extract failed: {e}", file=sys.stderr)

    n_with_roadmap = sum(1 for v in roadmap_cache.values() if v is not None)
    n_no_roadmap = sum(1 for v in roadmap_cache.values() if v is None)
    print(f"Roadmaps: {n_with_roadmap} loaded, {n_no_roadmap} missing "
          f"(missing → terminal-only fallback)")

    # Score concurrently
    t0 = time.time()
    scored: dict[int, dict] = {}
    n_done = 0
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futures = {
            ex.submit(_score_one_record, records[i], roadmaps_dir, tasks_dir,
                      prompt_cache, roadmap_cache, args.max_turns_per_trajectory): i
            for i in todo_idx
        }
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                scored[i] = fut.result()
            except Exception as e:
                print(f"  ERROR record {i} ({records[i].get('task_id', '?')}): "
                      f"{type(e).__name__}: {e}", file=sys.stderr)
                # Keep the original record untouched on hard failure
                rec = dict(records[i])
                rec.update(prm_turn_scores=[], prm_status=f"exec_error:{type(e).__name__}")
                scored[i] = rec
            n_done += 1
            if n_done % 5 == 0 or n_done == len(todo_idx):
                elapsed = time.time() - t0
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (len(todo_idx) - n_done) / rate if rate > 0 else 0
                print(f"  [{n_done}/{len(todo_idx)}] {elapsed:.0f}s elapsed, "
                      f"ETA {eta:.0f}s")

    # Write merged output
    out_records = []
    status_counts: dict[str, int] = {}
    for i, orig in enumerate(records):
        rec = scored.get(i, orig)
        out_records.append(rec)
        st = rec.get("prm_status", "passthrough")
        status_counts[st] = status_counts.get(st, 0) + 1

    with open(out_path, "w") as f:
        for r in out_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    # Summary
    print(f"\nWrote {len(out_records)} records → {out_path}")
    print(f"PRM status breakdown:")
    for st, c in sorted(status_counts.items(), key=lambda x: -x[1]):
        print(f"  {st}: {c}")

    # Per-task aggregate
    by_task: dict[str, list[dict]] = {}
    for r in out_records:
        by_task.setdefault(r.get("task_id", "?"), []).append(r)
    print(f"\nPer-task PRM averages (ok-status only):")
    for tid in sorted(by_task):
        ok = [r for r in by_task[tid] if r.get("prm_status") == "ok"]
        if not ok:
            continue
        all_scores = [s for r in ok for s in r["prm_turn_scores"]]
        if not all_scores:
            continue
        mean = sum(all_scores) / len(all_scores)
        n_pos = sum(1 for s in all_scores if s > 0)
        n_neg = sum(1 for s in all_scores if s < 0)
        n_zero = sum(1 for s in all_scores if s == 0)
        print(f"  {tid}: n_resp={len(ok)}, n_turns_total={len(all_scores)}, "
              f"mean={mean:+.2f}, +{n_pos}/0:{n_zero}/-{n_neg}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
