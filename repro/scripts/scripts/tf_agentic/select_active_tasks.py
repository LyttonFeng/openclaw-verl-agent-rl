#!/usr/bin/env python3
"""Adaptive task selection for rollout generation — landmine-PROOF skip.

Reads a per-task health record and decides which tasks to ATTEMPT this round vs
temporarily SKIP. A task is skipped ONLY when it has been all-dead (every rollout
timeout/empty) for >= DEAD_THRESHOLD consecutive ATTEMPTED rounds. Skipped tasks
are FORCE re-probed every REPROBE_EVERY rounds, so nothing is ever permanently
excluded — if a future fix (bigger context, etc.) makes them viable, the re-probe
detects it automatically and they re-enter training. Every decision is logged loudly.

NOT a static hand-maintained list (that would be a silent-permanent-exclusion landmine).
"""
import json, os, sys, argparse

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tasks-file", required=True)
    ap.add_argument("--out-file", required=True)         # filtered tasks (attempted only)
    ap.add_argument("--skip-out", required=True)         # json list of skipped task_ids (for the post-round updater)
    ap.add_argument("--health", required=True)           # persistent per-task health json
    ap.add_argument("--dead-threshold", type=int, default=2)
    ap.add_argument("--reprobe-every", type=int, default=3)
    a = ap.parse_args()

    d = json.load(open(a.tasks_file))
    tasks = d if isinstance(d, list) else d.get("tasks", [])
    health = json.load(open(a.health)) if os.path.exists(a.health) else {}

    active, skipped, reprobed = [], [], []
    for t in tasks:
        tid = t.get("task_id") or t.get("id")
        h = health.get(tid, {"dead_streak": 0, "skip_streak": 0})
        ds, ss = h.get("dead_streak", 0), h.get("skip_streak", 0)
        if ds >= a.dead_threshold and ss >= a.reprobe_every:
            reprobed.append(tid); active.append(t)        # force re-attempt
        elif ds >= a.dead_threshold:
            skipped.append(tid)                            # temporarily skip
        else:
            active.append(t)

    out = active if isinstance(d, list) else {**d, "tasks": active}
    json.dump(out, open(a.out_file, "w"), ensure_ascii=False, indent=2)
    json.dump(skipped, open(a.skip_out, "w"))

    print(f"[task-select] total={len(tasks)} attempt={len(active)} skip={len(skipped)} reprobe={len(reprobed)}", flush=True)
    if skipped:
        print(f"[task-select] ⚠️ SKIPPED this round (all-dead >= {a.dead_threshold} rounds; auto re-probe every {a.reprobe_every}): {skipped}", flush=True)
    if reprobed:
        print(f"[task-select] 🔁 RE-PROBING (forcing attempt after {a.reprobe_every} skips — would re-enter training if now viable): {reprobed}", flush=True)
    if not skipped and not reprobed:
        print("[task-select] no tasks skipped (none all-dead long enough).", flush=True)

if __name__ == "__main__":
    main()
