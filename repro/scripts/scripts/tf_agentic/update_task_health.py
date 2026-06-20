#!/usr/bin/env python3
"""Update per-task health AFTER a rollout round (single writer).

For each ATTEMPTED task: was it all-dead (every rollout timeout or empty)?
  all-dead  -> dead_streak += 1, skip_streak = 0
  alive(>=2 good) -> dead_streak = 0, skip_streak = 0
For each SKIPPED task: skip_streak += 1 (dead_streak unchanged).
Keeps a short rolling history string for transparency.
"""
import json, os, sys, argparse
from collections import defaultdict

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--graded", required=True)     # graded_trajectories.jsonl from this round
    ap.add_argument("--skip-list", required=True)  # json list of task_ids skipped this round
    ap.add_argument("--health", required=True)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.graded) if l.strip()] if os.path.exists(a.graded) else []
    skipped = json.load(open(a.skip_list)) if os.path.exists(a.skip_list) else []
    health = json.load(open(a.health)) if os.path.exists(a.health) else {}

    good = defaultdict(int)
    seen = set()
    for r in rows:
        tid = r.get("task_id"); seen.add(tid)
        resp = r.get("response") or ""
        if len(resp) >= 50 and not r.get("timed_out"):
            good[tid] += 1

    def bump(tid, **kw):
        h = health.get(tid, {"dead_streak": 0, "skip_streak": 0, "history": ""})
        h.update(kw); health[tid] = h

    for tid in seen:
        h = health.get(tid, {"dead_streak": 0, "skip_streak": 0, "history": ""})
        if good[tid] >= 2:
            h = {"dead_streak": 0, "skip_streak": 0, "history": (h.get("history", "") + "A")[-12:]}
        else:
            h = {"dead_streak": h.get("dead_streak", 0) + 1, "skip_streak": 0,
                 "history": (h.get("history", "") + "D")[-12:]}
        health[tid] = h
    for tid in skipped:
        h = health.get(tid, {"dead_streak": 0, "skip_streak": 0, "history": ""})
        h = {"dead_streak": h.get("dead_streak", 0), "skip_streak": h.get("skip_streak", 0) + 1,
             "history": (h.get("history", "") + "s")[-12:]}
        health[tid] = h

    json.dump(health, open(a.health, "w"), ensure_ascii=False, indent=2)
    print("[task-health] updated:", flush=True)
    for tid, h in sorted(health.items()):
        print(f"   {tid:34s} dead_streak={h['dead_streak']} skip_streak={h['skip_streak']} hist={h.get('history','')}", flush=True)

if __name__ == "__main__":
    main()
