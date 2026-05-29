#!/usr/bin/env python3
"""Pick top-K trajectories per task by score (DSv4 Flash teacher data).

Reads extracted JSONL from 1_extract_oc_transcript.py and writes a filtered
JSONL keeping the K best samples per task_id. Ties broken by message count
(fewer = more efficient).
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--top-k", type=int, default=2,
                    help="Keep top-K trajectories per task (default 2)")
    ap.add_argument("--min-score", type=float, default=0.5,
                    help="Drop records below this score (default 0.5)")
    ap.add_argument("--min-msgs", type=int, default=0,
                    help="Drop trajectories with < N messages (default 0 = off). Use to filter short/incomplete traj")
    args = ap.parse_args()

    by_task = defaultdict(list)
    n_in = 0
    n_no_score = 0
    n_below_min = 0
    n_too_short = 0
    with open(args.input) as f:
        for line in f:
            r = json.loads(line)
            n_in += 1
            s = r.get("score")
            if s is None:
                n_no_score += 1
                continue
            if s < args.min_score:
                n_below_min += 1
                continue
            if args.min_msgs > 0 and r.get("n_messages", 0) < args.min_msgs:
                n_too_short += 1
                continue
            by_task[r["task_id"]].append(r)

    n_out = 0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        for task_id, recs in by_task.items():
            recs.sort(key=lambda r: (-r["score"], r["n_messages"]))
            for r in recs[: args.top_k]:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
                n_out += 1

    print(f"in: {n_in}  no_score: {n_no_score}  below_min_score: {n_below_min}  too_short: {n_too_short}")
    print(f"tasks: {len(by_task)}  kept: {n_out}  output: {out_path}")
    print("\nPer-task top scores kept:")
    for tid in sorted(by_task):
        recs = sorted(by_task[tid], key=lambda r: (-r["score"], r["n_messages"]))[: args.top_k]
        ss = [f"{r['score']:.2f}({r['n_messages']}msg)" for r in recs]
        print(f"  {tid:50s} {' '.join(ss)}")


if __name__ == "__main__":
    main()
