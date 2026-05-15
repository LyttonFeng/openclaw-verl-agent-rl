#!/usr/bin/env python3
"""Summarize one or more jiuwenclaw bench runs.

Usage:
    bench_summarize.py <results.json>
    bench_summarize.py <out_root>          # walks for runN/<ts>/results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from statistics import mean, pstdev


def _find_results(root: Path) -> list[Path]:
    if root.is_file():
        return [root]
    files = sorted(root.rglob("results.json"))
    if not files:
        # try partial as fallback
        files = sorted(root.rglob("results.partial.json"))
    return files


def _per_task_score(run: dict) -> list[tuple[str, str, float | None]]:
    rows = []
    for t in run.get("tasks", []):
        g = t.get("grading") or {}
        runs = g.get("runs") or []
        s = runs[0].get("score") if runs else None
        rows.append((t.get("task_id", "?"), t.get("status", "?"), s))
    return rows


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(2)
    target = Path(sys.argv[1])
    files = _find_results(target)
    if not files:
        print(f"no results.json under {target}")
        sys.exit(1)

    per_run_avgs = []
    for f in files:
        data = json.loads(f.read_text())
        rows = _per_task_score(data)
        scores = [s for _, _, s in rows if s is not None]
        avg = (sum(scores) / len(scores)) if scores else 0.0
        per_run_avgs.append(avg)
        print(f"=== {f} ===")
        print(f"  completed={data.get('completed')} task_count={data.get('task_count')}")
        for tid, st, s in rows:
            print(f"  {tid[:45]:45s} status={st:8s} score={s}")
        print(f"  avg = {avg:.4f}")
        print()

    if len(per_run_avgs) > 1:
        m = mean(per_run_avgs)
        sd = pstdev(per_run_avgs) if len(per_run_avgs) > 1 else 0.0
        print(f"--- across {len(per_run_avgs)} runs ---")
        print(f"  mean = {m:.4f}  std = {sd:.4f}  range = [{min(per_run_avgs):.4f}, {max(per_run_avgs):.4f}]")


if __name__ == "__main__":
    main()
