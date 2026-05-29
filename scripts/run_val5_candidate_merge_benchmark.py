#!/usr/bin/env python3
"""Run a Val5 candidate-merge benchmark with the same task set and judge
settings as the isolated single-agent benchmark.

This wrapper keeps the comparison clean:
  - same Val5 task list
  - same isolated runtime policy
  - same judge backend/model
  - same output-file schema

The only difference is the execution policy:
  1. Generate K independent candidates per task
  2. Let a lead model select / merge them
  3. Grade the merged report with the normal terminal judge

Per-task results are written to:
  <output-dir>/<run_id>/task_<task_id>/result.json

The top-level summary is written to:
  <output-dir>/summary.json
  <output-dir>/summary.txt
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TASKS = [
    "task_meeting_advisory_stakeholders",
    "task_meeting_council_votes",
    "task_meeting_gov_speaker_summary",
    "task_meeting_tech_action_items",
    "task_meeting_sentiment_analysis",
]


def _task_output_file(task_id: str) -> str:
    return {
        "task_meeting_advisory_stakeholders": "stakeholder_analysis.md",
        "task_meeting_council_votes": "votes_report.md",
        "task_meeting_gov_speaker_summary": "speaker_summary.md",
        "task_meeting_tech_action_items": "action_items.md",
        "task_meeting_sentiment_analysis": "sentiment_analysis.md",
    }.get(task_id, "output.md")


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _run_task(
    *,
    task_id: str,
    run_idx: int,
    args: argparse.Namespace,
    run_output_dir: Path,
) -> dict[str, Any]:
    task_dir = run_output_dir / f"run{run_idx:02d}" / task_id
    task_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        str(REPO_ROOT / "scripts" / "swarm_policy" / "run_candidate_merge_experiment.py"),
        "--task-id",
        task_id,
        "--tasks-dir",
        args.tasks_dir,
        "--assets-dir",
        args.assets_dir,
        "--candidate-base-urls",
        args.candidate_base_urls,
        "--candidate-model",
        args.candidate_model,
        "--lead-base-url",
        args.lead_base_url,
        "--lead-model",
        args.lead_model,
        "--output-file",
        _task_output_file(task_id),
        "--output-dir",
        str(task_dir),
        "--timeout",
        str(args.timeout),
    ]

    env = os.environ.copy()
    env["PINCHBENCH_FORCE_LOCAL_OPENCLAW"] = env.get("PINCHBENCH_FORCE_LOCAL_OPENCLAW", "1")
    env["PINCHBENCH_SKIP_OPENCLAW_WEB_PREFLIGHT"] = env.get(
        "PINCHBENCH_SKIP_OPENCLAW_WEB_PREFLIGHT", "1"
    )
    env["PINCHBENCH_MODEL_TEMPERATURE"] = str(args.candidate_temperature)
    env["PINCHBENCH_MODEL_TOP_P"] = str(args.candidate_top_p)
    env["PINCHBENCH_GRADE_JUDGE_MODEL"] = args.judge_model
    env["PINCHBENCH_GRADE_JUDGE_BASE_URL"] = args.judge_base_url
    if args.judge_api_key_env:
        api_key = os.environ.get(args.judge_api_key_env, "")
        if api_key:
            env["PINCHBENCH_GRADE_JUDGE_API_KEY"] = api_key

    print(f"[{task_id} run {run_idx}] launching candidate-merge benchmark", flush=True)
    subprocess.run(cmd, check=True, env=env)
    result_path = task_dir / "result.json"
    if not result_path.exists():
        raise RuntimeError(f"missing result file: {result_path}")
    return _load_json(result_path)


def _mean(values: list[float]) -> float:
    return statistics.mean(values) if values else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Val5 candidate-merge benchmark.")
    parser.add_argument(
        "--tasks",
        default=",".join(DEFAULT_TASKS),
        help="Comma-separated Val5 task ids.",
    )
    parser.add_argument("--tasks-dir", default=str(REPO_ROOT / "pinchbench_tasks" / "meeting_analysis"))
    parser.add_argument("--assets-dir", default=str(REPO_ROOT / "pinchbench_tasks" / "meeting_analysis" / "assets"))
    parser.add_argument("--candidate-base-urls", required=True, help="Comma-separated candidate vLLM URLs.")
    parser.add_argument("--candidate-model", default="qwen3-r08-lora")
    parser.add_argument("--lead-base-url", default="http://localhost:8767/v1")
    parser.add_argument("--lead-model", default="qwen3-base")
    parser.add_argument("--judge-model", default="deepseek-v4-pro")
    parser.add_argument("--judge-base-url", default="https://api.deepseek.com/v1")
    parser.add_argument("--judge-api-key-env", default="DEEPSEEK_API_KEY")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--candidate-temperature", type=float, default=0.2)
    parser.add_argument("--candidate-top-p", type=float, default=0.9)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]
    run_output_dir = Path(args.output_dir)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    run_id = time.strftime("%Y%m%d_%H%M%S")

    records: list[dict[str, Any]] = []
    for run_idx in range(1, args.runs + 1):
        for task_id in tasks:
            rec = _run_task(task_id=task_id, run_idx=run_idx, args=args, run_output_dir=run_output_dir)
            rec["run_idx"] = run_idx
            records.append(rec)

    by_task: dict[str, list[dict[str, Any]]] = {}
    for rec in records:
        by_task.setdefault(rec["task_id"], []).append(rec)

    summary_tasks: dict[str, Any] = {}
    merged_means: list[float] = []
    oracle_means: list[float] = []
    random_means: list[float] = []
    for task_id, rows in by_task.items():
        merged = [float(r.get("merged_score", 0.0)) for r in rows]
        oracle = [
            max((float(x) for x in r.get("candidate_scores", []) or [0.0]), default=0.0)
            for r in rows
        ]
        random = [
            _mean([float(x) for x in r.get("candidate_scores", []) or [0.0]])
            for r in rows
        ]
        summary_tasks[task_id] = {
            "merged": merged,
            "oracle": oracle,
            "random": random,
            "merged_mean": _mean(merged),
            "oracle_mean": _mean(oracle),
            "random_mean": _mean(random),
        }
        merged_means.extend(merged)
        oracle_means.extend(oracle)
        random_means.extend(random)

    summary = {
        "run_id": run_id,
        "mode": "val5_candidate_merge_benchmark",
        "tasks": tasks,
        "runs": args.runs,
        "candidate_base_urls": [u.strip() for u in args.candidate_base_urls.split(",") if u.strip()],
        "candidate_model": args.candidate_model,
        "lead_model": args.lead_model,
        "judge_model": args.judge_model,
        "candidate_temperature": args.candidate_temperature,
        "candidate_top_p": args.candidate_top_p,
        "merged_mean": _mean(merged_means),
        "oracle_mean": _mean(oracle_means),
        "random_mean": _mean(random_means),
        "tasks_summary": summary_tasks,
        "records": records,
    }

    (run_output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    summary_txt = [
        f"Val5 candidate-merge benchmark ({run_id})",
        f"candidate_mean: {summary['merged_mean']:.4f}",
        f"oracle_mean:    {summary['oracle_mean']:.4f}",
        f"random_mean:    {summary['random_mean']:.4f}",
        "",
        "Per-task means:",
    ]
    for task_id in tasks:
        t = summary_tasks[task_id]
        summary_txt.append(
            f"- {task_id}: merged={t['merged_mean']:.4f} oracle={t['oracle_mean']:.4f} random={t['random_mean']:.4f}"
        )
    (run_output_dir / "summary.txt").write_text("\n".join(summary_txt) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
