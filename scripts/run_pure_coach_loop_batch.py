#!/usr/bin/env python3
"""Run DSv4-Pro pure-coach loops over PinchBench tasks.

This is an orchestration script only. DSv4-Pro writes policies; Qwen3 executes
workers/final. No DSv4-Pro reference trajectories are provided.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


VAL5_TASKS = [
    "task_meeting_advisory_stakeholders",
    "task_meeting_council_votes",
    "task_meeting_gov_speaker_summary",
    "task_meeting_tech_action_items",
    "task_meeting_sentiment_analysis",
]


def _run(cmd: list[str], *, cwd: Path, env: dict[str, str], log_path: Path | None = None) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True) if log_path else None
    print("[loop] $", " ".join(cmd), flush=True)
    if log_path:
        with log_path.open("w") as log:
            proc = subprocess.run(cmd, cwd=str(cwd), env=env, stdout=log, stderr=subprocess.STDOUT, text=True)
    else:
        proc = subprocess.run(cmd, cwd=str(cwd), env=env, text=True)
    return proc.returncode


def _score(result_path: Path, task_id: str) -> tuple[float | None, str | None]:
    if not result_path.is_file():
        return None, None
    data = json.loads(result_path.read_text("utf-8"))
    for row in data.get("tasks", []):
        if row.get("task_id") != task_id:
            return None, None
        grading = row.get("grading") or {}
        score = grading.get("mean")
        return (float(score) if isinstance(score, (int, float)) else None), row.get("status")
    return None, None


def _result_notes(result_path: Path, task_id: str) -> str:
    if not result_path.is_file():
        return ""
    data = json.loads(result_path.read_text("utf-8"))
    for row in data.get("tasks", []):
        if row.get("task_id") != task_id:
            continue
        grading = row.get("grading") or {}
        run = (grading.get("runs") or [{}])[0]
        return str(run.get("notes") or "")
    return ""


def _existing_loop_results(output_root: Path, task_id: str) -> list[Path]:
    paths = []
    for result in sorted(output_root.glob(f"{task_id}/v*/run/*_team.json")):
        paths.append(result)
    return paths


def _qwen_team_transcripts(qwen_transcript_dir: Path, task_id: str) -> list[Path]:
    paths = [qwen_transcript_dir / f"{task_id}.jsonl", qwen_transcript_dir / f"{task_id}__lead_policy.jsonl"]
    paths.extend(qwen_transcript_dir / f"{task_id}__worker_{idx}.jsonl" for idx in range(1, 5))
    return [path for path in paths if path.is_file()]


def _loop_transcripts(loop_result_path: Path, task_id: str) -> list[Path]:
    # .../<task>/vN/run/<run_id>_team.json -> .../<task>/vN/run/<run_id>_transcripts/
    run_dir = loop_result_path.parent
    prefix = loop_result_path.name.removesuffix("_team.json")
    transcript_dir = run_dir / f"{prefix}_transcripts"
    paths = [transcript_dir / f"{task_id}.jsonl"]
    paths.extend(transcript_dir / f"{task_id}__worker_{idx}.jsonl" for idx in range(1, 7))
    paths.append(transcript_dir / f"{task_id}__lead_policy.jsonl")
    return [path for path in paths if path.is_file()]


def _task_prompt(tasks_dir: Path, task_id: str) -> Path:
    direct = tasks_dir / f"{task_id}.md"
    if direct.is_file():
        return direct
    matches = list(tasks_dir.rglob(f"{task_id}.md"))
    if not matches:
        raise FileNotFoundError(task_id)
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pure DSv4-Pro coach loop for tasks.")
    parser.add_argument("--tasks", nargs="*", default=VAL5_TASKS)
    parser.add_argument("--tasks-dir", default="pinchbench_tasks/meeting_analysis")
    parser.add_argument("--output-root", default="/workspace/verl_port/bench/pure_coach_loop_val5")
    parser.add_argument("--max-iters", type=int, default=4)
    parser.add_argument("--min-improvement", type=float, default=0.01)
    parser.add_argument("--target-score", type=float, default=0.90)
    parser.add_argument("--timeout-multiplier", type=float, default=8.0)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--repo-root", default="/workspace/openclaw-verl-agent-rl")
    parser.add_argument("--qwen-result", default="/workspace/verl_port/bench/team_policy_ab_val5/qwen_policy/qwenpol_val5_team.json")
    parser.add_argument("--qwen-transcript-dir", default="/workspace/verl_port/bench/team_policy_ab_val5/qwen_policy/qwenpol_val5_transcripts")
    parser.add_argument("--single-transcript-dir", default="/workspace/verl_port/bench/v38_base_r1/0015_transcripts")
    args = parser.parse_args()

    repo = Path(args.repo_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PINCHBENCH_FORCE_LOCAL_OPENCLAW": "1",
            "OPENCLAW_HOST": "localhost",
            "OPENCLAW_MODEL_REASONING": "0",
            "PINCHBENCH_OPENCLAW_CONTEXT_WINDOW": "65536",
            "PINCHBENCH_OPENCLAW_MAX_TOKENS": "8192",
        }
    )

    summary: dict[str, Any] = {"tasks": {}, "started_at": time.time()}
    summary_path = output_root / "summary.json"

    for task_id in args.tasks:
        print(f"[loop] task={task_id}", flush=True)
        task_root = output_root / task_id
        task_root.mkdir(parents=True, exist_ok=True)
        single_transcript = Path(args.single_transcript_dir) / f"{task_id}.jsonl"
        qwen_transcripts = _qwen_team_transcripts(Path(args.qwen_transcript_dir), task_id)
        prior_results = _existing_loop_results(output_root, task_id)
        best = -1.0
        last_score: float | None = None
        task_rows = []

        for iteration in range(1, args.max_iters + 1):
            iter_root = task_root / f"v{iteration}"
            policy_path = repo / "experiments" / "team_policies" / f"{task_id}_pure_loop_v{iteration}.json"
            report_path = iter_root / "coach_report.json"
            run_root = iter_root / "run"
            run_id = f"{task_id}_pure_v{iteration}"
            result_path = run_root / f"{run_id}_team.json"

            if result_path.is_file():
                score, status = _score(result_path, task_id)
                print(f"[loop] existing task={task_id} v={iteration} score={score} status={status}", flush=True)
            else:
                coach_cmd = [
                    "python3",
                    "scripts/coach_team_policy.py",
                    "--task",
                    task_id,
                    "--task-prompt-file",
                    str(_task_prompt(repo / args.tasks_dir, task_id)),
                    "--result",
                    args.qwen_result,
                    "--output-policy",
                    str(policy_path),
                    "--output-report",
                    str(report_path),
                ]
                if single_transcript.is_file():
                    coach_cmd.extend(["--transcript", str(single_transcript)])
                for path in qwen_transcripts:
                    coach_cmd.extend(["--transcript", str(path)])
                for path in prior_results:
                    coach_cmd.extend(["--result", str(path)])
                    for transcript in _loop_transcripts(path, task_id):
                        coach_cmd.extend(["--transcript", str(transcript)])
                if iteration > 1:
                    prev_policy = repo / "experiments" / "team_policies" / f"{task_id}_pure_loop_v{iteration - 1}.json"
                    if prev_policy.is_file():
                        coach_cmd.extend(["--base-policy", str(prev_policy)])

                rc = _run(coach_cmd, cwd=repo, env=env, log_path=iter_root / "coach.log")
                if rc != 0:
                    print(f"[loop] coach failed task={task_id} v={iteration} rc={rc}", flush=True)
                    break

                bench_cmd = [
                    "python3",
                    "scripts/team_agent_benchmark.py",
                    "--task",
                    task_id,
                    "--policy-file",
                    str(policy_path),
                    "--policy-model",
                    "qwen3-base",
                    "--policy-base-url",
                    "http://127.0.0.1:8770/v1",
                    "--worker-model",
                    "qwen3-base",
                    "--worker-base-url",
                    "http://127.0.0.1:8770/v1",
                    "--final-model",
                    "qwen3-base",
                    "--final-base-url",
                    "http://127.0.0.1:8770/v1",
                    "--workers",
                    str(args.workers),
                    "--namespace",
                    f"pcl{abs(hash(task_id)) % 100000}-{iteration}",
                    "--output-dir",
                    str(run_root),
                    "--run-id",
                    run_id,
                    "--timeout-multiplier",
                    str(args.timeout_multiplier),
                ]
                rc = _run(bench_cmd, cwd=repo, env=env, log_path=iter_root / "run.log")
                if rc != 0:
                    print(f"[loop] run failed task={task_id} v={iteration} rc={rc}", flush=True)
                score, status = _score(result_path, task_id)

            notes = _result_notes(result_path, task_id)
            row = {"iteration": iteration, "score": score, "status": status, "notes": notes, "result": str(result_path)}
            task_rows.append(row)
            prior_results.append(result_path)
            if score is not None:
                best = max(best, score)
            summary["tasks"][task_id] = {"best": best, "runs": task_rows}
            summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", "utf-8")

            if score is None:
                break
            if score >= args.target_score:
                print(f"[loop] saturated target task={task_id} score={score}", flush=True)
                break
            if last_score is not None and score - last_score < args.min_improvement:
                print(f"[loop] saturated improvement task={task_id} prev={last_score} score={score}", flush=True)
                break
            if status == "timeout" and last_score is not None:
                print(f"[loop] saturated timeout task={task_id} score={score}", flush=True)
                break
            last_score = score

    summary["finished_at"] = time.time()
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print(summary_path)


if __name__ == "__main__":
    main()
