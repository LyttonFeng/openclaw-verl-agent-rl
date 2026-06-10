#!/usr/bin/env python3
"""Generate online RL rollouts from ledger task registry.

This driver consumes:

    data/meeting_analysis_val3_slim_train/claude_code_14_tasks.json

Each registry entry defines a prompt, workspace files, expected output file,
embedded automated grading function, reward contract, and GRPO grouping metadata.

The script runs OpenClaw rollouts, grades each rollout, applies the task's
process gate, and writes both flattened trajectory records and grouped GRPO
records.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent


def _find_repo_root(start: Path) -> Path:
    # Works from both rl/train/ (repo) and train/ (naive flat) layouts.
    for p in (start, *start.parents):
        if (p / "agent_loop").is_dir() and (p / "rewards").is_dir():
            return p
    return start.parent


REPO_ROOT = _find_repo_root(_THIS_DIR)
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "agent_loop"))
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(REPO_ROOT / "rewards"))

from agent_loop.diagnostics import diagnose  # noqa: E402
from meeting_reward import _parse_judge_score, _read_workspace_files, _summarize_transcript  # noqa: E402


DEFAULT_TASKS_FILE = (
    REPO_ROOT
    / "data"
    / "meeting_analysis_val3_slim_train"
    / "claude_code_14_tasks.json"
)

_ROLLOUT_RUN_ID = f"ledger_online_{int(time.time())}"
_rollout_seq = 0  # per-rollout unique short agent id (no reuse -> one wedge can't cascade)
_worker_lock = threading.Lock()


@dataclass
class LedgerTask:
    task_id: str
    name: str
    category: str
    grading_type: str
    prompt: str
    workspace_files: list[dict[str, str]]
    timeout_seconds: int
    expected_output_file: str
    grading_weights: dict[str, float]
    grade_function: str
    llm_rubric: list[dict[str, Any]]
    reward_contract: dict[str, Any]
    rl_grouping: dict[str, Any]
    source: str
    split: str
    meeting_family: str
    target_capability: str
    frontmatter: dict[str, Any]


def load_tasks(path: Path) -> list[LedgerTask]:
    raw = json.loads(path.read_text())
    if not isinstance(raw, list):
        raise ValueError(f"Expected list in {path}, got {type(raw).__name__}")

    tasks: list[LedgerTask] = []
    for item in raw:
        grading = item.get("grading") or {}
        task_id = item.get("id") or item.get("task_id")
        if not task_id:
            raise ValueError(f"Task record missing id/task_id: {item}")
        tasks.append(
            LedgerTask(
                task_id=task_id,
                name=item.get("name") or task_id,
                category=str(item.get("category") or "meeting"),
                grading_type=str(item.get("grading_type") or "hybrid"),
                prompt=item["prompt"],
                workspace_files=list(item.get("workspace_files") or []),
                timeout_seconds=int(item.get("timeout_seconds") or 180),
                expected_output_file=item["expected_output_file"],
                grading_weights=dict(grading.get("weights") or {"automated": 0.6, "llm_judge": 0.4}),
                grade_function=str(grading.get("grade_function") or ""),
                llm_rubric=list(grading.get("llm_rubric") or []),
                reward_contract=dict(item.get("reward_contract") or {}),
                rl_grouping=dict(item.get("rl_grouping") or {}),
                source=str(item.get("source") or "claw_data_agent_tasks"),
                split=str(item.get("split") or "train"),
                meeting_family=str(item.get("meeting_family") or item.get("meeting") or ""),
                target_capability=str(item.get("target_capability") or item.get("ledger_type") or ""),
                frontmatter={},
            )
        )
    return tasks


def create_agent(model: str, vllm_base_url: str, worker_idx: int) -> str:
    """Fresh agent per rollout (no reuse) so one wedged rollout (e.g. a long-doc
    context overflow) cannot cascade the rest to fatal. Short id to avoid
    OpenClaw id truncation breaking transcript lookup."""
    global _rollout_seq
    from lib_agent import ensure_agent_exists

    with _worker_lock:
        seq = _rollout_seq
        _rollout_seq += 1
    agent_id = f"ldg-w{worker_idx}r{seq:03d}"
    workspace_dir = Path(f"/tmp/pinchbench/{_ROLLOUT_RUN_ID}/{agent_id}/agent_workspace")
    workspace_dir.mkdir(parents=True, exist_ok=True)
    ensure_agent_exists(
        agent_id=agent_id,
        model_id=f"custom/{model}",
        workspace_dir=workspace_dir,
        base_url=vllm_base_url,
        api_key="dummy",
    )
    return agent_id


def run_single_rollout(
    task: LedgerTask,
    assets_dir: Path,
    vllm_base_url: str,
    model: str,
    worker_idx: int,
) -> dict[str, Any]:
    from lib_agent import execute_openclaw_task

    agent_id = create_agent(model, vllm_base_url, worker_idx)
    run_id = f"{task.task_id}_w{worker_idx}_{int(time.time() * 1000)}"
    try:
        result = execute_openclaw_task(
            task=task,
            agent_id=agent_id,
            model_id=f"custom/{model}",
            run_id=run_id,
            timeout_multiplier=2.0,
            skill_dir=REPO_ROOT,
        )
        return {
            "transcript": result.get("transcript", []),
            "workspace_path": result.get("workspace", ""),
            "execution_time": result.get("execution_time", 0),
            "timed_out": result.get("timed_out", False),
            "status": result.get("status", "unknown"),
            "exit_code": result.get("exit_code"),
            "stdout": result.get("stdout", ""),
            "stderr": result.get("stderr", ""),
        }
    except Exception as exc:
        logger.error("Rollout failed for %s: %s", task.task_id, exc)
        return {
            "transcript": [],
            "workspace_path": "",
            "execution_time": 0,
            "timed_out": False,
            "status": "error",
            "error": str(exc),
        }


def snapshot_workspace(workspace_path: str, snapshot_dir: Path) -> None:
    ws = Path(workspace_path)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    if not ws.is_dir():
        return

    skip_names = {
        "BOOTSTRAP.md",
        "AGENTS.md",
        "IDENTITY.md",
        "HEARTBEAT.md",
        "USER.md",
        "TOOLS.md",
        "SOUL.md",
    }
    skip_dirs = {".openclaw"}
    for item in ws.iterdir():
        if item.name in skip_dirs:
            continue
        if item.is_file() and item.name not in skip_names:
            shutil.copy2(item, snapshot_dir / item.name)


def save_transcript(transcript: list[Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for entry in transcript:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def run_automated_grade(task: LedgerTask, workspace_path: Path, transcript: list[Any]) -> dict[str, float]:
    namespace: dict[str, Any] = {}
    exec(task.grade_function, namespace)
    grade_func = namespace.get("grade")
    if not callable(grade_func):
        return {}
    scores = grade_func(transcript, str(workspace_path))
    if not isinstance(scores, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in scores.items():
        if isinstance(value, (int, float)):
            out[str(key)] = max(0.0, min(1.0, float(value)))
    return out


def run_llm_judge(task: LedgerTask, workspace_path: Path, transcript: list[Any]) -> float:
    import urllib.request

    if os.environ.get("MEETING_REWARD_AUTO_ONLY") == "1":
        return 0.0

    model = os.environ.get("MEETING_JUDGE_MODEL", "deepseek-chat")
    base_url = os.environ.get("MEETING_JUDGE_BASE_URL", "https://api.deepseek.com/v1")
    api_key = os.environ.get("DEEPSEEK_API_KEY", "")
    if not api_key:
        logger.warning("DEEPSEEK_API_KEY is missing; judge_score=0")
        return 0.0

    rubric_lines: list[str] = []
    for item in task.llm_rubric:
        name = item.get("name", "criterion")
        weight = item.get("weight", "")
        anchors = item.get("anchors", {})
        rubric_lines.append(f"- {name} (weight={weight}): {json.dumps(anchors, ensure_ascii=False)}")
    rubric = "\n".join(rubric_lines)

    judge_prompt = (
        f"## Task\n{task.prompt}\n\n"
        f"## Expected Output File\n{task.expected_output_file}\n\n"
        f"## Agent Transcript\n{_summarize_transcript(transcript)}\n\n"
        f"## Workspace Files Created\n{_read_workspace_files(str(workspace_path))}\n\n"
        f"## Grading Rubric\n{rubric}\n\n"
        "Score the output. Respond with ONLY JSON:\n"
        '{"scores": {"criterion_name": float_0_to_1}, "total": float_0_to_1, "notes": "brief"}'
    )
    endpoint = f"{base_url.rstrip('/')}/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are a strict grading function. Respond with ONLY a JSON object, "
                        "no prose and no markdown fences."
                    ),
                },
                {"role": "user", "content": judge_prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 2048,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=300) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        text = data["choices"][0]["message"]["content"]
        return max(0.0, min(1.0, float(_parse_judge_score(text))))
    except Exception as exc:
        logger.warning("Judge failed for %s: %s", task.task_id, exc)
        return 0.0


def passes_process_gate(task: LedgerTask, workspace_path: Path, breakdown: dict[str, float]) -> tuple[bool, list[str]]:
    gate = (task.reward_contract.get("process_gate") or {}) if task.reward_contract else {}
    failures: list[str] = []

    if gate.get("require_output_file", False) and not (workspace_path / task.expected_output_file).exists():
        failures.append("missing_expected_output_file")

    min_quote = gate.get("min_quote_verified")
    if min_quote is not None and breakdown.get("quote_verified", 0.0) < float(min_quote):
        failures.append("quote_verified_below_gate")

    min_table = gate.get("min_table_format")
    if min_table is not None and breakdown.get("table_format", 0.0) < float(min_table):
        failures.append("table_format_below_gate")

    return not failures, failures


_TIERED_SCORING = False  # opt-in via --tiered-scoring; default off = mean of all keys


def _aggregate_automated(breakdown: dict[str, float], reward_contract: dict[str, Any]) -> float:
    """Automated score from the per-check breakdown.

    Default (tiered off): equal-weight mean of all numeric checks — legacy.
    Tiered on: score ONLY from the contract's declared evidence keys
    (`score_keys`, e.g. quote_verified / entity_verified). Format/keyword
    checks then act purely as gates (process_gate) and earn no points, so
    keyword-stuffing cannot inflate the score — fixes 3.1, defuses 3.3.
    Coverage/completeness is left to the LLM rubric (judge weight).
    """
    numeric = {k: float(v) for k, v in breakdown.items() if isinstance(v, (int, float))}
    if not numeric:
        return 0.0
    if _TIERED_SCORING:
        score_keys = [k for k in (reward_contract.get("score_keys") or []) if k in numeric]
        if score_keys:
            return sum(numeric[k] for k in score_keys) / len(score_keys)
    return sum(numeric.values()) / len(numeric)


def grade_rollout(task: LedgerTask, rollout: dict[str, Any]) -> dict[str, Any]:
    workspace_path = Path(rollout.get("workspace_path") or "")
    transcript = rollout.get("transcript") or []

    diag = diagnose(
        trajectory=transcript,
        workspace_path=str(workspace_path),
        task_id=task.task_id,
        execution_time=rollout.get("execution_time", 0),
        timed_out=rollout.get("timed_out", False),
    )
    if diag.fatal:
        return {
            "score": 0.0,
            "automated_score": 0.0,
            "judge_score": 0.0,
            "reward_breakdown": {},
            "process_gate_passed": False,
            "process_gate_failures": ["fatal_diagnostics"],
            "diagnostics": diag.to_dict(),
            "skipped_judge": True,
        }

    breakdown = run_automated_grade(task, workspace_path, transcript)
    required = task.reward_contract.get("required_breakdown_keys") or []
    for key in required:
        breakdown.setdefault(key, 0.0)

    automated_score = _aggregate_automated(breakdown, task.reward_contract)

    gate_ok, gate_failures = passes_process_gate(task, workspace_path, breakdown)
    if not gate_ok:
        return {
            "score": 0.0,
            "automated_score": automated_score,
            "judge_score": 0.0,
            "reward_breakdown": breakdown,
            "process_gate_passed": False,
            "process_gate_failures": gate_failures,
            "diagnostics": diag.to_dict(),
            "skipped_judge": True,
        }

    judge_weight = float(task.grading_weights.get("llm_judge", 0.0))
    judge_score = run_llm_judge(task, workspace_path, transcript) if judge_weight > 0 else 0.0
    auto_weight = float(task.grading_weights.get("automated", 1.0))
    denom = auto_weight + judge_weight
    score = (auto_weight * automated_score + judge_weight * judge_score) / denom if denom else automated_score
    score = max(0.0, min(1.0, score))

    return {
        "score": score,
        "automated_score": automated_score,
        "judge_score": judge_score,
        "reward_breakdown": breakdown,
        "process_gate_passed": True,
        "process_gate_failures": [],
        "diagnostics": diag.to_dict(),
        "skipped_judge": judge_weight <= 0,
    }


def response_text_from_workspace(task: LedgerTask, workspace_path: Path) -> str:
    primary = workspace_path / task.expected_output_file
    if primary.exists():
        return primary.read_text(encoding="utf-8", errors="replace")
    for candidate in sorted(workspace_path.glob("*.md")):
        if candidate.name not in {"transcript.md", "meeting_transcript.md", "meeting-transcript.md"}:
            return candidate.read_text(encoding="utf-8", errors="replace")
    return ""


def build_grouped_records(records: list[dict[str, Any]], tasks_by_id: dict[str, LedgerTask]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for record in records:
        task = tasks_by_id[record["task_id"]]
        group_id = task.rl_grouping.get("group_id") or task.task_id
        grouped.setdefault(group_id, []).append(record)

    out: list[dict[str, Any]] = []
    for group_id, items in sorted(grouped.items()):
        task = tasks_by_id[items[0]["task_id"]]
        responses = []
        for item in sorted(items, key=lambda x: x["response_idx"]):
            responses.append(
                {
                    "response_idx": item["response_idx"],
                    "response": item.get("response", ""),
                    "score": item.get("score", 0.0),
                    "automated_score": item.get("automated_score", 0.0),
                    "judge_score": item.get("judge_score", 0.0),
                    "reward_breakdown": item.get("reward_breakdown", {}),
                    "process_gate_passed": item.get("process_gate_passed", False),
                    "process_gate_failures": item.get("process_gate_failures", []),
                    "workspace_path": item.get("workspace_path", ""),
                    "transcript_path": item.get("transcript_path", ""),
                }
            )
        out.append(
            {
                "group_id": group_id,
                "task_id": task.task_id,
                "source": task.source,
                "split": task.split,
                "meeting_family": task.meeting_family,
                "target_capability": task.target_capability,
                "prompt": task.prompt,
                "expected_output_file": task.expected_output_file,
                "responses": responses,
                "metadata": {
                    "n_responses": len(responses),
                    "score_min": min((r["score"] for r in responses), default=0.0),
                    "score_max": max((r["score"] for r in responses), default=0.0),
                    "score_spread": (
                        max((r["score"] for r in responses), default=0.0)
                        - min((r["score"] for r in responses), default=0.0)
                    ),
                },
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate online ledger RL rollouts")
    parser.add_argument("--tasks-file", type=Path, default=DEFAULT_TASKS_FILE)
    parser.add_argument("--assets-dir", type=Path, default=REPO_ROOT / "assets")
    parser.add_argument("--vllm-base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--n-responses", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--judge-model", default=None)
    parser.add_argument("--judge-base-url", default=None)
    parser.add_argument("--auto-only", action="store_true")
    parser.add_argument("--tiered-scoring", action="store_true",
                        help="OPT-IN (default off): automated score uses only the contract's "
                             "score_keys (evidence), making format/keyword checks pure gates. "
                             "Enable for the A/B treatment arm.")
    args = parser.parse_args()

    if args.judge_model:
        os.environ["MEETING_JUDGE_MODEL"] = args.judge_model
    if args.judge_base_url:
        os.environ["MEETING_JUDGE_BASE_URL"] = args.judge_base_url
    if args.auto_only:
        os.environ["MEETING_REWARD_AUTO_ONLY"] = "1"

    global _TIERED_SCORING
    _TIERED_SCORING = bool(args.tiered_scoring)
    if _TIERED_SCORING:
        logger.info("Tiered scoring ENABLED: automated score = mean(contract.score_keys)")

    tasks = load_tasks(args.tasks_file)
    tasks_by_id = {t.task_id: t for t in tasks}
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "transcripts").mkdir(parents=True, exist_ok=True)
    (output_dir / "workspaces").mkdir(parents=True, exist_ok=True)

    jobs: list[tuple[LedgerTask, int]] = []
    for task in tasks:
        n = args.n_responses or int(task.rl_grouping.get("rollouts_per_group") or 4)
        for resp_idx in range(n):
            jobs.append((task, resp_idx))

    logger.info("Generating ledger online rollouts: %d tasks, %d total jobs", len(tasks), len(jobs))
    logger.info("Model: %s @ %s", args.model, args.vllm_base_url)
    logger.info("Output: %s", output_dir)

    records: list[dict[str, Any]] = []
    write_lock = threading.Lock()

    def process_one(task: LedgerTask, resp_idx: int, worker_idx: int) -> dict[str, Any]:
        rollout = run_single_rollout(
            task=task,
            assets_dir=args.assets_dir,
            vllm_base_url=args.vllm_base_url,
            model=args.model,
            worker_idx=worker_idx,
        )
        snapshot_dir = output_dir / "workspaces" / f"{task.task_id}_resp{resp_idx}"
        snapshot_workspace(rollout.get("workspace_path", ""), snapshot_dir)
        rollout["workspace_path"] = str(snapshot_dir)

        transcript_path = output_dir / "transcripts" / f"{task.task_id}_resp{resp_idx}.jsonl"
        save_transcript(rollout.get("transcript") or [], transcript_path)

        grading = grade_rollout(task, rollout)
        record = {
            "task_id": task.task_id,
            "group_id": task.rl_grouping.get("group_id") or task.task_id,
            "prompt": task.prompt,
            "response_idx": resp_idx,
            "response": response_text_from_workspace(task, snapshot_dir),
            "score": grading.get("score", 0.0),
            "automated_score": grading.get("automated_score", 0.0),
            "judge_score": grading.get("judge_score", 0.0),
            "reward_breakdown": grading.get("reward_breakdown", {}),
            "process_gate_passed": grading.get("process_gate_passed", False),
            "process_gate_failures": grading.get("process_gate_failures", []),
            "diagnostics": grading.get("diagnostics", {}),
            "workspace_path": str(snapshot_dir),
            "transcript_path": str(transcript_path),
            "execution_time": rollout.get("execution_time", 0),
            "timed_out": rollout.get("timed_out", False),
            "exit_code": rollout.get("exit_code"),
            "stdout": rollout.get("stdout", ""),
            "stderr": rollout.get("stderr", ""),
            "status": rollout.get("status", "unknown"),
            "skipped_judge": grading.get("skipped_judge", False),
            "expected_output_file": task.expected_output_file,
            "meeting_family": task.meeting_family,
            "target_capability": task.target_capability,
            "_worker_idx": worker_idx,
        }
        return record

    flattened_path = output_dir / "graded_trajectories.jsonl"

    with flattened_path.open("w") as out_f:
        if args.num_workers <= 1:
            for task, resp_idx in jobs:
                logger.info("%s resp%d starting", task.task_id, resp_idx)
                record = process_one(task, resp_idx, worker_idx=0)
                record.pop("_worker_idx", None)
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                records.append(record)
                logger.info(
                    "%s resp%d score=%.3f gate=%s time=%.1fs",
                    task.task_id,
                    resp_idx,
                    record["score"],
                    record["process_gate_passed"],
                    record["execution_time"],
                )
        else:
            def wrap(job: tuple[LedgerTask, int]) -> dict[str, Any]:
                task, resp_idx = job
                tname = threading.current_thread().name
                try:
                    worker_idx = int(tname.rsplit("_", 1)[-1])
                except ValueError:
                    worker_idx = 0
                logger.info("[w%d] %s resp%d starting", worker_idx, task.task_id, resp_idx)
                return process_one(task, resp_idx, worker_idx)

            with ThreadPoolExecutor(max_workers=args.num_workers, thread_name_prefix="ledger") as pool:
                futures = [pool.submit(wrap, job) for job in jobs]
                for future in as_completed(futures):
                    record = future.result()
                    worker_idx = record.pop("_worker_idx", -1)
                    with write_lock:
                        out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                        out_f.flush()
                        records.append(record)
                    logger.info(
                        "[w%d] %s resp%d score=%.3f gate=%s time=%.1fs (%d/%d)",
                        worker_idx,
                        record["task_id"],
                        record["response_idx"],
                        record["score"],
                        record["process_gate_passed"],
                        record["execution_time"],
                        len(records),
                        len(jobs),
                    )

    grouped_records = build_grouped_records(records, tasks_by_id)
    grouped_path = output_dir / "grouped_grpo.jsonl"
    with grouped_path.open("w") as handle:
        for group in grouped_records:
            handle.write(json.dumps(group, ensure_ascii=False) + "\n")

    summary = {
        "tasks": len(tasks),
        "trajectories": len(records),
        "mean_score": sum((r["score"] for r in records), 0.0) / len(records) if records else 0.0,
        "gate_passed": sum(1 for r in records if r.get("process_gate_passed")),
        "grouped_grpo": str(grouped_path),
        "graded_trajectories": str(flattened_path),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    logger.info("Summary: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
