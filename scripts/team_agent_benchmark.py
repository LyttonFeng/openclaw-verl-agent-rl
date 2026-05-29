#!/usr/bin/env python3
"""Standalone lead-agent/team-agent runner for PinchBench tasks.

This intentionally does not modify the existing single-agent benchmark path.
It reuses PinchBench's task loading, workspace preparation, OpenClaw agent
configuration, transcript discovery, and grader helpers, but orchestrates a
separate workflow:

1. Lead agent emits a JSON team policy.
2. Worker agents inspect long source files and write compact notes.
3. Lead agent reads the worker notes and writes the final task deliverable.

The intended training split is that only the lead agent's policy/final-summary
behavior is trainable. Worker agents can be fixed strong models.
"""

from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from lib_agent import (  # noqa: E402
    _extract_usage_from_transcript,
    _has_successful_assistant_turn,
    _load_transcript,
    _run_with_process_group,
    cleanup_agent_sessions,
    ensure_agent_exists,
    prepare_task_workspace,
)
from lib_grading import grade_task, resolve_judge_backend_from_env  # noqa: E402
from lib_tasks import Task, TaskLoader, resolve_task_markdown_path  # noqa: E402


LOGGER = logging.getLogger("team_agent_benchmark")

VAL5_TASKS = [
    "task_meeting_advisory_stakeholders",
    "task_meeting_council_votes",
    "task_meeting_gov_speaker_summary",
    "task_meeting_tech_action_items",
    "task_meeting_sentiment_analysis",
]


def _bool_env(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).strip().lower() in {"1", "true", "yes", "on"}


@contextlib.contextmanager
def _temporary_env(updates: dict[str, str]):
    old: dict[str, Optional[str]] = {}
    for key, value in updates.items():
        old[key] = os.environ.get(key)
        os.environ[key] = value
    try:
        yield
    finally:
        for key, value in old.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value


@contextlib.contextmanager
def _openclaw_home_lock(lock_dir: Path, openclaw_home_parent: Path):
    lock_dir.mkdir(parents=True, exist_ok=True)
    lock_key = hashlib.sha1(str(openclaw_home_parent.expanduser()).encode("utf-8")).hexdigest()[:12]
    lock_path = lock_dir / f"home-{lock_key}.lock"
    with lock_path.open("w") as lock_file:
        LOGGER.info("Waiting for team-agent OpenClaw home lock %s", lock_path)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            lock_file.write(f"{os.getpid()}\n")
            lock_file.flush()
            yield lock_path
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def _openclaw_store_from_home_parent(home_parent: Path) -> Path:
    return home_parent.expanduser() / ".openclaw"


def _cleanup_namespace_agents(openclaw_home_parent: Path, namespace: str) -> int:
    """Delete only agents owned by this team benchmark namespace."""
    agents_dir = _openclaw_store_from_home_parent(openclaw_home_parent) / "agents"
    if not agents_dir.exists():
        return 0
    prefix = f"{namespace}-"
    removed = 0
    for path in agents_dir.iterdir():
        if not path.is_dir() or not path.name.startswith(prefix):
            continue
        try:
            shutil.rmtree(path)
            removed += 1
        except OSError as exc:
            LOGGER.warning("Failed to remove namespaced agent %s: %s", path, exc)
    return removed


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict) and item.get("type") == "text":
            parts.append(str(item.get("text") or ""))
    return "".join(parts)


def _assistant_text(transcript: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    for event in transcript:
        if event.get("type") != "message":
            continue
        msg = event.get("message") or {}
        if msg.get("role") != "assistant":
            continue
        text = _text_from_content(msg.get("content"))
        if text.strip():
            parts.append(text.strip())
    return "\n\n".join(parts)


def _json_from_text(text: str) -> Optional[dict[str, Any]]:
    candidates = [text.strip()]
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    if fence:
        candidates.insert(0, fence.group(1).strip())
    obj = re.search(r"\{[\s\S]*\}", text)
    if obj:
        candidates.append(obj.group(0))
    for candidate in candidates:
        if not candidate:
            continue
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _markdown_files_from_task(task: Task) -> list[str]:
    files: list[str] = []
    for spec in task.workspace_files:
        if not isinstance(spec, dict):
            continue
        path = spec.get("dest") or spec.get("path")
        if isinstance(path, str) and path.endswith(".md"):
            files.append(path)
    for match in re.finditer(r"`([^`]+\.md)`", task.prompt):
        files.append(match.group(1))
    for match in re.finditer(r"\b([A-Za-z0-9_.-]+\.md)\b", task.prompt):
        files.append(match.group(1))
    out: list[str] = []
    for item in files:
        if item not in out:
            out.append(item)
    return out


def _default_policy(task: Task, workers: int) -> dict[str, Any]:
    files = _markdown_files_from_task(task)
    transcript_files = [
        f for f in files if "transcript" in f.lower() or "meeting" in f.lower()
    ] or files[:1]
    output_files = [
        f
        for f in files
        if f not in transcript_files
        and any(word in f.lower() for word in ("analysis", "summary", "report", "items"))
    ]
    focuses = [
        "early sections: meeting framing, participants, agenda, initial factual anchors",
        "middle sections: key arguments, decisions, conflicts, named positions, evidence",
        "late sections: final outcomes, action items, votes, unresolved issues, closing context",
        "rubric cross-check: verify names, numbers, dates, coverage, and final-file requirements",
    ]
    offsets = ["0,200,400", "500,700,900", "1000,1300,1600", "0,600,1200,1800"]
    return {
        "strategy": "Use fixed workers to gather compact evidence notes from long transcripts, then have the lead synthesize the final deliverable from notes.",
        "transcript_files": transcript_files,
        "expected_output_files": output_files,
        "workers": [
            {
                "id": f"worker_{idx + 1}",
                "focus": focuses[idx % len(focuses)],
                "transcript_files": transcript_files,
                "read_offsets": offsets[idx % len(offsets)],
                "note_path": f"team_notes/worker_{idx + 1}.md",
            }
            for idx in range(workers)
        ],
    }


def _normalize_policy(policy: Optional[dict[str, Any]], task: Task, workers: int) -> dict[str, Any]:
    fallback = _default_policy(task, workers)
    if not isinstance(policy, dict):
        return fallback
    policy.setdefault("strategy", fallback["strategy"])
    policy.setdefault("transcript_files", fallback["transcript_files"])
    policy.setdefault("expected_output_files", fallback["expected_output_files"])
    raw_workers = policy.get("workers")
    if not isinstance(raw_workers, list) or not raw_workers:
        raw_workers = fallback["workers"]
    normalized = []
    for idx, worker in enumerate(raw_workers[:workers]):
        if not isinstance(worker, dict):
            worker = {"focus": str(worker)}
        wid = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(worker.get("id") or f"worker_{idx + 1}")).strip("_")
        worker["id"] = wid or f"worker_{idx + 1}"
        worker.setdefault("focus", fallback["workers"][idx % len(fallback["workers"])]["focus"])
        worker.setdefault("transcript_files", fallback["transcript_files"])
        worker.setdefault("read_offsets", fallback["workers"][idx % len(fallback["workers"])]["read_offsets"])
        worker.setdefault("note_path", f"team_notes/{worker['id']}.md")
        normalized.append(worker)
    while len(normalized) < workers:
        normalized.append(fallback["workers"][len(normalized)])
    policy["workers"] = normalized
    return policy


def _run_agent_message(
    *,
    agent_id: str,
    session_id: str,
    prompt: str,
    workspace: Path,
    timeout: float,
) -> subprocess.CompletedProcess[str]:
    openclaw_path = os.environ.get("OPENCLAW_PATH", "openclaw")
    return _run_with_process_group(
        [
            openclaw_path,
            "agent",
            "--agent",
            agent_id,
            "--session-id",
            session_id,
            "--message",
            prompt,
            "--local",
        ],
        cwd=str(workspace),
        timeout=timeout,
        shell=False,
    )


def _archive_transcript(path: Optional[Path], output_dir: Path, name: str) -> None:
    if not path:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(path, output_dir / name)


def _note_preview(workspace: Path, policy: dict[str, Any], max_chars: int) -> str:
    parts = []
    for worker in policy.get("workers") or []:
        rel = str(worker.get("note_path") or "")
        path = workspace / rel
        if not path.is_file():
            parts.append(f"## {worker.get('id')}\nMISSING NOTE: {rel}")
            continue
        text = path.read_text("utf-8", errors="replace")
        if len(text) > max_chars:
            text = text[:max_chars] + "\n\n[truncated]"
        parts.append(f"## {worker.get('id')} ({rel})\n{text}")
    return "\n\n".join(parts)


def run_team_task(
    *,
    task: Task,
    run_id: str,
    policy_model: str,
    policy_base_url: Optional[str],
    policy_api_key: str,
    final_model: str,
    final_base_url: Optional[str],
    final_api_key: str,
    worker_model: str,
    worker_base_url: Optional[str],
    worker_api_key: str,
    agent_prefix: str,
    skill_dir: Path,
    output_dir: Path,
    workers: int,
    timeout_multiplier: float,
    verbose: bool,
    policy_override: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    short_id = hashlib.sha1(f"{task.task_id}:{run_id}".encode("utf-8")).hexdigest()[:8]
    policy_agent_id = f"{agent_prefix}-{short_id}-pol"
    final_agent_id = f"{agent_prefix}-{short_id}-fin"
    workspace_probe = Path(os.environ.get("PINCHBENCH_RUN_ROOT", "/tmp/pinchbench")) / run_id / "agent_workspace"
    ensure_agent_exists(
        policy_agent_id,
        policy_model,
        workspace_probe,
        base_url=policy_base_url,
        api_key=policy_api_key,
    )
    cleanup_agent_sessions(policy_agent_id)
    workspace = prepare_task_workspace(skill_dir, run_id, task, policy_agent_id)
    ensure_agent_exists(
        policy_agent_id,
        policy_model,
        workspace,
        base_url=policy_base_url,
        api_key=policy_api_key,
    )
    ensure_agent_exists(
        final_agent_id,
        final_model,
        workspace,
        base_url=final_base_url,
        api_key=final_api_key,
    )
    cleanup_agent_sessions(final_agent_id)

    started_at = time.time()
    session_id = f"{task.task_id}_team_{int(started_at * 1000)}"
    timeout_seconds = task.timeout_seconds * timeout_multiplier
    output_dir.mkdir(parents=True, exist_ok=True)
    notes_dir = workspace / "team_notes"
    notes_dir.mkdir(parents=True, exist_ok=True)
    stdout = ""
    stderr = ""
    exit_code = 0
    timed_out = False

    def remaining() -> float:
        return max(0.0, timeout_seconds - (time.time() - started_at))

    policy_prompt = f"""You are the lead agent for a long-context PinchBench task.

Do not solve the task yet. First produce a team policy for fixed helper agents.
The helper agents will gather compact evidence notes; you will later synthesize
the final deliverable. Return ONLY JSON, no markdown fences.

Original task:
{task.prompt}

Required JSON schema:
{{
  "strategy": "short practical plan",
  "transcript_files": ["source transcript file names"],
  "expected_output_files": ["final deliverable file names"],
  "workers": [
    {{
      "id": "worker_1",
      "focus": "specific evidence to extract",
      "transcript_files": ["..."],
      "read_offsets": "suggested offsets/sections",
      "note_path": "team_notes/worker_1.md"
    }}
  ]
}}

Create exactly {workers} workers. Optimize for a 32k-64k lead context: workers
must write concise notes, not raw transcript dumps.
"""
    lead_transcript: list[dict[str, Any]] = []
    lead_transcript_path: Optional[Path] = None
    raw_policy = policy_override
    if raw_policy is None:
        try:
            result = _run_agent_message(
                agent_id=policy_agent_id,
                session_id=session_id,
                prompt=policy_prompt,
                workspace=workspace,
                timeout=min(remaining(), max(30.0, timeout_seconds * 0.2)),
            )
            stdout += result.stdout
            stderr += result.stderr
            exit_code = result.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout += str(exc.stdout or "")
            stderr += str(exc.stderr or "")
        except FileNotFoundError as exc:
            exit_code = 127
            stderr += f"openclaw command not found: {exc}"

        lead_transcript, lead_transcript_path = _load_transcript(policy_agent_id, session_id, started_at)
        raw_policy = _json_from_text(_assistant_text(lead_transcript))
    policy = _normalize_policy(raw_policy, task, workers)
    (notes_dir / "team_policy.json").write_text(json.dumps(policy, indent=2, ensure_ascii=False) + "\n", "utf-8")
    _archive_transcript(lead_transcript_path, output_dir, f"{task.task_id}__lead_policy.jsonl")

    worker_results = []
    for idx, worker in enumerate(policy["workers"], start=1):
        if remaining() <= 5:
            timed_out = True
            break
        worker_agent_id = f"{agent_prefix}-{short_id}-w{idx}"
        ensure_agent_exists(
            worker_agent_id,
            worker_model,
            workspace,
            base_url=worker_base_url,
            api_key=worker_api_key,
        )
        cleanup_agent_sessions(worker_agent_id)
        worker_session = f"{session_id}_{worker['id']}"
        note_path = str(worker.get("note_path") or f"team_notes/worker_{idx}.md")
        worker_prompt = f"""You are a fixed helper agent in a team workflow.

Do NOT write the final task output. Your only deliverable is `{note_path}`.

Original task:
{task.prompt}

Full team policy:
{json.dumps(policy, indent=2, ensure_ascii=False)}

Your assignment:
{json.dumps(worker, indent=2, ensure_ascii=False)}

Instructions:
- Read relevant transcript file(s), using the suggested offsets/sections when useful.
- Extract concrete evidence: names, dates, decisions, votes, action items, stakeholder positions, sentiment evidence, and exact facts relevant to your focus.
- Keep the note compact and structured. Do not paste long transcript chunks.
- Write exactly one note file: `{note_path}`.
"""
        worker_started = time.time()
        worker_exit = 0
        try:
            result = _run_agent_message(
                agent_id=worker_agent_id,
                session_id=worker_session,
                prompt=worker_prompt,
                workspace=workspace,
                timeout=min(remaining(), max(30.0, timeout_seconds * 0.55 / workers)),
            )
            stdout += result.stdout
            stderr += result.stderr
            worker_exit = result.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            worker_exit = -1
            stdout += str(exc.stdout or "")
            stderr += str(exc.stderr or "")
        except FileNotFoundError as exc:
            worker_exit = 127
            stderr += f"openclaw command not found: {exc}"

        worker_transcript, worker_transcript_path = _load_transcript(worker_agent_id, worker_session, worker_started)
        _archive_transcript(worker_transcript_path, output_dir, f"{task.task_id}__{worker['id']}.jsonl")
        worker_results.append(
            {
                "agent_id": worker_agent_id,
                "worker_id": worker["id"],
                "note_path": note_path,
                "note_exists": (workspace / note_path).is_file(),
                "exit_code": worker_exit,
                "transcript_length": len(worker_transcript),
                "execution_time": time.time() - worker_started,
            }
        )

    final_prompt = f"""You are the lead agent. The fixed helper agents have written
compact evidence notes. Now synthesize the final deliverable for the original
task.

Original task:
{task.prompt}

Team policy:
{json.dumps(policy, indent=2, ensure_ascii=False)}

Worker note preview:
{_note_preview(workspace, policy, int(os.environ.get("PINCHBENCH_TEAM_NOTE_PREVIEW_CHARS", "6000")))}

Instructions:
- Use the notes as compressed evidence.
- Read `team_notes/team_policy.json` or individual `team_notes/*.md` files if needed.
- Create the exact final output file(s) requested by the original task.
- After writing the final deliverable, respond with a concise team summary:
  worker coverage, decisive evidence, and remaining gaps.
"""
    if remaining() > 5 and exit_code != 127:
        try:
            result = _run_agent_message(
                agent_id=final_agent_id,
                session_id=session_id,
                prompt=final_prompt,
                workspace=workspace,
                timeout=min(remaining(), max(40.0, timeout_seconds * 0.25)),
            )
            stdout += result.stdout
            stderr += result.stderr
            exit_code = result.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout += str(exc.stdout or "")
            stderr += str(exc.stderr or "")
        except FileNotFoundError as exc:
            exit_code = 127
            stderr += f"openclaw command not found: {exc}"

    final_transcript, final_transcript_path = _load_transcript(final_agent_id, session_id, started_at)
    _archive_transcript(final_transcript_path, output_dir, f"{task.task_id}.jsonl")
    usage = _extract_usage_from_transcript(final_transcript)

    status = "success"
    if timed_out:
        status = "timeout"
    if not final_transcript or not _has_successful_assistant_turn(final_transcript):
        status = "error"
    if exit_code not in (0, -1) and not timed_out:
        status = "error"
    if "openclaw command not found" in stderr:
        status = "error"

    result = {
        "agent_id": final_agent_id,
        "policy_agent_id": policy_agent_id,
        "final_agent_id": final_agent_id,
        "task_id": task.task_id,
        "status": status,
        "transcript": final_transcript,
        "usage": usage,
        "workspace": str(workspace),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "execution_time": time.time() - started_at,
        "stdout": stdout,
        "stderr": stderr,
        "team_policy": policy,
        "team_workers": worker_results,
    }
    if verbose:
        LOGGER.info("Team policy for %s:\n%s", task.task_id, json.dumps(policy, indent=2, ensure_ascii=False))
        LOGGER.info("Worker results for %s: %s", task.task_id, worker_results)
    return result


def grade_execution(task: Task, result: dict[str, Any], skill_dir: Path, judge_model: Optional[str]) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"task": task, "execution_result": result, "skill_dir": skill_dir}
    if judge_model:
        cfg = resolve_judge_backend_from_env(default_backend="api")
        kwargs.update(
            {
                "judge_model": judge_model,
                "judge_backend": cfg["judge_backend"],
                "judge_base_url": cfg["judge_base_url"],
                "judge_api_key": cfg["judge_api_key"],
            }
        )
    grade = grade_task(**kwargs)
    return {
        "runs": [grade.to_dict()],
        "mean": grade.score,
        "std": 0.0,
        "min": grade.score,
        "max": grade.score,
    }


def load_tasks(tasks_dir: Path, task_ids: list[str]) -> list[Task]:
    loader = TaskLoader(tasks_dir)
    out = []
    for task_id in task_ids:
        task_file = resolve_task_markdown_path(tasks_dir, task_id)
        if not task_file.exists():
            matches = list(tasks_dir.rglob(f"{task_id}.md"))
            if matches:
                task_file = matches[0]
        out.append(loader.load_task(task_file))
    return out


def load_policy_overrides(path: Optional[str]) -> dict[str, dict[str, Any]]:
    if not path:
        return {}
    data = json.loads(Path(path).read_text("utf-8"))
    if not isinstance(data, dict):
        raise ValueError("--policy-file must contain a JSON object")
    if isinstance(data.get("workers"), list):
        return {"*": data}
    tasks = data.get("tasks")
    if isinstance(tasks, dict):
        return {str(k): v for k, v in tasks.items() if isinstance(v, dict)}
    return {str(k): v for k, v in data.items() if isinstance(v, dict)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run standalone lead/team-agent PinchBench tasks.")
    parser.add_argument("--tasks-dir", default=str(REPO_ROOT / "pinchbench_tasks" / "meeting_analysis"))
    parser.add_argument("--skill-dir", default=str(REPO_ROOT))
    parser.add_argument("--task", action="append", dest="tasks", help="Task id. Repeatable. Defaults to val_5.")
    parser.add_argument("--policy-model", default=os.environ.get("PINCHBENCH_TEAM_POLICY_MODEL", os.environ.get("PINCHBENCH_TEAM_LEAD_MODEL", "Qwen3-4B-base")))
    parser.add_argument("--policy-base-url", default=os.environ.get("PINCHBENCH_TEAM_POLICY_BASE_URL", os.environ.get("PINCHBENCH_TEAM_LEAD_BASE_URL")))
    parser.add_argument("--policy-api-key", default=os.environ.get("PINCHBENCH_TEAM_POLICY_API_KEY", os.environ.get("PINCHBENCH_TEAM_LEAD_API_KEY", "dummy")))
    parser.add_argument("--final-model", default=os.environ.get("PINCHBENCH_TEAM_FINAL_MODEL", "Qwen3-4B-base"))
    parser.add_argument("--final-base-url", default=os.environ.get("PINCHBENCH_TEAM_FINAL_BASE_URL"))
    parser.add_argument("--final-api-key", default=os.environ.get("PINCHBENCH_TEAM_FINAL_API_KEY", "dummy"))
    parser.add_argument("--worker-model", default=os.environ.get("PINCHBENCH_TEAM_WORKER_MODEL", "Qwen3-4B-base"))
    parser.add_argument("--worker-base-url", default=os.environ.get("PINCHBENCH_TEAM_WORKER_BASE_URL"))
    parser.add_argument("--worker-api-key", default=os.environ.get("PINCHBENCH_TEAM_WORKER_API_KEY", "dummy"))
    parser.add_argument("--workers", type=int, default=int(os.environ.get("PINCHBENCH_TEAM_WORKERS", "3")))
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--run-id", default=time.strftime("team_%Y%m%d_%H%M%S"))
    parser.add_argument("--agent-prefix", default=os.environ.get("PINCHBENCH_TEAM_AGENT_PREFIX", "team-bench"))
    parser.add_argument("--namespace", default=os.environ.get("PINCHBENCH_TEAM_NAMESPACE"))
    parser.add_argument("--openclaw-home", default=os.environ.get("PINCHBENCH_TEAM_OPENCLAW_HOME"))
    parser.add_argument("--lock-dir", default=os.environ.get("PINCHBENCH_TEAM_LOCK_DIR", "/tmp/openclaw_team_locks"))
    parser.add_argument("--no-cleanup-agents", action="store_true")
    parser.add_argument("--policy-file", default=os.environ.get("PINCHBENCH_TEAM_POLICY_FILE"))
    parser.add_argument("--output-dir", default="/tmp/pinchbench_team")
    parser.add_argument("--timeout-multiplier", type=float, default=3.0)
    parser.add_argument("--judge", default=os.environ.get("PINCHBENCH_GRADE_JUDGE_MODEL", "deepseek-chat"))
    parser.add_argument("--no-grade", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO if args.verbose else logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")

    task_ids = args.tasks or VAL5_TASKS
    tasks = load_tasks(Path(args.tasks_dir), task_ids)
    policy_overrides = load_policy_overrides(args.policy_file)
    output_root = Path(args.output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    namespace = re.sub(r"[^A-Za-z0-9_.-]+", "-", args.namespace or args.agent_prefix).strip("-").lower()
    if not namespace:
        namespace = "team-bench"
    args.agent_prefix = namespace
    openclaw_home_parent = Path(args.openclaw_home).expanduser() if args.openclaw_home else output_root / "_openclaw_home" / args.run_id
    openclaw_store = _openclaw_store_from_home_parent(openclaw_home_parent)
    openclaw_store.mkdir(parents=True, exist_ok=True)
    removed_agents = 0
    env_updates = {
        "OPENCLAW_HOME": str(openclaw_home_parent),
        "PINCHBENCH_OPENCLAW_HOME": str(openclaw_store),
    }

    all_results = []
    grades: dict[str, Any] = {}
    with _openclaw_home_lock(Path(args.lock_dir), openclaw_home_parent) as lock_path, _temporary_env(env_updates):
        if not args.no_cleanup_agents:
            removed_agents = _cleanup_namespace_agents(openclaw_home_parent, namespace)
        print(f"[team] namespace={namespace} openclaw_home={openclaw_home_parent} lock={lock_path} cleaned_agents={removed_agents}")
        for run_idx in range(args.runs):
            for task in tasks:
                task_run_id = f"{args.run_id}-{run_idx + 1}"
                transcript_dir = output_root / f"{args.run_id}_transcripts"
                print(f"[team] task={task.task_id} run={run_idx + 1}/{args.runs}")
                result = run_team_task(
                    task=task,
                    run_id=task_run_id,
                    policy_model=args.policy_model,
                    policy_base_url=args.policy_base_url,
                    policy_api_key=args.policy_api_key,
                    final_model=args.final_model,
                    final_base_url=args.final_base_url,
                    final_api_key=args.final_api_key,
                    worker_model=args.worker_model,
                    worker_base_url=args.worker_base_url,
                    worker_api_key=args.worker_api_key,
                    agent_prefix=args.agent_prefix,
                    skill_dir=Path(args.skill_dir),
                    output_dir=transcript_dir,
                    workers=max(1, args.workers),
                    timeout_multiplier=args.timeout_multiplier,
                    verbose=args.verbose,
                    policy_override=policy_overrides.get(task.task_id) or policy_overrides.get("*"),
                )
                all_results.append(result)
                if not args.no_grade:
                    grades[task.task_id] = grade_execution(task, result, Path(args.skill_dir), args.judge)
                    score = grades[task.task_id]["mean"]
                    print(f"[team]   score={score:.3f} status={result['status']} workspace={result['workspace']}")
                else:
                    print(f"[team]   status={result['status']} workspace={result['workspace']}")

                result_json = {
                    "policy_model": args.policy_model,
                    "final_model": args.final_model,
                    "worker_model": args.worker_model,
                    "run_id": args.run_id,
                    "namespace": namespace,
                    "openclaw_home": str(openclaw_home_parent),
                    "openclaw_store": str(openclaw_store),
                    "cleaned_agents": removed_agents,
                    "tasks": [
                        {
                            "task_id": r["task_id"],
                            "status": r["status"],
                            "timed_out": r["timed_out"],
                            "execution_time": r["execution_time"],
                            "transcript_length": len(r["transcript"]),
                            "usage": r.get("usage", {}),
                            "workspace": r["workspace"],
                            "grading": grades.get(r["task_id"], {}),
                            "team_policy": r.get("team_policy"),
                            "team_workers": r.get("team_workers"),
                        }
                        for r in all_results
                    ],
                    "timestamp": time.time(),
                }
                (output_root / f"{args.run_id}_team.json").write_text(
                    json.dumps(result_json, indent=2, ensure_ascii=False),
                    "utf-8",
                )

    print(f"[team] wrote {output_root / f'{args.run_id}_team.json'}")
    print(f"[team] transcripts: {output_root / f'{args.run_id}_transcripts'}")


if __name__ == "__main__":
    main()
