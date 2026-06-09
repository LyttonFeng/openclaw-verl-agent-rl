#!/usr/bin/env python3
"""
Generate meeting_analysis rollouts via OpenClaw + external vLLM, then grade them.

For each training task:
  1. Seed workspace with transcript
  2. Run OpenClaw agent (→ external vLLM)
  3. Collect transcript + workspace output
  4. Run trajectory diagnostics
  5. Grade (automated + LLM judge) if not fatal
  6. Output graded trajectories as JSONL

Output: graded_trajectories.jsonl with schema:
  {"task_id", "prompt", "response", "score", "diagnostics", "grading", "round_idx"}
"""

import argparse
import difflib
import json
import logging
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))                       # for `import agent_loop.*`
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "agent_loop"))
sys.path.insert(0, str(REPO_ROOT / "train"))
sys.path.insert(0, str(REPO_ROOT / "rewards"))           # for `import meeting_reward`

from lib_tasks import TaskLoader
from agent_loop.diagnostics import diagnose
from meeting_reward import compute_score

from agent_loop.diagnostics.plugins.meeting_analysis import (
    EXPECTED_INPUT_FILES,
    EXPECTED_OUTPUT_FILE,
)


# Per-worker agent setup. Each worker gets its OWN agent_id + workspace_dir
# so that concurrent rollouts cannot stomp on each other's workspace files.
# This implements the rule recorded in feedback_workspace_isolation.md:
#   "并发时每个 worker 独立 agent+workspace".
_ROLLOUT_RUN_ID = f"rollout_{int(time.time())}"
_worker_agents: dict[int, str] = {}  # worker_idx → agent_id
_worker_lock = threading.Lock()


def _ensure_agent(model: str, vllm_base_url: str, worker_idx: int) -> str:
    """Ensure the benchmark agent for this worker exists and is configured for
    vLLM endpoint. Each worker owns a unique agent_id + workspace_dir."""
    with _worker_lock:
        if worker_idx in _worker_agents:
            return _worker_agents[worker_idx]

        from lib_agent import ensure_agent_exists
        # Suffix agent_id with worker index so concurrent OpenClaw sessions
        # don't share state. Lowercase + safe chars only.
        base = f"bench-custom-{model.lower().replace('/', '-')}"
        agent_id = f"{base}-w{worker_idx}"

        # Each worker gets its own workspace path under the same run_id.
        workspace_dir = Path(
            f"/tmp/pinchbench/{_ROLLOUT_RUN_ID}/worker_{worker_idx}/agent_workspace"
        )
        workspace_dir.mkdir(parents=True, exist_ok=True)
        ensure_agent_exists(
            agent_id=agent_id,
            model_id=f"custom/{model}",
            workspace_dir=workspace_dir,
            base_url=vllm_base_url,
            api_key="dummy",
        )
        _worker_agents[worker_idx] = agent_id
        return agent_id


def run_single_rollout(
    task,
    assets_dir: Path,
    vllm_base_url: str,
    model: str,
    timeout: int,
    worker_idx: int = 0,
) -> dict:
    """
    Run one OpenClaw rollout for a meeting_analysis task.

    Uses execute_openclaw_task (same as benchmark.py) for consistency.
    Returns dict with: transcript, workspace_path, execution_time, timed_out, status
    """
    from lib_agent import execute_openclaw_task

    agent_id = _ensure_agent(model, vllm_base_url, worker_idx)
    # Per-rollout run_id so transcript file names don't collide between
    # concurrent workers handling the same task.
    run_id = f"{task.task_id}_w{worker_idx}_{int(time.time() * 1000)}"

    try:
        result = execute_openclaw_task(
            task=task,
            agent_id=agent_id,
            model_id=f"custom/{model}",
            run_id=run_id,
            timeout_multiplier=2.0,
            skill_dir=assets_dir.parent,  # repo root
        )
        return {
            "transcript": result.get("transcript", []),
            "workspace_path": result.get("workspace", ""),
            "execution_time": result.get("execution_time", 0),
            "timed_out": result.get("timed_out", False),
            "status": result.get("status", "unknown"),
        }
    except Exception as e:
        logger.error(f"Rollout failed for {task.task_id}: {e}")
        return {
            "transcript": [],
            "workspace_path": "",
            "execution_time": 0,
            "timed_out": False,
            "status": "error",
            "error": str(e),
        }


def grade_rollout(task, rollout_result: dict) -> dict:
    """Grade a rollout using meeting_reward.compute_score."""
    task_id = task.task_id
    workspace_path = rollout_result["workspace_path"]
    transcript = rollout_result.get("transcript", [])

    # Run diagnostics first
    diag = diagnose(
        trajectory=transcript,
        workspace_path=workspace_path,
        task_id=task_id,
        execution_time=rollout_result.get("execution_time", 0),
        timed_out=rollout_result.get("timed_out", False),
    )

    if diag.fatal:
        # Skip expensive judge call
        return {
            "score": 0.0,
            "automated_score": 0.0,
            "judge_score": 0.0,
            "diagnostics": diag.to_dict(),
            "skipped_judge": True,
        }

    # Run full grading
    result = compute_score(
        data_source="pinchbench/meeting_analysis",
        solution_str="",
        ground_truth=task_id,
        extra_info={
            "task_id": task_id,
            "workspace_path": workspace_path,
            "transcript": transcript,
            "grading_weights": task.grading_weights or {"automated": 0.5, "llm_judge": 0.5},
        },
    )

    result["diagnostics"] = diag.to_dict()
    result["skipped_judge"] = False
    return result


def _snapshot_workspace(workspace_path: str, snapshot_dir: str) -> None:
    """Copy workspace output files to a snapshot directory.

    Only copies .md/.txt/.json files (not OpenClaw internals).
    This must be called IMMEDIATELY after each rollout, before
    the next task overwrites the shared workspace.
    """
    import shutil

    ws = Path(workspace_path)
    snap = Path(snapshot_dir)
    if not ws.is_dir():
        return

    skip_names = {
        "BOOTSTRAP.md", "AGENTS.md", "IDENTITY.md", "HEARTBEAT.md",
        "USER.md", "TOOLS.md", "SOUL.md",
    }
    skip_dirs = {".openclaw"}

    for item in ws.iterdir():
        if item.name in skip_dirs:
            continue
        if item.is_file() and item.name not in skip_names:
            shutil.copy2(item, snap / item.name)


BAD_ACCESS_PHRASES = (
    "unable to access",
    "cannot access",
    "can't access",
    "not provided",
    "not available",
    "transcript was truncated",
    "transcript is truncated",
    "based on the available information",
    "based on limited information",
    "hypothetical",
    "assuming that",
    "i do not have access",
)


READ_TOOL_RE = re.compile(r"\b(read|read_file|grep|rg|search|find)\b", re.I)
WRITE_TOOL_RE = re.compile(r"\b(write|write_file|edit)\b", re.I)


def _read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""


def _workspace_source_text(workspace: Path) -> str:
    chunks: list[str] = []
    for name in EXPECTED_INPUT_FILES:
        p = workspace / name
        if p.is_file():
            chunks.append(_read_text(p))
    return "\n".join(chunks)


def _workspace_output_text(task_id: str, workspace: Path) -> tuple[str, bool, int]:
    expected = EXPECTED_OUTPUT_FILE.get(task_id, "")
    expected_exists = False
    chunks: list[str] = []

    if expected:
        p = workspace / expected
        expected_exists = p.is_file()
        if expected_exists:
            chunks.append(_read_text(p))

    for p in sorted(workspace.iterdir()) if workspace.is_dir() else []:
        if not p.is_file() or p.name in EXPECTED_INPUT_FILES:
            continue
        if expected and p.name == expected:
            continue
        if p.suffix.lower() in {".md", ".txt", ".json", ".csv"}:
            chunks.append(_read_text(p))

    output = "\n".join(chunks)
    return output, expected_exists, len(output)


def _trace_features(transcript: list) -> dict:
    tool_names: list[str] = []
    tool_success = 0
    tool_errors = 0
    for row in transcript or []:
        if not isinstance(row, dict) or row.get("type") != "message":
            continue
        msg = row.get("message") or {}
        role = msg.get("role")
        content = msg.get("content") or []
        if role == "assistant" and isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "toolCall":
                    tool_names.append(str(item.get("name") or ""))
        elif role == "toolResult":
            if msg.get("isError"):
                tool_errors += 1
            else:
                tool_success += 1

    joined = " ".join(tool_names)
    return {
        "read_calls": len(READ_TOOL_RE.findall(joined)),
        "write_calls": len(WRITE_TOOL_RE.findall(joined)),
        "tool_success": tool_success,
        "tool_errors": tool_errors,
    }


def _normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _extract_quotes(text: str) -> list[str]:
    quotes: list[str] = []
    patterns = [
        r'"([^"\n]{18,240})"',
        r"'([^'\n]{18,240})'",
        r"`([^`\n]{18,240})`",
        r"[“”]([^“”\n]{18,240})[“”]",
    ]
    for pattern in patterns:
        quotes.extend(re.findall(pattern, text))
    cleaned: list[str] = []
    for q in quotes:
        q = re.sub(r"\s+", " ", q).strip()
        if len(q.split()) >= 4:
            cleaned.append(q)
    return cleaned[:20]


def _quote_in_source(quote: str, source: str, threshold: float = 0.86) -> bool:
    q = _normalize_space(quote)
    s = _normalize_space(source)
    if not q or not s:
        return False
    if q in s:
        return True
    if len(q) < 24:
        return False

    q_len = len(q)
    step = max(24, q_len // 3)
    for start in range(0, max(len(s) - q_len + step, 1), step):
        window = s[start : start + q_len + 80]
        if difflib.SequenceMatcher(None, q, window).ratio() >= threshold:
            return True
    return False


def _policy_features(task_id: str, workspace_path: str, transcript: list, diagnostics: dict) -> dict:
    workspace = Path(workspace_path)
    output, expected_exists, output_chars = _workspace_output_text(task_id, workspace)
    source = _workspace_source_text(workspace)
    trace = _trace_features(transcript)

    output_lower = output.lower()
    quotes = _extract_quotes(output)
    valid_quotes = sum(1 for q in quotes if _quote_in_source(q, source))
    quote_ratio = valid_quotes / len(quotes) if quotes else None
    failure_tags = set(diagnostics.get("failure_tags") or [])

    return {
        "expected_output_file": EXPECTED_OUTPUT_FILE.get(task_id, ""),
        "expected_output_exists": expected_exists,
        "output_chars": output_chars,
        "writes_output": 1.0 if expected_exists or output_chars > 0 else 0.0,
        "multi_read": 1.0 if trace["read_calls"] > 0 or "transcript_not_read" not in failure_tags else 0.0,
        "output_quality": 0.0 if "output_too_short" in failure_tags or output_chars < 50 else 1.0,
        "read_calls": trace["read_calls"],
        "write_calls": trace["write_calls"],
        "tool_success": trace["tool_success"],
        "tool_errors": trace["tool_errors"],
        "bad_access_phrase": any(phrase in output_lower for phrase in BAD_ACCESS_PHRASES),
        "quotes": len(quotes),
        "valid_quotes": valid_quotes,
        "quote_verified_ratio": quote_ratio,
    }


def _save_transcript(transcript: list, path: str) -> None:
    """Save raw transcript entries as JSONL for training."""
    with open(path, "w") as f:
        for entry in transcript:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate meeting analysis rollouts")
    parser.add_argument("--split-file", required=True, help="Path to meeting_analysis_split.json")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--tasks-dir", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--vllm-base-url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--n-responses", type=int, default=4)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--judge-model", default="openai/qwen-plus")
    parser.add_argument("--timeout", type=int, default=300)
    parser.add_argument("--num-workers", type=int, default=1,
                        help="Concurrent rollouts (each worker owns an isolated "
                             "agent_id + workspace). Default 1 = serial.")
    args = parser.parse_args()

    # Set judge model
    os.environ["MEETING_JUDGE_MODEL"] = args.judge_model
    os.environ["MEETING_TASKS_DIR"] = str(Path(args.tasks_dir).resolve())

    # Load split
    with open(args.split_file) as f:
        split_data = json.load(f)
    task_ids = split_data[args.split]

    # Load tasks
    tasks_dir = Path(args.tasks_dir)
    loader = TaskLoader(tasks_dir)
    assets_dir = Path(args.assets_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    graded_file = output_dir / "graded_trajectories.jsonl"

    logger.info(f"Generating rollouts: {len(task_ids)} tasks × {args.n_responses} responses")
    logger.info(f"Model: {args.model} @ {args.vllm_base_url}")
    logger.info(f"Concurrency: {args.num_workers} worker(s)")

    # Build (task, resp_idx) work list
    jobs: list[tuple] = []
    for task_id in task_ids:
        task_file = tasks_dir / f"{task_id}.md"
        if not task_file.exists():
            logger.warning(f"Task file not found: {task_file}, skipping")
            continue
        task = loader.load_task(task_file)
        for resp_idx in range(args.n_responses):
            jobs.append((task_id, task, resp_idx))

    total = 0
    fatal_count = 0
    scores: list[float] = []
    write_lock = threading.Lock()

    (output_dir / "transcripts").mkdir(parents=True, exist_ok=True)

    def _process_one(task_id: str, task, resp_idx: int, worker_idx: int) -> dict:
        """Run + grade + persist one rollout. Returns the record (also written to file)."""
        rollout = run_single_rollout(
            task=task,
            assets_dir=assets_dir,
            vllm_base_url=args.vllm_base_url,
            model=args.model,
            timeout=args.timeout,
            worker_idx=worker_idx,
        )

        # Snapshot workspace IMMEDIATELY before the worker's next task wipes it.
        # Each worker owns its own workspace path so concurrent workers don't
        # stomp on each other; but we still snapshot to free that workspace
        # for the next task this same worker handles.
        snapshot_dir = output_dir / "workspaces" / f"{task_id}_resp{resp_idx}"
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        _snapshot_workspace(rollout.get("workspace_path", ""), str(snapshot_dir))
        rollout["workspace_path"] = str(snapshot_dir)

        grading = grade_rollout(task, rollout)
        score = grading.get("score", 0.0)
        diagnostics = grading.get("diagnostics", {})
        is_fatal = diagnostics.get("fatal", False)
        policy_features = _policy_features(
            task_id=task_id,
            workspace_path=str(snapshot_dir),
            transcript=rollout.get("transcript", []),
            diagnostics=diagnostics,
        )

        transcript_save_path = str(
            output_dir / "transcripts" / f"{task_id}_resp{resp_idx}.jsonl"
        )
        _save_transcript(rollout.get("transcript", []), transcript_save_path)

        record = {
            "task_id": task_id,
            "prompt": task.prompt,
            "response_idx": resp_idx,
            "score": score,
            "diagnostics": diagnostics,
            "policy_features": policy_features,
            "automated_score": grading.get("automated_score", 0.0),
            "judge_score": grading.get("judge_score", 0.0),
            "workspace_path": str(snapshot_dir),
            "transcript_path": transcript_save_path,
            "execution_time": rollout.get("execution_time", 0),
            "timed_out": rollout.get("timed_out", False),
            "status": rollout.get("status", "unknown"),
            "skipped_judge": grading.get("skipped_judge", False),
        }
        record["_is_fatal"] = is_fatal  # internal, stripped before write
        record["_worker_idx"] = worker_idx
        return record

    with open(graded_file, "w") as out_f:
        if args.num_workers <= 1:
            # Serial path (preserve original behaviour for sanity / debugging)
            for (task_id, task, resp_idx) in jobs:
                logger.info(f"  {task_id} resp{resp_idx}: starting...")
                record = _process_one(task_id, task, resp_idx, worker_idx=0)
                is_fatal = record.pop("_is_fatal", False)
                record.pop("_worker_idx", None)
                out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
                out_f.flush()
                scores.append(record["score"])
                total += 1
                fatal_count += int(is_fatal)
                logger.info(
                    f"    {task_id} resp{resp_idx} score={record['score']:.3f} "
                    f"fatal={is_fatal} time={record['execution_time']:.1f}s"
                )
        else:
            # Parallel path. Each future is a (task, resp_idx) pair; the
            # ThreadPoolExecutor maps each future to a worker thread, and
            # _ensure_agent caches one (agent_id, workspace_dir) pair per
            # worker_idx so the agents are reused across futures by the
            # same worker.
            #
            # We assign worker_idx = thread index by hashing thread name —
            # ThreadPoolExecutor names threads predictably as
            # "ThreadPoolExecutor-N_M", we use M.
            def _wrap(job):
                task_id, task, resp_idx = job
                tname = threading.current_thread().name
                # parse trailing integer
                try:
                    widx = int(tname.rsplit("_", 1)[-1])
                except ValueError:
                    widx = 0
                logger.info(f"  [w{widx}] {task_id} resp{resp_idx}: starting...")
                rec = _process_one(task_id, task, resp_idx, worker_idx=widx)
                return rec

            with ThreadPoolExecutor(
                max_workers=args.num_workers,
                thread_name_prefix="rollout",
            ) as ex:
                futures = [ex.submit(_wrap, job) for job in jobs]
                for fut in as_completed(futures):
                    try:
                        rec = fut.result()
                    except Exception as e:
                        logger.error(f"Rollout future failed: {e}")
                        continue
                    is_fatal = rec.pop("_is_fatal", False)
                    widx = rec.pop("_worker_idx", -1)
                    with write_lock:
                        out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                        out_f.flush()
                        scores.append(rec["score"])
                        total += 1
                        fatal_count += int(is_fatal)
                    logger.info(
                        f"    [w{widx}] {rec['task_id']} resp{rec['response_idx']} "
                        f"score={rec['score']:.3f} fatal={is_fatal} "
                        f"time={rec['execution_time']:.1f}s "
                        f"({total}/{len(jobs)})"
                    )

    # Summary
    mean_score = sum(scores) / len(scores) if scores else 0.0
    logger.info("")
    logger.info(f"═══ Generation Summary ═��═")
    logger.info(f"  Total trajectories: {total}")
    logger.info(f"  Fatal (reward=0):   {fatal_count} ({fatal_count/total*100:.0f}%)" if total else "")
    logger.info(f"  Mean score:         {mean_score:.4f}")
    logger.info(f"  Output:             {graded_file}")


if __name__ == "__main__":
    main()
