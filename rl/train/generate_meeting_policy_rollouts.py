#!/usr/bin/env python3
"""Generate rollouts for meeting policy RL JSONL datasets.

This runner consumes records exported by ``scripts/rl_data/export_grpo_dataset.py``
instead of the original PinchBench split file. It is used for scaffolded
policy training data where reward comes from multi-teacher consensus gold.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "scripts" / "rl_data"))

from lib_agent import ensure_agent_exists, execute_openclaw_task  # noqa: E402
from lib_tasks import Task  # noqa: E402
from sanitize_scaffolded_transcript import sanitize_entries, write_transcript  # noqa: E402
from score_meeting_rollout import score as score_consensus_rollout  # noqa: E402


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("meeting_policy_rollouts")

_RUN_ID = f"policy_rollout_{int(time.time())}_{os.getpid()}"
_DEFAULT_AGENT_SUFFIX = _RUN_ID
_worker_agents: dict[int, str] = {}
_worker_lock = threading.Lock()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def row_task_id(row: dict[str, Any]) -> str:
    return row.get("extra_info", {}).get("task_id") or row.get("reward_model", {}).get("ground_truth")


def build_task(row: dict[str, Any], inject_scaffold: bool) -> Task:
    extra = row.get("extra_info", {})
    metadata = extra.get("metadata", {})
    prompt_messages = row.get("prompt") or []
    user_prompt = next((m.get("content", "") for m in prompt_messages if m.get("role") == "user"), "")
    if inject_scaffold and extra.get("training_scaffold"):
        user_prompt = f"{user_prompt}\n\n{extra['training_scaffold']}"
    return Task(
        task_id=row_task_id(row),
        name=metadata.get("task_name", row_task_id(row)),
        category="meeting",
        grading_type="policy_gold",
        timeout_seconds=int(metadata.get("timeout_seconds") or 180),
        workspace_files=extra.get("workspace_files", []),
        prompt=user_prompt,
        expected_behavior="",
        grading_criteria=[],
        automated_checks=None,
        llm_judge_rubric=None,
        grading_weights=metadata.get("grading_weights"),
        file_path=None,
        frontmatter={},
    )


def make_default_agent_suffix(output_dir: Path, shard_index: int) -> str:
    """Return a per-process/per-shard suffix to avoid cross-shard workspace reuse."""
    digest = hashlib.sha1(str(output_dir.resolve()).encode("utf-8")).hexdigest()[:8]
    return f"{_RUN_ID}_s{shard_index}_{digest}"


def ensure_worker_agent(model: str, base_url: str, worker_idx: int, output_dir: Path) -> str:
    with _worker_lock:
        if worker_idx in _worker_agents:
            return _worker_agents[worker_idx]
        safe_model = model.lower().replace("/", "-").replace(":", "-")
        suffix = os.environ.get("PINCHBENCH_AGENT_SUFFIX", _DEFAULT_AGENT_SUFFIX)
        agent_id = f"bench-policy-{safe_model}-w{worker_idx}-{suffix}"
        workspace = output_dir / "worker_workspaces" / f"worker_{worker_idx}"
        workspace.mkdir(parents=True, exist_ok=True)
        ensure_agent_exists(
            agent_id=agent_id,
            model_id=f"custom/{model}",
            workspace_dir=workspace,
            base_url=base_url,
            api_key=os.environ.get("PINCHBENCH_MODEL_API_KEY", "dummy"),
        )
        _worker_agents[worker_idx] = agent_id
        return agent_id


def snapshot_workspace(workspace_path: str, snapshot_dir: Path) -> None:
    import shutil

    snapshot_dir.mkdir(parents=True, exist_ok=True)
    ws = Path(workspace_path)
    if not ws.is_dir():
        return
    skip = {"BOOTSTRAP.md", "AGENTS.md", "IDENTITY.md", "HEARTBEAT.md", "USER.md", "TOOLS.md", "SOUL.md"}
    for item in ws.iterdir():
        if item.is_file() and item.name not in skip:
            shutil.copy2(item, snapshot_dir / item.name)


def write_raw_transcript(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def process_one(
    row: dict[str, Any],
    response_idx: int,
    worker_idx: int,
    args: argparse.Namespace,
) -> dict[str, Any]:
    output_dir = Path(args.output_dir)
    task = build_task(row, inject_scaffold=not args.no_scaffold)
    agent_id = ensure_worker_agent(args.model, args.vllm_base_url, worker_idx, output_dir)
    run_id = f"{task.task_id}_resp{response_idx}_{int(time.time() * 1000)}"

    result = execute_openclaw_task(
        task=task,
        agent_id=agent_id,
        model_id=f"custom/{args.model}",
        run_id=run_id,
        timeout_multiplier=args.timeout_multiplier,
        skill_dir=REPO_ROOT,
    )

    snapshot_dir = output_dir / "workspaces" / f"{task.task_id}_resp{response_idx}"
    snapshot_workspace(result.get("workspace", ""), snapshot_dir)

    raw_transcript = result.get("transcript", [])
    raw_path = output_dir / "transcripts_raw" / f"{task.task_id}_resp{response_idx}.jsonl"
    clean_path = output_dir / "transcripts" / f"{task.task_id}_resp{response_idx}.jsonl"
    write_raw_transcript(raw_path, raw_transcript)
    write_transcript(clean_path, sanitize_entries(raw_transcript))

    gold = row.get("reward_model", {}).get("gold", {})
    scoring = score_consensus_rollout(gold, clean_path, snapshot_dir)
    return {
        "task_id": task.task_id,
        "source_task_id": row.get("extra_info", {}).get("source_task_id"),
        "variant": row.get("extra_info", {}).get("variant"),
        "mix_bucket": row.get("extra_info", {}).get("mix_bucket"),
        "response_idx": response_idx,
        "score": float(scoring.get("score", 0.0)),
        "gold_recall": scoring.get("gold_recall", 0.0),
        "policy_score": scoring.get("policy_score", 0.0),
        "policy_features": scoring.get("policy_features", {}),
        "gold_total": scoring.get("gold_total", 0),
        "gold_hits": scoring.get("gold_hits", 0),
        "missed_claims": scoring.get("missed_claims", []),
        "workspace_path": str(snapshot_dir),
        "transcript_path": str(clean_path),
        "raw_transcript_path": str(raw_path),
        "execution_time": result.get("execution_time", 0),
        "timed_out": result.get("timed_out", False),
        "status": result.get("status", "unknown"),
        "prm_turn_scores": [],
    }


def shard_rows(rows: list[dict[str, Any]], shard_index: int, num_shards: int) -> list[dict[str, Any]]:
    if num_shards <= 1:
        return rows
    return [row for idx, row in enumerate(rows) if idx % num_shards == shard_index]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate meeting policy consensus-gold rollouts.")
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--vllm-base-url", required=True)
    parser.add_argument("--model", default="Qwen3-4B")
    parser.add_argument("--n-responses", type=int, default=2)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--timeout-multiplier", type=float, default=3.0)
    parser.add_argument("--no-scaffold", action="store_true")
    args = parser.parse_args()

    global _DEFAULT_AGENT_SUFFIX
    _DEFAULT_AGENT_SUFFIX = make_default_agent_suffix(Path(args.output_dir), args.shard_index)
    logger.info("Agent suffix: %s", os.environ.get("PINCHBENCH_AGENT_SUFFIX", _DEFAULT_AGENT_SUFFIX))

    rows = shard_rows(load_jsonl(Path(args.dataset)), args.shard_index, args.num_shards)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Loaded %d rows for shard %d/%d", len(rows), args.shard_index, args.num_shards)

    jobs = [(row, resp_idx) for row in rows for resp_idx in range(args.n_responses)]
    graded_path = output_dir / "graded_trajectories.jsonl"
    scores: list[float] = []
    write_lock = threading.Lock()
    total = 0

    def wrap(job):
        row, resp_idx = job
        tname = threading.current_thread().name
        try:
            worker_idx = int(tname.rsplit("_", 1)[-1])
        except ValueError:
            worker_idx = 0
        return process_one(row, resp_idx, worker_idx, args)

    with graded_path.open("w", encoding="utf-8") as out_f:
        with ThreadPoolExecutor(max_workers=args.num_workers, thread_name_prefix="policy_rollout") as ex:
            futures = [ex.submit(wrap, job) for job in jobs]
            for future in as_completed(futures):
                try:
                    rec = future.result()
                except Exception as exc:
                    logger.exception("Rollout failed: %s", exc)
                    continue
                with write_lock:
                    out_f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
                    out_f.flush()
                    scores.append(float(rec["score"]))
                    total += 1
                logger.info(
                    "[%d/%d] %s resp%d score=%.3f recall=%.3f policy=%.3f time=%.1fs",
                    total,
                    len(jobs),
                    rec["task_id"],
                    rec["response_idx"],
                    rec["score"],
                    rec["gold_recall"],
                    rec["policy_score"],
                    rec["execution_time"],
                )

    summary = {
        "dataset": args.dataset,
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "n_rows": len(rows),
        "n_responses": args.n_responses,
        "n_trajectories": total,
        "mean_score": sum(scores) / len(scores) if scores else 0.0,
        "min_score": min(scores) if scores else 0.0,
        "max_score": max(scores) if scores else 0.0,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Summary: %s", summary)


if __name__ == "__main__":
    main()
