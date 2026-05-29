#!/usr/bin/env python3
"""Run candidate-generation + selector/merger experiment for one task.

This tests a simpler swarm policy:
  1. K independent LoRA agents each solve the same task.
  2. A small base model sees only the candidate reports and writes the final file.
  3. The final file is graded with the normal meeting_analysis terminal reward.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "agent_loop"))
sys.path.insert(0, str(REPO_ROOT / "rl" / "train"))
sys.path.insert(0, str(REPO_ROOT / "rewards"))

from lib_tasks import TaskLoader  # noqa: E402

import generate_meeting_rollouts as gmr  # noqa: E402


def chat_completion(base_url: str, model: str, messages: list[dict], *, temperature: float, max_tokens: int) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "top_p": 0.9,
        "max_tokens": max_tokens,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=data,
        headers={"Content-Type": "application/json", "Authorization": "Bearer dummy"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=600) as resp:
        obj = json.loads(resp.read().decode("utf-8"))
    return obj["choices"][0]["message"].get("content", "")


def strip_markdown_fence(text: str) -> str:
    text = text.strip()
    m = re.match(r"^```(?:markdown|md)?\s*(.*?)\s*```$", text, flags=re.DOTALL | re.IGNORECASE)
    return m.group(1).strip() if m else text


def read_expected_output(workspace: Path, filename: str) -> str:
    p = workspace / filename
    if p.exists():
        return p.read_text(encoding="utf-8", errors="replace")
    for f in workspace.iterdir() if workspace.exists() else []:
        if f.name.lower() == filename.lower():
            return f.read_text(encoding="utf-8", errors="replace")
    return ""


def run_candidate(
    task,
    assets_dir: Path,
    base_url: str,
    model: str,
    worker_idx: int,
    timeout: int,
    out_dir: Path,
    output_file: str,
) -> dict:
    rollout = gmr.run_single_rollout(
        task=task,
        assets_dir=assets_dir,
        vllm_base_url=base_url,
        model=model,
        timeout=timeout,
        worker_idx=worker_idx,
    )
    snap = out_dir / f"candidate_{worker_idx}"
    snap.mkdir(parents=True, exist_ok=True)
    gmr._snapshot_workspace(rollout.get("workspace_path", ""), str(snap))
    rollout["workspace_path"] = str(snap)
    grading = gmr.grade_rollout(task, rollout)
    transcript_path = out_dir / f"candidate_{worker_idx}.jsonl"
    gmr._save_transcript(rollout.get("transcript", []), str(transcript_path))
    output_text = read_expected_output(snap, output_file)
    return {
        "candidate_id": worker_idx,
        "base_url": base_url,
        "model": model,
        "score": grading.get("score", 0.0),
        "diagnostics": grading.get("diagnostics", {}),
        "workspace_path": str(snap),
        "transcript_path": str(transcript_path),
        "output_text": output_text,
        "output_chars": len(output_text),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", default="task_meeting_council_votes")
    parser.add_argument("--tasks-dir", required=True)
    parser.add_argument("--assets-dir", required=True)
    parser.add_argument("--candidate-base-urls", required=True, help="Comma-separated LoRA vLLM URLs")
    parser.add_argument("--candidate-model", default="qwen3-r08-lora")
    parser.add_argument("--lead-base-url", default="http://localhost:8767/v1")
    parser.add_argument("--lead-model", default="qwen3-base")
    parser.add_argument("--output-file", default="", help="Expected final deliverable filename")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    run_id = f"candidate_merge_{args.task_id}_{int(time.time())}_{os.getpid()}"
    openclaw_home = Path("/tmp/openclaw_home") / run_id
    run_root = Path("/tmp/pinchbench") / run_id
    openclaw_home.mkdir(parents=True, exist_ok=True)
    run_root.mkdir(parents=True, exist_ok=True)
    os.environ["OPENCLAW_HOME"] = str(openclaw_home)
    os.environ["PINCHBENCH_OPENCLAW_HOME"] = str(openclaw_home / ".openclaw")
    os.environ["PINCHBENCH_RUN_ROOT"] = str(run_root)
    os.environ["PINCHBENCH_AGENT_SUFFIX"] = run_id
    os.environ["PINCHBENCH_FORCE_LOCAL_OPENCLAW"] = "1"
    for k in list(os.environ):
        if k.startswith("PINCHBENCH_REMOTE_OPENCLAW_"):
            os.environ.pop(k, None)

    loader = TaskLoader(Path(args.tasks_dir))
    task = loader.load_task(Path(args.tasks_dir) / f"{args.task_id}.md")
    assets_dir = Path(args.assets_dir)
    output_file = args.output_file
    if not output_file:
        output_file = {
            "task_meeting_council_votes": "votes_report.md",
            "task_meeting_advisory_stakeholders": "stakeholder_analysis.md",
            "task_meeting_gov_speaker_summary": "speaker_summary.md",
            "task_meeting_tech_action_items": "action_items.md",
            "task_meeting_sentiment_analysis": "sentiment_analysis.md",
        }.get(args.task_id, "output.md")

    candidate_urls = [u.strip() for u in args.candidate_base_urls.split(",") if u.strip()]
    print(f"Running {len(candidate_urls)} candidates: {candidate_urls}", flush=True)

    candidates: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(candidate_urls)) as ex:
        futures = [
            ex.submit(run_candidate, task, assets_dir, url, args.candidate_model, i, args.timeout, out_dir, output_file)
            for i, url in enumerate(candidate_urls)
        ]
        for fut in as_completed(futures):
            c = fut.result()
            candidates.append(c)
            print(
                f"candidate_{c['candidate_id']} score={c['score']:.4f} chars={c['output_chars']} "
                f"fatal={c.get('diagnostics', {}).get('fatal')}",
                flush=True,
            )
    candidates.sort(key=lambda x: x["candidate_id"])

    candidate_block = []
    for c in candidates:
        text = c["output_text"] or f"(candidate did not write {output_file})"
        candidate_block.append(
            f"## Candidate {c['candidate_id']}\n"
            f"Observed candidate score: {c['score']:.4f}\n"
            f"Output:\n{text[:20000]}"
        )

    system = (
        "You are a selector and merger. You do not have tools. "
        "Your job is to write the final required report from candidate reports only. "
        "Prefer specific, evidence-backed details. If candidates conflict, keep the more precise and consistent claim. "
        f"Do not invent facts. Output only the final Markdown content for {output_file}."
    )
    user = (
        "Original task:\n"
        f"{task.prompt}\n\n"
        "You are given independent candidate reports generated by other agents.\n"
        "Merge the best supported facts into one final report. Follow the original schema exactly, including summary count.\n\n"
        + "\n\n---\n\n".join(candidate_block)
    )
    merged = strip_markdown_fence(
        chat_completion(
            args.lead_base_url,
            args.lead_model,
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.2,
            max_tokens=8192,
        )
    )

    merge_ws = out_dir / "merged_workspace"
    merge_ws.mkdir(parents=True, exist_ok=True)
    (merge_ws / output_file).write_text(merged, encoding="utf-8")
    fake_transcript = [
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [{"type": "toolCall", "name": "read", "arguments": {"path": "transcript.md"}}],
            },
        },
        {
            "type": "message",
            "message": {
                "role": "assistant",
                "content": [
                    {
                        "type": "toolCall",
                        "name": "write",
                        "arguments": {"path": output_file, "content": merged},
                    }
                ],
            },
        },
    ]
    merged_rollout = {
        "workspace_path": str(merge_ws),
        "transcript": fake_transcript,
        "execution_time": 0,
        "timed_out": False,
        "status": "success",
    }
    merged_grade = gmr.grade_rollout(task, merged_rollout)

    record = {
        "task_id": args.task_id,
        "mode": "k_candidate_selector_merger",
        "candidate_model": args.candidate_model,
        "candidate_base_urls": candidate_urls,
        "lead_model": args.lead_model,
        "lead_base_url": args.lead_base_url,
        "output_file": output_file,
        "openclaw_home": str(openclaw_home),
        "run_root": str(run_root),
        "candidate_scores": [c["score"] for c in candidates],
        "merged_score": merged_grade.get("score", 0.0),
        "merged_grading": merged_grade,
        "candidates": [{k: v for k, v in c.items() if k != "output_text"} for c in candidates],
        "merged_workspace": str(merge_ws),
        "created_at": int(time.time()),
    }
    (out_dir / "result.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(record, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
