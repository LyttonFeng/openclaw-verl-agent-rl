#!/usr/bin/env python3
"""Use a stronger coach model to improve a Qwen3-executable team policy."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.request
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text("utf-8"))


def _text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(x.get("text", "")) for x in content if isinstance(x, dict))
    return ""


def _transcript_summary(path: Path, max_chars: int) -> dict[str, Any]:
    if not path.is_file():
        return {"path": str(path), "missing": True}
    rows = []
    for line in path.read_text("utf-8", errors="replace").splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    items = []
    for row in rows:
        typ = row.get("type")
        msg = row.get("message") or {}
        role = msg.get("role")
        text = _text_from_content(msg.get("content")).strip()
        if not text and typ != "toolResult":
            continue
        items.append(
            {
                "type": typ,
                "role": role,
                "text": text[:max_chars],
            }
        )
    return {"path": str(path), "events": len(rows), "items": items[-8:]}


def _task_result_summary(result_path: Path, task_id: str) -> dict[str, Any]:
    data = _load_json(result_path)
    for row in data.get("tasks", []):
        if row.get("task_id") != task_id:
            continue
        grading = row.get("grading") or {}
        run = (grading.get("runs") or [{}])[0]
        return {
            "path": str(result_path),
            "task_id": task_id,
            "status": row.get("status"),
            "score": grading.get("mean"),
            "breakdown": run.get("breakdown"),
            "grader_notes": run.get("notes"),
            "team_policy": row.get("team_policy"),
            "team_workers": row.get("team_workers"),
        }
    return {"path": str(result_path), "task_id": task_id, "missing_task": True}


def _extract_json_object(text: str) -> dict[str, Any]:
    candidates = [text.strip()]
    if "```" in text:
        parts = text.split("```")
        candidates.extend(part.strip() for part in parts if "{" in part)
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        candidates.append(text[start : end + 1])
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise ValueError("coach response did not contain a JSON object")


def _call_openai_compatible(
    *,
    base_url: str,
    api_key: str,
    model: str,
    prompt: str,
    timeout: float,
) -> str:
    url = base_url.rstrip("/") + "/chat/completions"
    body = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": "You are a coach for a weak Qwen3 agent team. Return only valid JSON.",
            },
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.2,
    }
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    return data["choices"][0]["message"]["content"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Coach a team policy from failed Qwen3 trajectories.")
    parser.add_argument("--task", required=True)
    parser.add_argument("--task-prompt-file", required=True)
    parser.add_argument("--result", action="append", default=[], help="Existing team result JSON. Repeatable.")
    parser.add_argument("--transcript", action="append", default=[], help="Extra transcript JSONL to summarize. Repeatable.")
    parser.add_argument("--base-policy", help="Optional policy JSON to patch.")
    parser.add_argument("--output-policy", required=True)
    parser.add_argument("--output-report")
    parser.add_argument("--model", default=os.environ.get("PINCHBENCH_COACH_MODEL", "deepseek-v4-pro"))
    parser.add_argument("--base-url", default=os.environ.get("PINCHBENCH_COACH_BASE_URL", "https://api.deepseek.com/v1"))
    parser.add_argument("--api-key", default=os.environ.get("PINCHBENCH_COACH_API_KEY") or os.environ.get("DEEPSEEK_API_KEY"))
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-event-chars", type=int, default=1200)
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("Missing --api-key or DEEPSEEK_API_KEY")

    task_prompt = Path(args.task_prompt_file).read_text("utf-8", errors="replace")
    context = {
        "task_id": args.task,
        "task_prompt": task_prompt,
        "existing_results": [
            _task_result_summary(Path(path), args.task) for path in args.result
        ],
        "extra_transcripts": [
            _transcript_summary(Path(path), args.max_event_chars) for path in args.transcript
        ],
        "base_policy": _load_json(Path(args.base_policy)) if args.base_policy else None,
    }
    prompt = f"""
We are improving a Qwen3-4B agent team on one PinchBench task.

You must inspect the Qwen3 failure modes and produce a Qwen3-executable team
policy. Do not solve the task or include hidden-answer assumptions. The policy
must constrain workers to concrete evidence extraction that Qwen3 can execute.

Return ONLY JSON with this schema:
{{
  "diagnosis": ["specific Qwen3 limitations observed"],
  "policy": {{
    "strategy": "...",
    "transcript_files": ["..."],
    "expected_output_files": ["..."],
    "workers": [
      {{
        "id": "worker_1",
        "focus": "...",
        "transcript_files": ["..."],
        "read_offsets": "...",
        "note_path": "team_notes/worker_1.md"
      }}
    ]
  }},
  "final_agent_instructions": ["checks the final Qwen3 agent must perform"]
}}

Rules:
- Optimize for Qwen3 execution, not for a human expert.
- Prefer explicit evidence rows/tables over narrative summaries.
- Avoid asking workers to infer positions from affiliations.
- Include coverage/checker work if that is needed.
- Keep note outputs compact enough for a 64k final context.

Context:
{json.dumps(context, ensure_ascii=False, indent=2)}
"""
    response = _call_openai_compatible(
        base_url=args.base_url,
        api_key=args.api_key,
        model=args.model,
        prompt=prompt,
        timeout=args.timeout,
    )
    coached = _extract_json_object(response)
    policy = coached.get("policy")
    if not isinstance(policy, dict):
        raise SystemExit("Coach JSON missing policy object")
    output = {"tasks": {args.task: policy}}
    out_path = Path(args.output_policy)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", "utf-8")
    if args.output_report:
        report_path = Path(args.output_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(coached, indent=2, ensure_ascii=False) + "\n", "utf-8")
    print(out_path)


if __name__ == "__main__":
    main()
