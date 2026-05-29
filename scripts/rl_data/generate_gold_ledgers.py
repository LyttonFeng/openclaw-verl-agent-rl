#!/usr/bin/env python3
"""Generate and cross-check teacher gold ledgers for meeting_analysis RL tasks."""

from __future__ import annotations

import argparse
import http.client
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from common import (
    DEFAULT_ASSETS_DIR,
    REPO_ROOT,
    jaccard,
    read_jsonl,
    stable_slug,
    write_jsonl,
)


DEFAULT_TEACHERS = [
    "deepseek=deepseek-chat@https://api.deepseek.com/v1#DEEPSEEK_API_KEY",
    "qwen=qwen3.7-max@https://dashscope.aliyuncs.com/compatible-mode/v1#DASHSCOPE_API_KEY",
    "glm=glm-4-plus@https://open.bigmodel.cn/api/paas/v4#GLM_API_KEY",
]


def load_env_file_once(path: Path) -> None:
    if not path.exists():
        return
    try:
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[len("export "):].strip()
            if "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip().strip("'\"")
    except OSError:
        return


def load_default_env_files() -> None:
    candidates = [
        Path(os.environ.get("PINCHBENCH_ENV_FILE", "")) if os.environ.get("PINCHBENCH_ENV_FILE") else None,
        Path("/Users/lytton/.pinchbench_env"),
        Path("/root/.pinchbench_env"),
        Path.home() / ".pinchbench_env",
    ]
    for path in candidates:
        if path is not None:
            load_env_file_once(path.expanduser())


@dataclass(frozen=True)
class TeacherSpec:
    name: str
    model: str
    base_url: str
    api_key_env: str


def parse_teacher(raw: str) -> TeacherSpec:
    # name=model@base_url#API_KEY_ENV
    if "=" not in raw or "@" not in raw:
        raise ValueError(f"Bad teacher spec: {raw}")
    name, rest = raw.split("=", 1)
    model, rest = rest.split("@", 1)
    if "#" in rest:
        base_url, api_key_env = rest.rsplit("#", 1)
    else:
        base_url, api_key_env = rest, f"{name.upper()}_API_KEY"
    return TeacherSpec(name=name, model=model, base_url=base_url.rstrip("/"), api_key_env=api_key_env)


def transcript_text(record: dict[str, Any], assets_dir: Path, max_chars: int) -> str:
    chunks = []
    for source in record.get("transcript_sources", []):
        path = assets_dir / source
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        chunks.append(f"# {source}\n{text}")
    joined = "\n\n".join(chunks)
    if len(joined) <= max_chars:
        return joined
    head = joined[: max_chars // 2]
    tail = joined[-max_chars // 2 :]
    return f"{head}\n\n[... transcript middle omitted for teacher prompt budget ...]\n\n{tail}"


def gold_prompt(record: dict[str, Any], assets_dir: Path, max_transcript_chars: int) -> list[dict[str, str]]:
    user_prompt = next((m["content"] for m in record["prompt"] if m["role"] == "user"), "")
    transcript = transcript_text(record, assets_dir, max_transcript_chars)
    criteria = "\n".join(f"- {c}" for c in record.get("grading_criteria", []))
    expected_files = ", ".join(record.get("expected_output_files", [])) or "the requested output file"
    content = f"""You are building a gold ledger for an agent RL reward model.

Task id: {record['id']}
Source task id: {record['source_task_id']}
Target capabilities: {', '.join(record.get('target_capabilities', []))}
Expected output file(s): {expected_files}

User task:
{user_prompt}

Grading criteria:
{criteria}

Transcript:
{transcript}

Return strict JSON only. Do not include markdown.
Schema:
{{
  "task_id": "{record['id']}",
  "answer_ledger": [
    {{
      "id": "short_snake_case_id",
      "claim": "atomic fact or required output element",
      "evidence": "short source quote or meeting location cue",
      "required": true,
      "capability": "one target capability this item trains",
      "confidence": 0.0
    }}
  ],
  "output_requirements": ["format/file/coverage requirements"],
  "failure_modes": ["likely model failure this gold item prevents"],
  "verification_checks": ["checks the agent should perform before final answer"]
}}
"""
    return [
        {"role": "system", "content": "You produce auditable JSON gold ledgers for long-context meeting-analysis tasks."},
        {"role": "user", "content": content},
    ]


def call_openai_compatible(teacher: TeacherSpec, messages: list[dict[str, str]], timeout: float) -> dict[str, Any]:
    api_key = os.environ.get(teacher.api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing API key env {teacher.api_key_env} for teacher {teacher.name}")
    is_qwen = "dashscope.aliyuncs.com" in teacher.base_url or teacher.model.lower().startswith("qwen")
    body = {
        "model": teacher.model,
        "messages": messages,
        "temperature": 0.1,
    }
    if is_qwen:
        body.update(
            {
                "stream": True,
                "top_p": 0.8,
                "temperature": 0.2,
                "result_format": "message",
                "enable_thinking": True,
                "thinking_budget": int(os.environ.get("QWEN_THINKING_BUDGET", "4000")),
            }
        )
    else:
        body["response_format"] = {"type": "json_object"}
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        f"{teacher.base_url}/chat/completions",
        data=data,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if is_qwen:
            return parse_streamed_json_response(resp)
        payload = json.loads(resp.read().decode("utf-8"))
    content = payload["choices"][0]["message"]["content"]
    return parse_json_content(content)


def parse_json_content(content: str) -> dict[str, Any]:
    content = content.strip()
    if content.startswith("```"):
        content = re.sub(r"^```(?:json)?\s*", "", content)
        content = re.sub(r"\s*```$", "", content)
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        start = content.find("{")
        end = content.rfind("}")
        if start >= 0 and end > start:
            return json.loads(content[start : end + 1])
        raise


def parse_streamed_json_response(resp: Any) -> dict[str, Any]:
    content_parts: list[str] = []
    for raw_line in resp:
        line = raw_line.decode("utf-8", errors="replace").strip()
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload == "[DONE]":
            break
        try:
            chunk = json.loads(payload)
        except json.JSONDecodeError:
            continue
        choices = chunk.get("choices") or []
        if not choices:
            continue
        delta = choices[0].get("delta") or {}
        # DashScope exposes reasoning_content separately; it is useful for the
        # teacher but should not be parsed as answer JSON.
        content = delta.get("content")
        if content:
            content_parts.append(content)
    if not content_parts:
        raise RuntimeError("streamed teacher response contained no content chunks")
    return parse_json_content("".join(content_parts))


def cluster_ledgers(teacher_outputs: list[dict[str, Any]], min_agree: int) -> dict[str, Any]:
    clusters: list[dict[str, Any]] = []
    for output in teacher_outputs:
        teacher = output["teacher"]
        for item in output.get("gold", {}).get("answer_ledger", []) or []:
            claim = str(item.get("claim", "")).strip()
            if not claim:
                continue
            best = None
            best_score = 0.0
            for cluster in clusters:
                score = max(jaccard(claim, existing["claim"]) for existing in cluster["items"])
                if score > best_score:
                    best = cluster
                    best_score = score
            packed = {
                "teacher": teacher,
                "claim": claim,
                "evidence": item.get("evidence", ""),
                "capability": item.get("capability", ""),
                "confidence": float(item.get("confidence", 0.7) or 0.7),
            }
            if best is not None and best_score >= 0.42:
                best["items"].append(packed)
                best["teachers"].add(teacher)
            else:
                clusters.append({"items": [packed], "teachers": {teacher}})

    consensus = []
    conflicts = []
    for cluster in clusters:
        teachers = sorted(cluster["teachers"])
        representative = max(cluster["items"], key=lambda x: (x["confidence"], len(x["claim"])))
        row = {
            "id": stable_slug(representative["claim"]),
            "claim": representative["claim"],
            "evidence": representative.get("evidence", ""),
            "capability": representative.get("capability", ""),
            "teachers": teachers,
            "agreement": len(teachers),
            "mean_confidence": round(sum(i["confidence"] for i in cluster["items"]) / len(cluster["items"]), 3),
            "alternates": [i["claim"] for i in cluster["items"] if i["claim"] != representative["claim"]],
        }
        if len(teachers) >= min_agree:
            consensus.append(row)
        else:
            conflicts.append(row)
    consensus.sort(key=lambda x: (-x["agreement"], -x["mean_confidence"], x["id"]))
    conflicts.sort(key=lambda x: (-x["mean_confidence"], x["id"]))
    return {"consensus_ledger": consensus, "low_agreement_items": conflicts}


def process_record(
    rec: dict[str, Any],
    teachers: list[TeacherSpec],
    assets_dir: Path,
    output_dir: Path,
    max_transcript_chars: int,
    timeout: float,
    min_agree: int,
    resume: bool,
) -> dict[str, Any]:
    gold_path = output_dir / f"{rec['id']}.teachers.json"
    if resume and gold_path.exists():
        teacher_outputs = json.loads(gold_path.read_text(encoding="utf-8"))
    else:
        messages = gold_prompt(rec, assets_dir, max_transcript_chars)
        teacher_outputs = []
        for teacher in teachers:
            started = time.time()
            try:
                gold = call_openai_compatible(teacher, messages, timeout)
                status = "ok"
                error = None
            except (
                RuntimeError,
                urllib.error.URLError,
                urllib.error.HTTPError,
                TimeoutError,
                json.JSONDecodeError,
                http.client.HTTPException,
                OSError,
            ) as exc:
                gold = {}
                status = "error"
                error = str(exc)
            teacher_outputs.append(
                {
                    "task_id": rec["id"],
                    "teacher": teacher.name,
                    "model": teacher.model,
                    "status": status,
                    "latency_seconds": round(time.time() - started, 2),
                    "error": error,
                    "gold": gold,
                }
            )
        gold_path.write_text(json.dumps(teacher_outputs, ensure_ascii=False, indent=2), encoding="utf-8")

    ok_outputs = [row for row in teacher_outputs if row["status"] == "ok"]
    consensus = cluster_ledgers(ok_outputs, min_agree)
    return {
        "task_id": rec["id"],
        "source_task_id": rec["source_task_id"],
        "target_capabilities": rec.get("target_capabilities", []),
        "gold_path": str(gold_path),
        "teacher_status": {
            row["teacher"]: {
                "status": row["status"],
                "items": len(row.get("gold", {}).get("answer_ledger", []) or []),
                "latency_seconds": row.get("latency_seconds"),
                "error": row.get("error"),
            }
            for row in teacher_outputs
        },
        **consensus,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate multi-teacher gold ledgers and consensus.")
    parser.add_argument("--tasks", default=str(REPO_ROOT / "rl" / "data" / "meeting_rl_tasks.jsonl"))
    parser.add_argument("--assets-dir", default=str(DEFAULT_ASSETS_DIR))
    parser.add_argument("--output-dir", default=str(REPO_ROOT / "rl" / "data" / "gold_ledgers"))
    parser.add_argument("--teacher", action="append", default=[], help="name=model@base_url#API_KEY_ENV")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--min-agree", type=int, default=2)
    parser.add_argument("--max-transcript-chars", type=int, default=40000)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--workers", type=int, default=1, help="Concurrent task-level teacher generation workers.")
    parser.add_argument("--no-resume", action="store_true", help="Do not reuse existing per-task teacher outputs.")
    parser.add_argument("--dry-run", action="store_true", help="Write teacher request JSONL without API calls.")
    args = parser.parse_args()

    load_default_env_files()
    teachers = [parse_teacher(t) for t in (args.teacher or DEFAULT_TEACHERS)]
    records = read_jsonl(Path(args.tasks))
    if args.limit:
        records = records[: args.limit]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    requests = []
    consensus_rows = []
    for rec in records:
        messages = gold_prompt(rec, Path(args.assets_dir), args.max_transcript_chars)
        requests.extend(
            {
                "task_id": rec["id"],
                "source_task_id": rec["source_task_id"],
                "teacher": teacher.name,
                "model": teacher.model,
                "base_url": teacher.base_url,
                "api_key_env": teacher.api_key_env,
                "messages": messages,
            }
            for teacher in teachers
        )
        if args.dry_run:
            continue

    request_path = output_dir / "teacher_requests.jsonl"
    write_jsonl(request_path, requests)
    print(f"Wrote teacher requests: {request_path} ({len(requests)} rows)")
    if args.dry_run:
        print("Dry run only; no API calls made.")
        return

    resume = not args.no_resume
    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
        futures = {
            executor.submit(
                process_record,
                rec,
                teachers,
                Path(args.assets_dir),
                output_dir,
                args.max_transcript_chars,
                args.timeout,
                args.min_agree,
                resume,
            ): rec
            for rec in records
        }
        done = 0
        for future in as_completed(futures):
            rec = futures[future]
            done += 1
            try:
                row = future.result()
            except Exception as exc:
                row = {
                    "task_id": rec["id"],
                    "source_task_id": rec["source_task_id"],
                    "target_capabilities": rec.get("target_capabilities", []),
                    "gold_path": "",
                    "teacher_status": {"fatal": {"status": "error", "error": str(exc)}},
                    "consensus_ledger": [],
                    "low_agreement_items": [],
                }
            consensus_rows.append(row)
            ok_teachers = sum(1 for status in row["teacher_status"].values() if status.get("status") == "ok")
            print(
                f"[{done}/{len(records)}] {row['task_id']} teachers_ok={ok_teachers} "
                f"consensus={len(row.get('consensus_ledger', []))} "
                f"low_agreement={len(row.get('low_agreement_items', []))}",
                flush=True,
            )

    consensus_rows.sort(key=lambda row: row["task_id"])
    consensus_path = output_dir / "consensus_gold.jsonl"
    write_jsonl(consensus_path, consensus_rows)
    print(f"Wrote consensus gold: {consensus_path} ({len(consensus_rows)} tasks)")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        sys.exit(130)
