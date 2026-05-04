from __future__ import annotations

import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any


def _tool_args(tool_call: dict[str, Any]) -> dict[str, Any]:
    fn = tool_call.get("function")
    if isinstance(fn, dict):
        args = fn.get("arguments", {})
    else:
        args = tool_call.get("arguments", {})
    if isinstance(args, dict):
        return args
    if isinstance(args, str):
        try:
            parsed = json.loads(args)
            return parsed if isinstance(parsed, dict) else {}
        except Exception:
            return {}
    return {}


def _tool_name(tool_call: dict[str, Any]) -> str:
    fn = tool_call.get("function")
    if isinstance(fn, dict):
        return str(fn.get("name", ""))
    return str(tool_call.get("name", ""))


def _assistant_turns(trajectory: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [turn for turn in trajectory if turn.get("role") == "assistant"]


def _read_paths(trajectory: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    for turn in _assistant_turns(trajectory):
        for tc in turn.get("tool_calls") or []:
            if _tool_name(tc) == "read":
                path = _tool_args(tc).get("path")
                if path:
                    paths.append(str(path))
    return paths


def _write_paths(trajectory: list[dict[str, Any]]) -> list[str]:
    paths: list[str] = []
    for turn in _assistant_turns(trajectory):
        for tc in turn.get("tool_calls") or []:
            if _tool_name(tc) == "write":
                path = _tool_args(tc).get("path")
                if path:
                    paths.append(str(path))
    return paths


def _is_task16_inbox_path(path: str) -> bool:
    return path.strip().lower().startswith("inbox/email_")


def _is_task16_bad_inbox_path(path: str) -> bool:
    p = path.strip().lower().replace("\\", "/")
    return bool(
        re.search(r"(^|/)(inboxes|inabox)/email_\d{2}\.txt$", p)
        or re.search(r"/\s+inbox/email_\d{2}\.txt$", p)
        or ("/inbox/email_" in p and not _is_task16_inbox_path(p))
    )


def _is_report_path(path: str) -> bool:
    return path.strip().lower().endswith("triage_report.md")


def _is_bad_report_path(path: str) -> bool:
    name = path.strip().lower().replace(" ", "").rsplit("/", 1)[-1]
    return name != "triage_report.md" and "triage" in name and "report" in name and name.endswith(".md")


def _workspace_evidence(task_id: str, workspace_path: str) -> dict[str, Any]:
    path = Path(workspace_path) if workspace_path else Path()
    evidence: dict[str, Any] = {
        "workspace_path": str(path) if workspace_path else "",
        "workspace_exists": bool(workspace_path and path.exists()),
        "report_exists_local": bool(workspace_path and (path / "triage_report.md").exists()),
    }
    if task_id == "task_16_email_triage" and workspace_path:
        expected = [path / "inbox" / f"email_{i:02d}.txt" for i in range(1, 14)]
        existing = [p for p in expected if p.exists()]
        evidence["workspace_seeded_files"] = len(existing)
        evidence["workspace_missing_files"] = [
            str(p.relative_to(path)) for p in expected if not p.exists()
        ]
    return evidence


def classify_rollout(
    *,
    task_id: str,
    session_id: str,
    workspace_path: str,
    trajectory: list[dict[str, Any]],
    terminal_success: bool,
    terminal_grade_score: float | None,
    episode_tags: dict[str, Any] | None,
    diagnostics_state: dict[str, Any] | None,
    grading_status: dict[str, Any] | None,
) -> dict[str, Any]:
    reads = _read_paths(trajectory)
    writes = _write_paths(trajectory)
    assistant = _assistant_turns(trajectory)
    clipped_turns = [
        idx for idx, turn in enumerate(assistant)
        if turn.get("response_clipped") or str(turn.get("finish_reason", "")).lower() in {"length", "max_tokens"}
    ]

    evidence = {
        "read_paths": reads[:50],
        "write_paths": writes[:20],
        "bad_read_paths": [p for p in reads if _is_task16_bad_inbox_path(p)],
        "bad_report_paths": [p for p in writes if _is_bad_report_path(p)],
        "report_write_calls": [p for p in writes if _is_report_path(p)],
        "assistant_turns": len(assistant),
        "response_clipped_turns": clipped_turns,
        "prompt_compactions": int((diagnostics_state or {}).get("prompt_compactions", 0)),
        "response_compactions": int((diagnostics_state or {}).get("response_compactions", 0)),
        "response_budget_exhausted": bool((diagnostics_state or {}).get("response_budget_exhausted", False)),
        "max_prompt_tokens_seen": int((diagnostics_state or {}).get("max_prompt_tokens_seen", 0)),
        "max_response_tokens_seen": int((diagnostics_state or {}).get("max_response_tokens_seen", 0)),
        "grading_status": grading_status or {},
    }
    evidence.update(_workspace_evidence(task_id, workspace_path))

    tags = list((episode_tags or {}).get("failure_tags") or [])
    harness_tags: list[str] = []
    model_tags: list[str] = []
    grader_tags: list[str] = []

    if not trajectory:
        harness_tags.append("empty_transcript")
    if grading_status and grading_status.get("sync_status") == "failed":
        harness_tags.append("workspace_rsync_failed")
    if grading_status and grading_status.get("error"):
        grader_tags.append("grader_failed")
    judge_status = (grading_status or {}).get("judge_status")
    if isinstance(judge_status, dict):
        if judge_status.get("preflight") == "failed":
            grader_tags.append("judge_preflight_failed")
        if judge_status.get("grade_call") == "failed":
            grader_tags.append("judge_call_failed")
        if judge_status.get("backend") != "api":
            grader_tags.append("judge_not_api")
    if task_id == "task_16_email_triage":
        if evidence.get("workspace_exists") and evidence.get("workspace_seeded_files", 13) < 13:
            harness_tags.append("workspace_missing_task_files")
        if evidence["bad_read_paths"]:
            model_tags.append("bad_tool_path")
        if evidence["bad_report_paths"]:
            model_tags.append("report_path_typo")
        if not evidence["report_exists_local"] and not evidence["report_write_calls"]:
            model_tags.append("no_report")
    if clipped_turns:
        model_tags.append("response_clipped")
    if evidence["response_budget_exhausted"]:
        model_tags.append("response_budget_exhausted")
    if evidence["prompt_compactions"]:
        tags.append("context_compacted")
    if evidence["response_compactions"]:
        tags.append("response_compacted")

    tags.extend(harness_tags)
    tags.extend(model_tags)
    tags.extend(grader_tags)
    tags = sorted(set(tags))

    if harness_tags and model_tags:
        owner = "mixed"
    elif harness_tags:
        owner = "harness"
    elif grader_tags and not model_tags:
        owner = "grader"
    elif model_tags:
        owner = "model"
    elif terminal_success or (terminal_grade_score is not None and terminal_grade_score >= 0.5):
        owner = "none"
    else:
        owner = "unknown"

    return {
        "schema_version": 1,
        "created_at": time.time(),
        "task_id": task_id,
        "episode_id": session_id,
        "terminal_success": bool(terminal_success),
        "terminal_grade_score": terminal_grade_score,
        "failure_owner": owner,
        "failure_tags": tags,
        "harness_tags": sorted(set(harness_tags)),
        "model_tags": sorted(set(model_tags)),
        "grader_tags": sorted(set(grader_tags)),
        "evidence": evidence,
    }


def _safe_json_obj(text: str) -> dict[str, Any]:
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return {}
    try:
        parsed = json.loads(m.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        return {}


def _trajectory_summary(trajectory: list[dict[str, Any]], max_turns: int = 12) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for idx, turn in enumerate(trajectory):
        role = str(turn.get("role", ""))
        if role not in {"assistant", "tool", "user", "system"}:
            continue
        item: dict[str, Any] = {"idx": idx, "role": role}
        if role == "assistant":
            item["tool_calls"] = turn.get("tool_calls") or []
            content = str(turn.get("content", ""))
            if content:
                item["content_head"] = content[:600]
                if len(content) > 600:
                    item["content_tail"] = content[-300:]
            if turn.get("response_clipped"):
                item["response_clipped"] = True
            if turn.get("finish_reason"):
                item["finish_reason"] = turn.get("finish_reason")
        elif role == "tool":
            content = str(turn.get("content", ""))
            item["content_head"] = content[:500]
        else:
            content = str(turn.get("content", ""))
            item["content_head"] = content[:500]
        out.append(item)
    if len(out) <= max_turns:
        return out
    return out[:4] + [{"omitted_turns": len(out) - 8}] + out[-4:]


def attach_llm_diagnostics(
    record: dict[str, Any],
    *,
    trajectory: list[dict[str, Any]],
    pinchbench_dir: str | os.PathLike[str] | None,
) -> dict[str, Any]:
    """Use the configured qwen-plus judge to diagnose rollout failure ownership.

    Deterministic diagnostics above are only evidence extraction. This optional
    LLM pass is the actual attribution layer used for reports/debugging.
    """
    if os.environ.get("PINCHBENCH_LLM_DIAGNOSTICS", "1").strip().lower() in {"0", "false", "no", "off"}:
        record["llm_diagnostics"] = {"status": "skipped_disabled"}
        return record

    try:
        root = Path(pinchbench_dir or os.getcwd())
        scripts_dir = root / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from lib_agent import call_judge_api
        from lib_grading import resolve_judge_backend_from_env

        judge_cfg = resolve_judge_backend_from_env(
            default_backend="api",
            default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
        )
        if judge_cfg.get("judge_backend") != "api":
            record["llm_diagnostics"] = {
                "status": "skipped_non_api_backend",
                "backend": judge_cfg.get("judge_backend"),
            }
            return record
        if not str(judge_cfg.get("judge_api_key") or "").strip():
            record["llm_diagnostics"] = {"status": "failed", "error": "missing_api_key"}
            return record

        payload = {
            "task_id": record.get("task_id"),
            "episode_id": record.get("episode_id"),
            "terminal_success": record.get("terminal_success"),
            "terminal_grade_score": record.get("terminal_grade_score"),
            "rule_based_owner": record.get("failure_owner"),
            "rule_based_tags": record.get("failure_tags"),
            "evidence": record.get("evidence"),
            "trajectory_summary": _trajectory_summary(trajectory),
        }
        prompt = (
            "You are diagnosing one failed or partially successful OpenClaw + PinchBench RL rollout.\n"
            "Decide the PRIMARY failure owner using the evidence and trajectory, not by guessing.\n\n"
            "Owner definitions:\n"
            "- model: the model made bad tool calls, wrong paths, did not write the required artifact, read-looped, hallucinated, or failed task reasoning.\n"
            "- harness: environment/workspace/OpenClaw/tool execution/sync corrupted the run despite reasonable model actions.\n"
            "- grader: judge/grading failed, unavailable, inconsistent, or could not inspect the artifact.\n"
            "- mixed: more than one owner materially caused failure.\n"
            "- none: rollout succeeded or no material failure.\n"
            "- unknown: evidence is insufficient.\n\n"
            "Return ONLY compact JSON with this schema:\n"
            "{\"owner\":\"model|harness|grader|mixed|none|unknown\","
            "\"confidence\":0.0,"
            "\"failure_tags\":[\"...\"],"
            "\"reason\":\"one concise paragraph\","
            "\"evidence\":[\"specific observed facts\"],"
            "\"actionable_fixes\":[\"concrete fixes\"]}\n\n"
            f"ROLLOUT_JSON:\n{json.dumps(payload, ensure_ascii=False)[:18000]}"
        )
        result = call_judge_api(
            prompt=prompt,
            model=str(judge_cfg.get("judge_model") or "qwen-plus"),
            timeout_seconds=float(os.environ.get("PINCHBENCH_LLM_DIAGNOSTICS_TIMEOUT", "45")),
            base_url=str(judge_cfg.get("judge_base_url") or ""),
            api_key=str(judge_cfg.get("judge_api_key") or ""),
            response_json=True,
        )
        if result.get("status") != "success":
            record["llm_diagnostics"] = {
                "status": "failed",
                "backend": "api",
                "model": judge_cfg.get("judge_model"),
                "error": result.get("error", result.get("status")),
            }
            return record
        parsed = _safe_json_obj(str(result.get("text", "")))
        record["llm_diagnostics"] = {
            "status": "ok",
            "backend": "api",
            "model": judge_cfg.get("judge_model"),
            **parsed,
        }
        if parsed.get("owner"):
            record["llm_failure_owner"] = parsed.get("owner")
        if isinstance(parsed.get("failure_tags"), list):
            record["llm_failure_tags"] = parsed.get("failure_tags")
        return record
    except Exception as e:
        record["llm_diagnostics"] = {"status": "failed", "error": str(e)}
        return record


def append_diagnostics(record: dict[str, Any], path: str | os.PathLike[str]) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")
