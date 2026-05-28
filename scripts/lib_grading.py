"""
PinchBench grading engine.
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from lib_agent import call_judge_api, ensure_agent_exists, run_openclaw_prompt, slugify_model
from lib_tasks import Task


logger = logging.getLogger(__name__)


# API judge (default): DeepSeek OpenAI-compatible API — stable JSON with response_format.
DEFAULT_JUDGE_MODEL = "deepseek-chat"
DEFAULT_JUDGE_BASE_URL = "https://api.deepseek.com/v1"
# When falling back to OpenClaw-embedded judge (no DEEPSEEK_API_KEY), use a capable cloud model id.
DEFAULT_JUDGE_MODEL_OPENCLAW_FALLBACK = "openrouter/anthropic/claude-opus-4.5"
DEFAULT_JUDGE_AGENT_PREFIX = "bench-judge"
# Bumped 180 -> 300 to absorb DeepSeek API congestion observed during a real
# reproduction run (R2 rollouts saw 16 'read operation timed out' errors
# against the 120s rollout-judge timeout while DS was busy). 300s is enough
# margin to keep bench scoring valid when the API takes ~minutes to respond.
DEFAULT_JUDGE_TIMEOUT_SECONDS = 300


@dataclass
class GradeResult:
    task_id: str
    score: float
    max_score: float
    grading_type: str
    breakdown: Dict[str, float]
    notes: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "task_id": self.task_id,
            "score": self.score,
            "max_score": self.max_score,
            "grading_type": self.grading_type,
            "breakdown": self.breakdown,
            "notes": self.notes,
        }


def grade_task(
    *,
    task: Task,
    execution_result: Dict[str, Any],
    skill_dir: Path,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_agent_prefix: str = DEFAULT_JUDGE_AGENT_PREFIX,
    judge_timeout_seconds: float = DEFAULT_JUDGE_TIMEOUT_SECONDS,
    judge_backend: str = "openclaw",
    judge_base_url: Optional[str] = None,
    judge_api_key: Optional[str] = None,
    verbose: bool = False,
) -> GradeResult:
    grading_type = task.grading_type
    if verbose:
        logger.info("   [VERBOSE] Grading task %s with type: %s", task.task_id, grading_type)
        logger.info("   [VERBOSE] Execution status: %s", execution_result.get("status", "unknown"))
    
    if grading_type == "automated":
        result = _grade_automated(task, execution_result, verbose=verbose)
        if verbose:
            logger.info("   [VERBOSE] Automated grade breakdown: %s", result.breakdown)
        return result
    if grading_type == "llm_judge":
        result = _grade_llm_judge(
            task=task,
            execution_result=execution_result,
            judge_model=judge_model,
            judge_agent_prefix=judge_agent_prefix,
            judge_timeout_seconds=judge_timeout_seconds,
            judge_backend=judge_backend,
            judge_base_url=judge_base_url,
            judge_api_key=judge_api_key,
            skill_dir=skill_dir,
            verbose=verbose,
        )
        if verbose:
            logger.info("   [VERBOSE] LLM judge breakdown: %s", result.breakdown)
        return result
    if grading_type == "hybrid":
        auto_result = _grade_automated(task, execution_result, verbose=verbose)
        if _task16_missing_required_report(task, execution_result):
            if verbose:
                logger.info(
                    "   [VERBOSE] Skipping LLM judge: task16 required artifact triage_report.md is missing"
                )
            return GradeResult(
                task_id=task.task_id,
                score=0.0,
                max_score=1.0,
                grading_type="hybrid",
                breakdown={
                    **{f"automated.{k}": v for k, v in auto_result.breakdown.items()},
                    "llm_judge.skipped_missing_required_report": 1.0,
                },
                notes="Artifact gate failed: task16 requires triage_report.md; LLM judge skipped.",
            )
        llm_result = _grade_llm_judge(
            task=task,
            execution_result=execution_result,
            judge_model=judge_model,
            judge_agent_prefix=judge_agent_prefix,
            judge_timeout_seconds=judge_timeout_seconds,
            judge_backend=judge_backend,
            judge_base_url=judge_base_url,
            judge_api_key=judge_api_key,
            skill_dir=skill_dir,
            verbose=verbose,
        )
        return _combine_grades(task, auto_result, llm_result)
    raise ValueError(f"Unknown grading type: {grading_type}")


def _task16_missing_required_report(task: Task, execution_result: Dict[str, Any]) -> bool:
    if task.task_id != "task_16_email_triage":
        return False
    workspace = execution_result.get("workspace", "")
    if not workspace:
        return True
    return not (Path(workspace) / "triage_report.md").is_file()


def preflight_judge_connection(
    *,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    judge_backend: str = "api",
    judge_base_url: Optional[str] = None,
    judge_api_key: Optional[str] = None,
    timeout_seconds: float = 30.0,
) -> None:
    """Fail fast if the configured LLM judge endpoint cannot be reached."""
    if judge_backend != "api":
        logger.info("Skipping judge preflight for backend=%s", judge_backend)
        return
    if not judge_base_url:
        raise ValueError("judge_base_url is required for api judge preflight")
    if not judge_api_key:
        raise ValueError("judge_api_key is required for api judge preflight")

    probe_prompt = (
        "Reply with only a JSON object: {\"ok\": true}. "
        "Do not include any extra text."
    )
    logger.info(
        "Preflighting judge connection: backend=%s model=%s base_url=%s",
        judge_backend,
        judge_model,
        judge_base_url,
    )
    result = call_judge_api(
        prompt=probe_prompt,
        model=judge_model,
        timeout_seconds=timeout_seconds,
        base_url=judge_base_url,
        api_key=judge_api_key,
        response_json=True,
    )
    if result.get("status") != "success":
        raise RuntimeError(
            f"Judge preflight failed: {result.get('error', result.get('status'))}"
        )
    text = str(result.get("text", "")).strip()
    if not text:
        raise RuntimeError("Judge preflight failed: empty response text")
    logger.info("Judge preflight succeeded: model=%s response=%s", judge_model, text[:120])


def resolve_judge_backend_from_env(
    *,
    default_backend: str = "api",
    default_base_url: Optional[str] = None,
    default_api_key: Optional[str] = None,
) -> Dict[str, Optional[str]]:
    """Resolve judge settings from env vars used by training/benchmark entrypoints.

    Defaults: DeepSeek ``deepseek-chat`` via OpenAI-compatible API
    (needs ``DEEPSEEK_API_KEY``). Without a key, falls back to OpenClaw-embedded
    judge (noisier JSON; use API key in production).

    Env vars checked in order: PINCHBENCH_GRADE_JUDGE_API_KEY → JUDGE_API_KEY →
    DEEPSEEK_API_KEY → default_api_key.
    """
    if default_base_url is None:
        default_base_url = DEFAULT_JUDGE_BASE_URL
    backend = os.environ.get("PINCHBENCH_GRADE_JUDGE_BACKEND", default_backend)
    base_url = os.environ.get("PINCHBENCH_GRADE_JUDGE_BASE_URL", default_base_url)
    api_key = os.environ.get(
        "PINCHBENCH_GRADE_JUDGE_API_KEY",
        os.environ.get("JUDGE_API_KEY", os.environ.get("DEEPSEEK_API_KEY", default_api_key)),
    )
    if backend == "api" and not (str(api_key or "").strip()):
        logger.warning(
            "No DEEPSEEK_API_KEY (or JUDGE_API_KEY / PINCHBENCH_GRADE_JUDGE_API_KEY): "
            "LLM judge falling back to OpenClaw embedded agent (%s). "
            "Set DEEPSEEK_API_KEY for the API judge (deepseek-chat).",
            DEFAULT_JUDGE_MODEL_OPENCLAW_FALLBACK,
        )
        backend = "openclaw"
        base_url = None
        api_key = None

    if os.environ.get("PINCHBENCH_GRADE_JUDGE_MODEL"):
        model = os.environ["PINCHBENCH_GRADE_JUDGE_MODEL"]
    elif backend == "api":
        model = DEFAULT_JUDGE_MODEL
    else:
        model = DEFAULT_JUDGE_MODEL_OPENCLAW_FALLBACK
    return {
        "judge_model": model,
        "judge_backend": backend,
        "judge_base_url": base_url,
        "judge_api_key": api_key,
    }


def _grade_automated(task: Task, execution_result: Dict[str, Any], verbose: bool = False) -> GradeResult:
    grading_code = _extract_grading_code(task)
    if not grading_code:
        return GradeResult(
            task_id=task.task_id,
            score=0.0,
            max_score=1.0,
            grading_type="automated",
            breakdown={},
            notes="No automated grading code found",
        )

    namespace: Dict[str, Any] = {}
    exec(grading_code, namespace)
    grade_func = namespace.get("grade")
    if not callable(grade_func):
        return GradeResult(
            task_id=task.task_id,
            score=0.0,
            max_score=1.0,
            grading_type="automated",
            breakdown={},
            notes="Automated grading function missing",
        )

    scores = grade_func(
        execution_result.get("transcript", []),
        execution_result.get("workspace", ""),
    )
    if not isinstance(scores, dict):
        scores = {}
    
    if verbose:
        logger.info("   [VERBOSE] Automated grading scores: %s", scores)

    total = _average_scores(scores)
    return GradeResult(
        task_id=task.task_id,
        score=total,
        max_score=1.0,
        grading_type="automated",
        breakdown=_normalize_score_dict(scores),
        notes="",
    )


def _grade_llm_judge(
    *,
    task: Task,
    execution_result: Dict[str, Any],
    judge_model: str,
    judge_agent_prefix: str,
    judge_timeout_seconds: float,
    judge_backend: str = "openclaw",
    judge_base_url: Optional[str] = None,
    judge_api_key: Optional[str] = None,
    skill_dir: Optional[Path] = None,
    verbose: bool = False,
) -> GradeResult:
    transcript = execution_result.get("transcript", [])
    execution_status = execution_result.get("status", "unknown")

    if execution_status != "success":
        if verbose:
            logger.info(
                "   [VERBOSE] Skipping LLM judge: status=%s",
                execution_status,
            )
        return GradeResult(
            task_id=task.task_id,
            score=0.0,
            max_score=1.0,
            grading_type="llm_judge",
            breakdown={},
            notes=f"Skipped: task execution failed ({execution_status}), no valid transcript to evaluate",
        )

    transcript_summary = _summarize_transcript(transcript)
    if verbose:
        logger.info("   [VERBOSE] Transcript summary for judge (first 1000 chars):\n%s", transcript_summary[:1000])
    workspace_content = _read_workspace_files(execution_result.get("workspace", ""), task=task)
    if verbose and workspace_content:
        logger.info("   [VERBOSE] Workspace files passed to judge (first 500 chars):\n%s", workspace_content[:500])
    rubric = task.llm_judge_rubric or _format_grading_criteria(task)
    prompt = _build_judge_prompt(task, transcript_summary, rubric, workspace_content)

    max_retries = int(os.environ.get("PINCHBENCH_JUDGE_MAX_RETRIES", "3"))
    backoff_base = float(os.environ.get("PINCHBENCH_JUDGE_RETRY_BACKOFF", "1.5"))
    judge_result: Dict[str, Any] = {}
    raw_parsed: Dict[str, Any] = {}
    last_failure_reason = ""
    for attempt in range(max_retries):
        if judge_backend == "api":
            judge_result = call_judge_api(
                prompt=prompt,
                model=judge_model,
                timeout_seconds=judge_timeout_seconds,
                base_url=judge_base_url,
                api_key=judge_api_key,
                response_json=True,
            )
            if verbose:
                logger.info("   [VERBOSE] Judge attempt %d/%d status: %s",
                            attempt + 1, max_retries, judge_result.get("status"))
                if judge_result.get("error"):
                    logger.info("   [VERBOSE] Judge error: %s", judge_result["error"])
            api_ok = judge_result.get("status") == "success"
            raw_parsed = _parse_judge_text(judge_result.get("text", "")) if api_ok else {}
        else:
            agent_id = _ensure_judge_agent(judge_agent_prefix, judge_model, skill_dir)
            judge_workspace = Path(f"/tmp/pinchbench/judge/{task.task_id}")
            judge_result = run_openclaw_prompt(
                agent_id=agent_id,
                prompt=prompt,
                workspace=judge_workspace,
                timeout_seconds=judge_timeout_seconds,
            )
            if verbose:
                logger.info("   [VERBOSE] Judge attempt %d/%d status: %s exit_code: %s",
                            attempt + 1, max_retries,
                            judge_result.get("status"), judge_result.get("exit_code"))
                logger.info("   [VERBOSE] Judge stderr: %s", judge_result.get("stderr", "")[:500])
            api_ok = judge_result.get("status") == "success"
            raw_parsed = _parse_judge_response(judge_result.get("transcript", [])) if api_ok else {}

        if api_ok and raw_parsed:
            break

        if not api_ok:
            last_failure_reason = f"api_status={judge_result.get('status')} error={judge_result.get('error', '')}"
        else:
            preview = (judge_result.get("text", "") or "")[:200].replace("\n", " ")
            last_failure_reason = f"parse_empty (api ok, raw text head='{preview}')"
        logger.warning("Judge attempt %d/%d failed for %s: %s",
                       attempt + 1, max_retries, task.task_id, last_failure_reason)

        if attempt < max_retries - 1:
            time.sleep(backoff_base ** attempt)

    if verbose:
        logger.info("   [VERBOSE] Judge raw response parsed: %s", raw_parsed)

    parsed = _normalize_judge_response(raw_parsed)
    if verbose:
        logger.info("   [VERBOSE] Normalized judge response: %s", parsed)

    breakdown = parsed.get("scores", {})
    total = parsed.get("total")
    notes = parsed.get("notes", "")
    if not raw_parsed:
        notes = (notes or "") + f" [judge_failed_after_{max_retries}_retries: {last_failure_reason}]"
        logger.error("Judge exhausted retries for %s: %s", task.task_id, last_failure_reason)
    return GradeResult(
        task_id=task.task_id,
        score=float(total) if total is not None else 0.0,
        max_score=1.0,
        grading_type="llm_judge",
        breakdown=_normalize_score_dict(breakdown),
        notes=str(notes) if notes is not None else "",
    )


def _combine_grades(task: Task, auto_result: GradeResult, llm_result: GradeResult) -> GradeResult:
    weights = task.grading_weights or {"automated": 0.5, "llm_judge": 0.5}
    auto_weight = float(weights.get("automated", 0.5))
    llm_weight = float(weights.get("llm_judge", 0.5))
    total_weight = auto_weight + llm_weight
    if total_weight <= 0:
        auto_weight = llm_weight = 0.5
        total_weight = 1.0
    combined_score = (
        auto_result.score * auto_weight + llm_result.score * llm_weight
    ) / total_weight
    breakdown = {
        **{f"automated.{k}": v for k, v in auto_result.breakdown.items()},
        **{f"llm_judge.{k}": v for k, v in llm_result.breakdown.items()},
    }
    notes = " | ".join(filter(None, [auto_result.notes, llm_result.notes]))
    return GradeResult(
        task_id=task.task_id,
        score=combined_score,
        max_score=1.0,
        grading_type="hybrid",
        breakdown=breakdown,
        notes=notes,
    )


def _extract_grading_code(task: Task) -> str:
    if not task.automated_checks:
        return ""
    match = re.search(r"```python\s*(.*?)\s*```", task.automated_checks, re.DOTALL)
    if not match:
        return ""
    return match.group(1)


def _average_scores(scores: Dict[str, Any]) -> float:
    values = [float(v) for v in scores.values() if isinstance(v, (int, float))]
    if not values:
        return 0.0
    return sum(values) / len(values)


def _normalize_score_dict(scores: Dict[str, Any]) -> Dict[str, float]:
    normalized: Dict[str, float] = {}
    for key, value in scores.items():
        try:
            normalized[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return normalized


def _format_grading_criteria(task: Task) -> str:
    if not task.grading_criteria:
        return ""
    return "\n".join(f"- {criterion}" for criterion in task.grading_criteria)


def _summarize_transcript(transcript: List[Dict[str, Any]]) -> str:
    summary_parts: List[str] = []
    for event in transcript:
        if event.get("type") != "message":
            continue
        msg = event.get("message", {})
        role = msg.get("role")
        if role == "assistant":
            for item in msg.get("content", []):
                if item.get("type") == "toolCall":
                    args = item.get("arguments", {})
                    truncated_args: Dict[str, Any] = {}
                    for k, v in args.items():
                        if isinstance(v, str) and len(v) > 200:
                            truncated_args[k] = v[:200] + "...[truncated]"
                        else:
                            truncated_args[k] = v
                    summary_parts.append(
                        f"Tool: {item.get('name')}({json.dumps(truncated_args)})"
                    )
                elif item.get("type") == "text":
                    text = item.get("text", "").strip()
                    if text:
                        summary_parts.append(f"Assistant: {text[:2000]}")
        elif role == "toolResult":
            content = msg.get("content", [])
            if content:
                result_preview = str(content[0])[:200]
                summary_parts.append(f"Result: {result_preview}")
        elif role == "user":
            content = msg.get("content", [])
            if content:
                summary_parts.append(f"User: {content[0]}")
    return "\n".join(summary_parts)


def _fixture_paths(task: Task) -> set[str]:
    paths: set[str] = set()
    for file_spec in task.workspace_files:
        rel = file_spec.get("path") or file_spec.get("dest")
        if rel:
            paths.add(str(Path(rel)))
    return paths


def _read_workspace_files(workspace_path: str, *, task: Optional[Task] = None) -> str:
    """Read user-created text files from workspace to provide grading context."""
    if not workspace_path:
        return ""
    workspace = Path(workspace_path)
    if not workspace.exists():
        return ""
    fixture_paths = _fixture_paths(task) if task is not None else set()
    skip_names = {
        "BOOTSTRAP.md", "SOUL.md", "USER.md", "IDENTITY.md",
        "HEARTBEAT.md", "TOOLS.md", "AGENTS.md",
    }
    skip_dirs = {".git", ".openclaw", "__pycache__", "node_modules", "skills"}
    file_contents: List[str] = []
    for f in sorted(workspace.rglob("*")):
        if not f.is_file():
            continue
        rel = f.relative_to(workspace)
        parts = rel.parts
        if str(rel) in fixture_paths:
            continue
        if any(part.startswith(".") or part in skip_dirs for part in parts):
            continue
        if f.name in skip_names:
            continue
        try:
            content = f.read_text(encoding="utf-8")
            file_contents.append(f"### File: {rel}\n{content[:3000]}")
        except (OSError, UnicodeDecodeError):
            pass
    return "\n\n".join(file_contents)


def _build_judge_prompt(task: Task, transcript_summary: str, rubric: str, workspace_content: str = "") -> str:
    workspace_section = ""
    if workspace_content.strip():
        workspace_section = (
            "## Workspace Files Created by Agent\n"
            f"{workspace_content}\n\n"
        )
    return (
        "You are a grading function. Your ONLY job is to output a single JSON object.\n\n"
        "CRITICAL RULES:\n"
        "- Do NOT use any tools (no Read, Write, exec, or any other tool calls)\n"
        "- Do NOT create files or run commands\n"
        "- Do NOT write any prose, explanation, or commentary outside the JSON\n"
        "- Respond with ONLY a JSON object — nothing else\n\n"
        "Be a strict evaluator. Reserve 1.0 for genuinely excellent performance. "
        "An average acceptable completion should score around 0.6-0.7. "
        "Deduct points for unnecessary steps, verbose output, and inefficient tool usage.\n\n"
        "## Task\n"
        f"{task.prompt}\n\n"
        "## Expected Behavior\n"
        f"{task.expected_behavior}\n\n"
        "## Agent Transcript (summarized)\n"
        f"{transcript_summary}\n\n"
        f"{workspace_section}"
        "## Grading Rubric\n"
        f"{rubric}\n\n"
        "Score each criterion from 0.0 to 1.0.\n"
        'The "total" field must also be between 0.0 and 1.0, and it must be the arithmetic mean of the criterion scores, not their sum.\n\n'
        "Respond with ONLY this JSON structure (no markdown, no code fences, no extra text):\n"
        '{"scores": {"criterion_name": 0.0}, "total": 0.0, "notes": "brief justification"}'
    )


def _ensure_judge_agent(judge_agent_prefix: str, judge_model: str, skill_dir: Path) -> str:
    model_slug = slugify_model(judge_model)
    agent_id = f"{judge_agent_prefix}-{model_slug}"
    workspace = Path("/tmp/pinchbench/judge/workspace")
    ensure_agent_exists(agent_id, judge_model, workspace)
    return agent_id


def _parse_judge_response(transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
    content_chunks: List[str] = []
    for event in transcript:
        if event.get("type") != "message":
            continue
        msg = event.get("message", {})
        if msg.get("role") != "assistant":
            continue
        for item in msg.get("content", []):
            if item.get("type") == "text":
                content_chunks.append(item.get("text", ""))
    raw_text = "\n".join(content_chunks).strip()
    logger.info("   [VERBOSE] Judge raw response text (first 2000 chars):\n%s", raw_text[:2000])
    if not raw_text:
        return {}

    # First, try to extract JSON from code blocks (```json ... ```)
    code_block_match = re.search(r"```json\s*(.*?)\s*```", raw_text, re.DOTALL)
    if code_block_match:
        try:
            parsed = json.loads(code_block_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Find all potential JSON objects by looking for balanced braces
    # We'll extract chunks that start with { and try to parse them
    json_candidates: List[str] = []
    brace_depth = 0
    current_json = []
    for char in raw_text:
        if char == "{":
            if brace_depth == 0:
                current_json = []
            brace_depth += 1

        if brace_depth > 0:
            current_json.append(char)

        if char == "}":
            brace_depth -= 1
            if brace_depth == 0 and current_json:
                json_candidates.append("".join(current_json))

    # Try parsing from the last JSON object backwards (most recent response)
    for candidate in reversed(json_candidates):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "scores" in parsed:
                # Prefer JSON that has the expected structure
                return parsed
        except json.JSONDecodeError:
            continue

    # Try any valid JSON dict
    for candidate in reversed(json_candidates):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    # Fallback: try to extract numeric scores from prose responses.
    # Models sometimes return "Total: 0.72" or "Overall score: 0.65" instead of JSON.
    score_pattern = re.search(
        r"(?:total|overall|final)\s*(?:score)?[:\s]*(0\.\d+|1\.0+)",
        raw_text,
        re.IGNORECASE,
    )
    if score_pattern:
        try:
            total = float(score_pattern.group(1))
            if 0.0 <= total <= 1.0:
                logger.warning(
                    "Fell back to regex score extraction from prose (total=%.2f)", total
                )
                return {"scores": {}, "total": total, "notes": "Score extracted from prose (JSON parse failed)"}
        except ValueError:
            pass

    logger.warning("Failed to parse judge JSON response")
    return {}


def _parse_judge_text(raw_text: str) -> Dict[str, Any]:
    """Parse judge response from raw text (direct API call, no OpenClaw transcript)."""
    raw_text = raw_text.strip()
    if not raw_text:
        return {}

    # Try direct JSON parse first (ideal case with system prompt enforcement)
    try:
        parsed = json.loads(raw_text)
        if isinstance(parsed, dict):
            return parsed
    except json.JSONDecodeError:
        pass

    # Try extracting from code blocks
    code_block_match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw_text, re.DOTALL)
    if code_block_match:
        try:
            parsed = json.loads(code_block_match.group(1))
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            pass

    # Find balanced-brace JSON objects
    json_candidates: List[str] = []
    brace_depth = 0
    current_json: List[str] = []
    for char in raw_text:
        if char == "{":
            if brace_depth == 0:
                current_json = []
            brace_depth += 1
        if brace_depth > 0:
            current_json.append(char)
        if char == "}":
            brace_depth -= 1
            if brace_depth == 0 and current_json:
                json_candidates.append("".join(current_json))

    for candidate in reversed(json_candidates):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and "scores" in parsed:
                return parsed
        except json.JSONDecodeError:
            continue
    for candidate in reversed(json_candidates):
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict):
                return parsed
        except json.JSONDecodeError:
            continue

    # Fallback: regex for total score
    score_pattern = re.search(
        r"(?:total|overall|final)\s*(?:score)?[:\s]*(0\.\d+|1\.0+)",
        raw_text,
        re.IGNORECASE,
    )
    if score_pattern:
        try:
            total = float(score_pattern.group(1))
            if 0.0 <= total <= 1.0:
                logger.warning("Fell back to regex score extraction (total=%.2f)", total)
                return {"scores": {}, "total": total, "notes": "Score extracted from prose"}
        except ValueError:
            pass

    logger.warning("Failed to parse judge text response")
    return {}


def _normalize_judge_response(parsed: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize judge response to expected format with 'scores', 'total', and 'notes'.
    
    Handles various response formats:
    - {"scores": {...}, "total": 0.9, "notes": "..."}  (expected)
    - {"criteria_scores": {...}, ...}  (Claude sometimes uses this)
    - {"score": 0.9, "justification": "..."}  (simplified format)
    """
    result: Dict[str, Any] = {"scores": {}, "total": None, "notes": ""}
    
    # Extract scores from various keys
    if "scores" in parsed:
        scores_data = parsed["scores"]
        if isinstance(scores_data, dict):
            # Handle nested structure: {"criterion": {"score": 0.9, "weight": 0.3}}
            for key, value in scores_data.items():
                if isinstance(value, dict) and "score" in value:
                    result["scores"][key] = float(value["score"]) if isinstance(value["score"], (int, float, str)) else value["score"]
                elif isinstance(value, (int, float)):
                    result["scores"][key] = value
    elif "criteria_scores" in parsed:
        # Handle Claude's alternate format
        criteria = parsed["criteria_scores"]
        if isinstance(criteria, dict):
            for key, value in criteria.items():
                if isinstance(value, dict) and "score" in value:
                    result["scores"][key] = value["score"]
                elif isinstance(value, (int, float)):
                    result["scores"][key] = value
    
    # Extract total score
    if "total" in parsed and parsed["total"] is not None:
        result["total"] = float(parsed["total"]) if isinstance(parsed["total"], (int, float)) else None
    elif "score" in parsed and isinstance(parsed["score"], (int, float)):
        result["total"] = float(parsed["score"])
    elif "overall_score" in parsed and isinstance(parsed["overall_score"], (int, float)):
        result["total"] = float(parsed["overall_score"])
    elif result["scores"]:
        # Calculate average if we have individual scores but no total
        values = [v for v in result["scores"].values() if isinstance(v, (int, float))]
        if values:
            result["total"] = sum(values) / len(values)

    # Some judge models return a summed total across criteria even though each
    # criterion is scored on a 0..1 scale. Normalize that back to a 0..1 mean.
    values = [v for v in result["scores"].values() if isinstance(v, (int, float))]
    if (
        values
        and result["total"] is not None
        and result["total"] > 1.0
        and all(0.0 <= float(v) <= 1.0 for v in values)
    ):
        result["total"] = sum(values) / len(values)
    
    # Extract notes/justification
    if "notes" in parsed:
        result["notes"] = str(parsed["notes"])
    elif "justification" in parsed:
        result["notes"] = str(parsed["justification"])
    elif "reasoning" in parsed:
        result["notes"] = str(parsed["reasoning"])
    
    return result


# ---- Stub: judge cache shims (originals not in this checkout) ----
_JUDGE_CACHE_DIR = None

def set_judge_cache_dir(path):
    global _JUDGE_CACHE_DIR
    _JUDGE_CACHE_DIR = path

def get_judge_cache_stats():
    return {'hits': 0, 'misses': 0, 'size': 0, 'dir': str(_JUDGE_CACHE_DIR) if _JUDGE_CACHE_DIR else None}

def clear_judge_cache():
    return 0
