#!/usr/bin/env python3
"""Run a sub-agent rollout — single-shot from Lead's perspective.

The sub-agent is a frozen Qwen3-4B base instance that operates in its own
OpenClaw session with full read/write/exec autonomy on a designated source
file (or slice). When done, ONLY its final assistant text summary is printed
to stdout. The full sub transcript is archived for analysis but not exposed
to the Lead.

Usage (called by Lead via `exec subagent.sh ...`):
    subagent.sh "<task description>" <source_file> [<offset>] [<length>]

Returns: stdout = sub's final summary text.
Side effect: archive sub transcript at $SUBAGENT_LOG_DIR/<sub_agent_id>__<task_id>.jsonl

Environment:
    SUBAGENT_MODEL           default qwen3-mr2 (base, frozen)
    SUBAGENT_VLLM_URL        default http://localhost:8766/v1
    SUBAGENT_TIMEOUT         default 180
    SUBAGENT_LOG_DIR         default /tmp/subagent_logs
"""
from __future__ import annotations

import argparse, json, os, re, sys, uuid
from pathlib import Path


def _autodetect_source(cwd: Path) -> Path | None:
    """Best-effort find the input transcript when the Lead omits source_file.

    The Lead's workspace holds the input transcript (large) plus possibly a
    partially-written deliverable (small/empty). Prefer transcript-like names,
    else the largest text file.
    """
    cands = sorted(
        [p for p in cwd.glob("*.md") if p.is_file()]
        + [p for p in cwd.glob("*.txt") if p.is_file()]
    )
    if not cands:
        return None
    pref = [p for p in cands if re.search(r"transcript|meeting|minutes|hearing", p.name, re.I)]
    pool = pref or cands
    return max(pool, key=lambda p: p.stat().st_size)

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Force OpenClaw to run locally — this is a pod-side helper, never use ECS remote.
os.environ.setdefault("PINCHBENCH_FORCE_LOCAL_OPENCLAW", "1")
os.environ.setdefault("OPENCLAW_HOST", "localhost")
os.environ.setdefault("PINCHBENCH_SKIP_OPENCLAW_WEB_PREFLIGHT", "1")
os.environ.setdefault("PINCHBENCH_ALLOW_LOCAL_OPENCLAW", "1")

from lib_agent import ensure_agent_exists, execute_openclaw_task  # noqa: E402
from lib_tasks import Task  # noqa: E402


def _build_sub_task(task_id: str, description: str, source_file: str,
                    offset: int | None, length: int | None) -> Task:
    slice_hint = ""
    if offset is not None or length is not None:
        slice_hint = f"\nFocus on: offset={offset or 0}"
        if length:
            slice_hint += f", length={length}"
    prompt = f"""You are a focused sub-agent. Your only job is one specific sub-task.

Source file (absolute path): {source_file}{slice_hint}

Sub-task: {description}

Use read/grep/exec as needed to inspect the file. Then produce a thorough,
structured result for the calling agent. Be as detailed as the sub-task
requires — INCLUDE verbatim quotes, exact numbers, names, and direct
references from the source whenever they are relevant. The calling agent has
plenty of context space (65k tokens) so DO NOT compress for brevity.

Format your output as structured markdown (tables, bullet lists, JSON blocks
as appropriate). Your final assistant message (the LAST thing you say) will
be returned to the calling agent verbatim — make it BE the structured result
content with raw evidence preserved.

Do not write narrative about your process. Just produce the requested data
with full fidelity."""

    return Task(
        task_id=task_id,
        name=f"subagent: {description[:60]}",
        category="subagent",
        grading_type="automated",
        timeout_seconds=int(os.environ.get("SUBAGENT_TIMEOUT", "180")),
        workspace_files=[],
        prompt=prompt,
        expected_behavior="Read the source, perform the sub-task, return a detailed structured result (verbatim quotes preserved) as the final assistant message.",
        grading_criteria=[],
        automated_checks="",
        llm_judge_rubric="",
        grading_weights={"automated": 1.0, "llm_judge": 0.0},
    )


def _final_assistant_text(transcript: list) -> str:
    """Extract the LAST assistant text-content from a transcript."""
    for event in reversed(transcript):
        if event.get("type") != "message":
            continue
        msg = event.get("message", {})
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str) and content.strip():
            return content.strip()
        if isinstance(content, list):
            for item in reversed(content):
                if not isinstance(item, dict):
                    continue
                if item.get("type") == "text":
                    t = (item.get("text") or "").strip()
                    if t:
                        return t
    return ""


def main():
    p = argparse.ArgumentParser()
    p.add_argument("task_description")
    p.add_argument("source_file", nargs="?", default=None)
    p.add_argument("offset", nargs="?", type=int, default=None)
    p.add_argument("length", nargs="?", type=int, default=None)
    args = p.parse_args()

    model = os.environ.get("SUBAGENT_MODEL", "qwen3-mr2")
    base_url = os.environ.get("SUBAGENT_VLLM_URL", "http://localhost:8766/v1")
    log_dir = Path(os.environ.get("SUBAGENT_LOG_DIR", "/tmp/subagent_logs"))
    log_dir.mkdir(parents=True, exist_ok=True)

    # Resolve source file (allow relative-to-cwd); auto-detect if Lead omitted it.
    if args.source_file:
        source = Path(args.source_file)
        if not source.is_absolute():
            source = Path.cwd() / source
        if not source.exists():
            auto = _autodetect_source(Path.cwd())
            if auto is not None:
                print(f"[subagent] source '{args.source_file}' not found; using {auto.name}", file=sys.stderr)
                source = auto
    else:
        source = _autodetect_source(Path.cwd())
        if source is None:
            print("[subagent error] no source_file given and none auto-detected", file=sys.stderr)
            sys.exit(2)
        print(f"[subagent] auto-detected source: {source.name}", file=sys.stderr)
    if source is None or not source.exists():
        print(f"[subagent error] source file not found: {source}", file=sys.stderr)
        sys.exit(2)

    task_id = f"subagent_{uuid.uuid4().hex[:8]}"
    sub_agent_id = f"sub-{model.lower().replace('/', '-')}-{uuid.uuid4().hex[:6]}"
    sub_workspace = log_dir / sub_agent_id / "workspace"
    sub_workspace.mkdir(parents=True, exist_ok=True)

    task = _build_sub_task(task_id, args.task_description, str(source), args.offset, args.length)

    ensure_agent_exists(
        agent_id=sub_agent_id,
        model_id=f"custom/{model}",
        workspace_dir=sub_workspace,
        base_url=base_url,
        api_key="dummy",
    )

    result = execute_openclaw_task(
        task=task,
        agent_id=sub_agent_id,
        model_id=f"custom/{model}",
        run_id=f"sub_{uuid.uuid4().hex[:8]}",
        timeout_multiplier=1.0,
        skill_dir=REPO_ROOT / "assets",
    )

    transcript = result.get("transcript", [])
    final_text = _final_assistant_text(transcript)
    if not final_text:
        # Fallback: look for a subagent_output.md the sub may have written
        out_file = sub_workspace / "subagent_output.md"
        if out_file.is_file():
            final_text = out_file.read_text(encoding="utf-8", errors="replace").strip()
    if not final_text:
        final_text = f"[subagent produced no summary; status={result.get('status', '?')}]"

    # Archive
    archive = log_dir / f"{sub_agent_id}__{task_id}.jsonl"
    with archive.open("w") as f:
        for ev in transcript:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")

    # ONLY the summary goes to stdout — this is what Lead sees
    print(final_text)


if __name__ == "__main__":
    main()
