#!/usr/bin/env python3
"""Self-evolve the swarm meeting analysis skill prompt.

Given a bench output dir (with N rollouts of current skill v_n), feed
DSv4-Pro: (1) current skill text, (2) top failures & successes, (3) ask for
a revised v_n+1 that addresses the dominant failure mode.

Inputs:
  --current-skill <path>           # current skill prompt (txt)
  --bench-dir <path>                # bench output dir with graded_swarm_trajectories.jsonl
  --out-skill <path>               # write evolved skill v_n+1 here
  --num-success <int>              # how many success rollouts to feed evolver
  --num-failure <int>              # how many failure rollouts to feed evolver

Output: new skill text in --out-skill.
"""
from __future__ import annotations
import argparse, json, os, sys, urllib.request
from pathlib import Path
from typing import Any


def _trim(t: str, n: int = 1200) -> str:
    return t if len(t) <= n else t[:n] + " ...[truncated]"


def _trajectory_summary(transcript_path: str) -> str:
    """Render a compact, human-readable summary of a Lead's trajectory."""
    if not Path(transcript_path).is_file():
        return f"(transcript missing: {transcript_path})"
    events = []
    for line in open(transcript_path):
        line = line.strip()
        if not line:
            continue
        try:
            events.append(json.loads(line))
        except Exception:
            continue
    summary_lines = []
    for e in events:
        if e.get("type") != "message":
            continue
        msg = e.get("message", {})
        role = msg.get("role")
        if role == "assistant":
            for item in msg.get("content", []):
                if not isinstance(item, dict):
                    continue
                t = item.get("type")
                if t == "toolCall":
                    name = item.get("name")
                    args = item.get("arguments", {}) or {}
                    cmd = str(args.get("command", ""))[:80]
                    eargs = args.get("args", []) or []
                    first = str(eargs[0])[:120] if eargs else ""
                    summary_lines.append(f"  asst.tool_call: {name}({cmd} | args0={first})")
                elif t == "text":
                    txt = item.get("text", "").strip()
                    if txt:
                        summary_lines.append(f"  asst.text: {_trim(txt, 200)}")
                elif t == "thinking":
                    th = item.get("thinking", "").strip()
                    if th:
                        summary_lines.append(f"  asst.thinking: {_trim(th, 200)}")
        elif role == "toolResult":
            for item in msg.get("content", []):
                if isinstance(item, dict):
                    t = str(item.get("text") or item.get("output") or "")
                    summary_lines.append(f"  tool_result: {_trim(t, 150)}")
    return "\n".join(summary_lines[:50])  # cap to 50 events


def _format_rollout(r: dict, transcripts_dir: Path) -> str:
    diag = r.get("diagnostics", {}) or {}
    fail_tags = diag.get("failure_tags") or []
    notes = (r.get("swarm_notes") or "")[:300]
    bk = r.get("swarm_breakdown", {}) or {}
    return f"""
=== Rollout: task={r['task_id']} template={r.get('template_id')} ===
terminal_score: {r.get('terminal_score'):.3f}
swarm_policy_score: {r.get('swarm_policy_score'):.3f}
composite_reward: {r.get('composite_reward'):.3f}
swarm_breakdown: {bk}
swarm_judge_notes: {notes}
diagnostics.fatal: {diag.get('fatal')}
diagnostics.failure_tags: {fail_tags}
n_tool_calls: {r.get('n_tool_calls')}
n_subagent_calls: {r.get('n_subagent_calls', 0)}
n_reads: {r.get('n_reads')}
n_files_written: {r.get('n_files_written')}
--- trajectory summary ---
{_trajectory_summary(r.get('transcript_path', ''))}
"""


def evolve(current_skill: str, rollouts: list[dict], transcripts_dir: Path,
           api_key: str, model: str = "deepseek-v4-pro",
           base_url: str = "https://api.deepseek.com/v1") -> str:
    rollouts_sorted = sorted(rollouts, key=lambda r: r.get("composite_reward", 0.0))
    failures = rollouts_sorted[:3]
    successes = rollouts_sorted[-3:]

    failure_blob = "\n\n".join(_format_rollout(r, transcripts_dir) for r in failures)
    success_blob = "\n\n".join(_format_rollout(r, transcripts_dir) for r in successes)

    prompt = f"""You are revising a "skill prompt" given to a STRONG Qwen3-14B Lead agent doing
meeting-transcript analysis. The Lead is CAPABLE: it can read, reason, and write a
high-quality deliverable itself. It also has a sub-agent helper (a Qwen3-4B LoRA) served via
`exec subagent.sh "<task description>" <source_file> [<offset>] [<length>]`. The sub-agent
reads the source and returns ONLY its final text as exec stdout (its internal reasoning is hidden).

CRITICAL — do NOT make this a pure relay. The 4B sub is WEAKER than the 14B Lead, so if the
Lead just copies the sub's output verbatim, deliverable quality is capped at the 4B level
(this is exactly why council/vote tasks score near 0). The correct division of labor is:
  - Sub = RESEARCHER: reads the long transcript, returns raw structured facts + verbatim
    quotes + attribution, so the Lead's context stays lean.
  - Lead (14B) = ANALYST/AUTHOR: synthesizes the sub's findings, applies the task-specific
    structure, and WRITES the final deliverable itself. For vote/council tasks the Lead MUST
    build the explicit tally (who voted what, counts, outcome) and per-member positions itself
    — it may even read the transcript directly when the sub's extraction is insufficient.

Your job: read the CURRENT skill protocol, then read 3 FAILURE rollouts and 3 SUCCESS
rollouts under that protocol. Identify the dominant failure mode(s). Propose a REVISED
skill protocol that:
  - flips the protocol from "dispatcher copies sub verbatim" to "Lead analyzes & authors,
    sub researches" — let the 14B do the hard analysis, do not cap it at the sub
  - keeps the anti-fatal lessons that work: ONE sub dispatch, poll the backgrounded session
    (sessionId = two-word name after "session"), never re-dispatch, and a MANDATORY final
    step that WRITES the deliverable file (the only thing that scores)
  - directly addresses the most common observed failure (e.g., output_not_written /
    empty_response from running out of budget, or weak council/vote output)
  - is action-first and clearly structured; it does NOT need to be ultra-short — a 14B can
    follow a fuller protocol, but avoid rambling that wastes token budget before the file write

Return ONLY the revised skill protocol text. Do not include any preamble or explanation.

=== CURRENT SKILL PROTOCOL ===
{current_skill}

=== 3 FAILURE ROLLOUTS ===
{failure_blob}

=== 3 SUCCESS ROLLOUTS ===
{success_blob}

=== Now write the revised skill protocol (Lead analyzes & authors, sub researches; keep the mandatory file-write): ==="""

    body = json.dumps({
        "model": model,
        "messages": [
            {"role": "system", "content": "You are an expert at writing concise effective agent skill prompts. Return only the prompt text."},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.3,
        "max_tokens": 4096,
    }).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    msg = data.get("choices", [{}])[0].get("message", {})
    text = msg.get("content") or msg.get("reasoning_content") or ""
    return text.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--current-skill", required=True, help="Path to current skill protocol text")
    p.add_argument("--bench-dir", required=True, help="Bench output dir containing graded_swarm_trajectories.jsonl")
    p.add_argument("--out-skill", required=True, help="Where to write the evolved skill v_n+1")
    p.add_argument("--api-key", default=os.environ.get("DEEPSEEK_API_KEY"))
    p.add_argument("--model", default="deepseek-v4-pro")
    args = p.parse_args()

    if not args.api_key:
        raise SystemExit("DEEPSEEK_API_KEY required")

    current = Path(args.current_skill).read_text(encoding="utf-8")
    graded = Path(args.bench_dir) / "graded_swarm_trajectories.jsonl"
    rollouts = [json.loads(l) for l in graded.open(encoding="utf-8") if l.strip()]
    transcripts_dir = Path(args.bench_dir) / "transcripts"
    print(f"Loaded {len(rollouts)} rollouts from {graded}")

    new_skill = evolve(current, rollouts, transcripts_dir, args.api_key, model=args.model)
    Path(args.out_skill).write_text(new_skill + "\n", encoding="utf-8")
    print(f"Wrote evolved skill ({len(new_skill)} chars) → {args.out_skill}")
    print("\n--- EVOLVED SKILL ---")
    print(new_skill)


if __name__ == "__main__":
    main()
