#!/usr/bin/env python3
"""Parse OC bench transcripts → unified messages JSONL.

Input dir layout (from scripts/benchmark.py):
  <bench_dir>/r{1,2,...}/<NNNN>_<model>.json           # final results
  <bench_dir>/r{1,2,...}/<NNNN>_transcripts/task_<id>.jsonl

OC message events have shape:
  {"type": "message", "message": {"role": "user"|"assistant"|"toolResult", "content": [...]}}

We emit one record per (run, task) into output JSONL with:
  - task_id, run_id, score (from results json)
  - messages: list in canonical OpenAI-ish form
      user:      {role: user, content: str}
      assistant: {role: assistant, content: str|None, reasoning: str|None, tool_calls: [...]}
      tool:      {role: tool, tool_call_id, name, content: str}
"""
import argparse
import glob
import json
import os
import re
from pathlib import Path


def _items_to_text(items):
    parts = []
    for it in items or []:
        if it.get("type") == "text":
            parts.append(it.get("text", ""))
    return "".join(parts)


def _content_to_tool_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _items_to_text(content)
    return json.dumps(content, ensure_ascii=False)


def parse_transcript(path: Path):
    msgs = []
    final_workspace_files = None
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        t = e.get("type")
        if t == "custom":
            # Sometimes final workspace listing is stashed here
            pl = e.get("payload") or {}
            if isinstance(pl, dict) and "workspace_files" in pl:
                final_workspace_files = pl.get("workspace_files")
            continue
        if t != "message":
            continue
        m = e.get("message") or {}
        role = m.get("role")
        content = m.get("content")
        if role == "user":
            msgs.append({"role": "user", "content": _items_to_text(content)})
        elif role == "assistant":
            text_parts = []
            think_parts = []
            tool_calls = []
            for it in content or []:
                it_t = it.get("type")
                if it_t == "text":
                    text_parts.append(it.get("text", ""))
                elif it_t == "thinking":
                    think_parts.append(it.get("thinking", ""))
                elif it_t == "toolCall":
                    tool_calls.append({
                        "id": it.get("id"),
                        "name": it.get("name"),
                        "arguments": it.get("arguments", {}),
                    })
            rec = {"role": "assistant"}
            txt = "".join(text_parts).strip()
            if txt:
                rec["content"] = txt
            else:
                rec["content"] = None
            think = "".join(think_parts).strip()
            if think:
                rec["reasoning"] = think
            if tool_calls:
                rec["tool_calls"] = tool_calls
            msgs.append(rec)
        elif role == "toolResult":
            msgs.append({
                "role": "tool",
                "tool_call_id": m.get("toolCallId"),
                "name": m.get("toolName"),
                "content": _content_to_tool_text(content),
            })
        else:
            # ignore system / unknown
            pass
    return msgs, final_workspace_files


def load_scores(results_json: Path):
    """Return {task_id: score_float_0to1}."""
    out = {}
    try:
        d = json.loads(results_json.read_text())
    except Exception:
        return out
    eff = d.get("efficiency") or {}
    for p in eff.get("per_task") or []:
        tid = p.get("task_id")
        s = p.get("score")
        if tid is not None and s is not None:
            out[tid] = float(s)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dirs", nargs="+", required=True,
                    help="One or more bench output dirs (each contains r*/ subdirs)")
    ap.add_argument("--output", required=True, help="Output JSONL path")
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_runs = 0
    n_records = 0
    n_skipped = 0
    with out_path.open("w") as f:
        for bench_dir in args.bench_dirs:
            bench_dir = Path(bench_dir)
            for run_dir in sorted(bench_dir.glob("r*/")):
                run_id = run_dir.name
                # find results json (one per run dir)
                results = list(run_dir.glob("*.json"))
                if not results:
                    n_skipped += 1
                    continue
                results_json = results[0]
                stem = results_json.stem  # e.g. "0070_deepseek-v4-flash"
                m = re.match(r"^(\d+)_", stem)
                prefix = m.group(1) if m else None
                scores = load_scores(results_json)
                ts_dir = run_dir / f"{prefix}_transcripts" if prefix else None
                if ts_dir is None or not ts_dir.exists():
                    # find any *_transcripts dir
                    alt = list(run_dir.glob("*_transcripts"))
                    if not alt:
                        n_skipped += 1
                        continue
                    ts_dir = alt[0]
                for ts_file in sorted(ts_dir.glob("task_*.jsonl")):
                    task_id = ts_file.stem  # e.g. task_meeting_advisory_stakeholders
                    msgs, wsfiles = parse_transcript(ts_file)
                    score = scores.get(task_id)
                    rec = {
                        "task_id": task_id,
                        "bench_dir": str(bench_dir),
                        "run_id": run_id,
                        "score": score,
                        "messages": msgs,
                        "n_messages": len(msgs),
                    }
                    if wsfiles is not None:
                        rec["workspace_files"] = wsfiles
                    f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    n_records += 1
                n_runs += 1

    print(f"runs: {n_runs}  records: {n_records}  skipped_runs: {n_skipped}")
    print(f"output: {out_path}")


if __name__ == "__main__":
    main()
