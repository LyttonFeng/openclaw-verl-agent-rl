#!/usr/bin/env python3
"""Extract SFT records from swarm bench dirs (graded_swarm_trajectories.jsonl).

Unlike 1_extract_oc_transcript.py (which expects the old benchmark.py r*/ layout),
this reads the swarm-rollout layout:
  <bench_dir>/graded_swarm_trajectories.jsonl   # rows with terminal_score + transcript_path
  <bench_dir>/transcripts/task_<id>__TEMPLATE__rN.jsonl

Emits the same downstream record format consumed by 2_filter_quality.py:
  {task_id, score, bench_dir, run_id, messages:[...], n_messages}

Usage:
  python3 0_extract_swarm.py --bench-dirs D1 [D2 ...] --output raw.jsonl \
      [--only-tasks t1 t2 ...] [--min-score 0.0]
"""
import argparse
import json
from pathlib import Path


def _items_to_text(items):
    return "".join(it.get("text", "") for it in (items or []) if it.get("type") == "text")


def _content_to_tool_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return _items_to_text(content)
    return json.dumps(content, ensure_ascii=False)


def parse_transcript(path: Path):
    msgs = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("type") != "message":
            continue
        m = e.get("message") or {}
        role = m.get("role")
        content = m.get("content")
        if role == "user":
            msgs.append({"role": "user", "content": _items_to_text(content)})
        elif role == "assistant":
            text_parts, think_parts, tool_calls = [], [], []
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
            rec["content"] = txt if txt else None
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
    return msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench-dirs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--only-tasks", nargs="*", default=None)
    ap.add_argument("--min-score", type=float, default=0.0)
    args = ap.parse_args()

    only = set(args.only_tasks) if args.only_tasks else None
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_rows = n_emit = n_missing = 0
    with out_path.open("w") as f:
        for bd in args.bench_dirs:
            bd = Path(bd)
            graded = bd / "graded_swarm_trajectories.jsonl"
            if not graded.is_file():
                continue
            for line in graded.read_text().splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                n_rows += 1
                tid = r.get("task_id")
                score = r.get("terminal_score")
                tp = r.get("transcript_path")
                if only and tid not in only:
                    continue
                if score is None or score < args.min_score:
                    continue
                if not tp or not Path(tp).is_file():
                    n_missing += 1
                    continue
                msgs = parse_transcript(Path(tp))
                rec = {
                    "task_id": tid,
                    "bench_dir": str(bd),
                    "run_id": f"{r.get('template_id')}_r{r.get('resp_idx')}",
                    "score": float(score),
                    "messages": msgs,
                    "n_messages": len(msgs),
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                n_emit += 1

    print(f"rows: {n_rows}  emitted: {n_emit}  missing_transcript: {n_missing}")
    print(f"output: {out_path}")


if __name__ == "__main__":
    main()
