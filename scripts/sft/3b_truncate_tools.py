#!/usr/bin/env python3
"""Truncate oversized tool-result content in ChatML JSONL.

Many meeting transcripts are 50K+ chars; multiple `read` calls bloat traj
context far past Qwen3-4B's 32K. We collapse any tool message content
exceeding --max-tool-chars to head + "...[truncated N chars]..." + tail
so the assistant's downstream reasoning still has the surrounding context.

This is *not* a substitute for a real long-context model; trajectories that
remain over --drop-above-tokens after truncation are dropped entirely.
"""
import argparse
import json
from pathlib import Path


def truncate_tool(content: str, max_chars: int) -> str:
    if len(content) <= max_chars:
        return content
    head = max_chars // 2
    tail = max_chars - head
    return (
        content[:head]
        + f"\n\n...[truncated {len(content) - max_chars} chars]...\n\n"
        + content[-tail:]
    )


def approx_tokens(msgs):
    n = 0
    for m in msgs:
        n += len(m.get("content") or "") // 4 + 4
        for tc in m.get("tool_calls") or []:
            n += len(tc.get("function", {}).get("arguments", "")) // 4 + 4
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--max-tool-chars", type=int, default=8000,
                    help="Max chars per tool message before head+tail truncation (default 8000 ~= 2K tok)")
    ap.add_argument("--drop-above-tokens", type=int, default=24000,
                    help="Drop traj if approx tokens still exceed this after truncation (default 24K)")
    args = ap.parse_args()

    n_in = 0
    n_out = 0
    n_truncated_msgs = 0
    n_dropped = 0
    char_saved = 0

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.input) as f, out_path.open("w") as g:
        for line in f:
            r = json.loads(line)
            n_in += 1
            for m in r["messages"]:
                if m.get("role") == "tool":
                    c = m.get("content") or ""
                    if len(c) > args.max_tool_chars:
                        nc = truncate_tool(c, args.max_tool_chars)
                        char_saved += len(c) - len(nc)
                        m["content"] = nc
                        n_truncated_msgs += 1
            tk = approx_tokens(r["messages"])
            if tk > args.drop_above_tokens:
                n_dropped += 1
                continue
            g.write(json.dumps(r, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"in: {n_in}  out: {n_out}  dropped: {n_dropped}")
    print(f"tool messages truncated: {n_truncated_msgs}  chars saved: {char_saved:,}")


if __name__ == "__main__":
    main()
