#!/usr/bin/env python3
"""Convert filtered trajectories → ChatML-style messages for Qwen3 SFT.

Output is JSONL where each line is {messages: [...]} ready for trainers that
consume OpenAI/ChatML format with tools. We collapse OC's tool_call/tool roles
into the standard format:
  user      → {role: "user", content: str}
  assistant → {role: "assistant", content: str, tool_calls?: [...]}
              Reasoning is wrapped in <think>...</think> prefix so Qwen3
              learns to emit thinking under the same chat template.
  tool      → {role: "tool", tool_call_id, content: str}

Qwen3 chat template expects `tool_calls` as a list of OpenAI-format objects:
  {"id": ..., "type": "function",
   "function": {"name": ..., "arguments": "<json string>"}}
"""
import argparse
import json
from pathlib import Path


def render_assistant(rec):
    parts = []
    reasoning = rec.get("reasoning")
    if reasoning:
        parts.append(f"<think>\n{reasoning.strip()}\n</think>")
    txt = rec.get("content") or ""
    if txt:
        parts.append(txt)
    content = "\n".join(parts) if parts else ""
    out = {"role": "assistant", "content": content}
    tcs = rec.get("tool_calls") or []
    if tcs:
        out["tool_calls"] = [
            {
                "id": tc.get("id"),
                "type": "function",
                "function": {
                    "name": tc.get("name"),
                    "arguments": json.dumps(tc.get("arguments") or {}, ensure_ascii=False),
                },
            }
            for tc in tcs
        ]
    return out


def convert(messages):
    out = []
    for m in messages:
        role = m["role"]
        if role == "user":
            out.append({"role": "user", "content": m.get("content", "")})
        elif role == "assistant":
            out.append(render_assistant(m))
        elif role == "tool":
            out.append({
                "role": "tool",
                "tool_call_id": m.get("tool_call_id"),
                "content": m.get("content", ""),
            })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--keep-reasoning", action="store_true", default=True,
                    help="Wrap assistant reasoning in <think>...</think> (default on)")
    ap.add_argument("--no-reasoning", dest="keep_reasoning", action="store_false")
    args = ap.parse_args()

    n_in = 0
    n_out = 0
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.input) as f, out_path.open("w") as g:
        for line in f:
            r = json.loads(line)
            n_in += 1
            msgs = r["messages"]
            if not args.keep_reasoning:
                # strip reasoning before convert
                for m in msgs:
                    if m.get("role") == "assistant":
                        m.pop("reasoning", None)
            chatml = convert(msgs)
            if not chatml:
                continue
            rec = {
                "task_id": r["task_id"],
                "run_id": r.get("run_id"),
                "score": r.get("score"),
                "messages": chatml,
            }
            g.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"in: {n_in}  out: {n_out}  → {out_path}")


if __name__ == "__main__":
    main()
