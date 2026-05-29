#!/usr/bin/env python3
"""Randomize /tmp/pinchbench/<NNNN>/ index per trajectory.

Training data has hardcoded indices like /tmp/pinchbench/0070/agent_workspace/...
At inference, the index changes (e.g., 0100, 0200). Model adapts from prompt,
but having all training data use the same digits risks memorization.

Randomize each trajectory's index to a different 4-digit number so the model
learns the digits are arbitrary — only the file structure matters.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path


PINCH_RE = re.compile(r"/tmp/pinchbench/\d+/")


def deterministic_rand_index(task_id: str, run_id: str | None) -> str:
    """Stable 4-digit string from task+run identifiers (so reruns identical)."""
    key = f"{task_id}|{run_id or 'r0'}"
    h = hashlib.md5(key.encode()).hexdigest()
    n = int(h[:6], 16) % 9000 + 1000  # 1000..9999
    return f"{n:04d}"


def rewrite_paths(text: str, new_idx: str) -> str:
    return PINCH_RE.sub(f"/tmp/pinchbench/{new_idx}/", text)


def rewrite_trajectory(rec: dict) -> dict:
    new_idx = deterministic_rand_index(rec.get("task_id", ""), rec.get("run_id"))
    for m in rec.get("messages", []):
        if m.get("role") == "user":
            c = m.get("content")
            if isinstance(c, str):
                m["content"] = rewrite_paths(c, new_idx)
        elif m.get("role") == "assistant":
            for tc in m.get("tool_calls") or []:
                args = tc.get("arguments") or {}
                if isinstance(args, dict):
                    for k, v in list(args.items()):
                        if isinstance(v, str):
                            args[k] = rewrite_paths(v, new_idx)
        elif m.get("role") == "tool":
            c = m.get("content")
            if isinstance(c, str):
                m["content"] = rewrite_paths(c, new_idx)
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(args.input) as f, out_path.open("w") as g:
        for line in f:
            rec = json.loads(line)
            rec = rewrite_trajectory(rec)
            g.write(json.dumps(rec, ensure_ascii=False) + "\n")
            n += 1
    print(f"normalized {n} records → {out_path}")


if __name__ == "__main__":
    main()
