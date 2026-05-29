#!/usr/bin/env python3
"""Merge consecutive read(path=X, ...) tool-call pairs in raw trajectories.

Many OC trajectories on long transcripts look like:
    assistant: think → tool_call(read path=X offset=0)
    tool:      result chunk 0
    assistant: think → tool_call(read path=X offset=200)
    tool:      result chunk 1
    ... 8-15 chunks ...

We merge runs into a single (assistant, tool) pair:
    assistant: think (from first turn) → tool_call(read path=X)
    tool:      "<concatenated chunks 0..N>"

Reduces message count drastically (92 msg → ~25 msg for council_votes),
which collapses total token count enough to fit under DROP_ABOVE=32000.

Only triggers when:
  - assistant has exactly 1 tool_call (no parallel calls)
  - tool name == "read"
  - same path as previous read in run
  - next assistant->tool pair follows immediately (no interleaved tool / user)
"""
import argparse
import json
from pathlib import Path


def is_pure_read(asst):
    """Assistant turn has exactly 1 read tool_call (any path)."""
    if asst.get("role") != "assistant":
        return False
    tcs = asst.get("tool_calls") or []
    if len(tcs) != 1:
        return False
    return tcs[0].get("name") == "read"


def read_path(asst):
    return asst["tool_calls"][0].get("arguments", {}).get("path")


def merge_one_traj(msgs):
    """Merge runs of consecutive read(path=X, ...) regardless of offset/limit.

    Strategy: when the assistant chains `read` calls to the *same file*, only
    the final cumulative content matters for downstream reasoning. We keep the
    first assistant turn (preserves intent + thinking) and concatenate every
    tool result in order, dropping the redundant interstitial assistant turns.
    """
    out = []
    i = 0
    merged_runs = 0
    saved_msgs = 0
    while i < len(msgs):
        m = msgs[i]
        if (
            is_pure_read(m)
            and i + 1 < len(msgs)
            and msgs[i + 1].get("role") == "tool"
        ):
            path = read_path(m)
            # Collect consecutive same-path read pairs (regardless of offset/limit)
            run = [(m, msgs[i + 1])]
            j = i + 2
            while (
                j + 1 < len(msgs)
                and is_pure_read(msgs[j])
                and read_path(msgs[j]) == path
                and msgs[j + 1].get("role") == "tool"
            ):
                run.append((msgs[j], msgs[j + 1]))
                j += 2
            if len(run) > 1:
                first_asst, _ = run[0]
                contents = []
                for k, (_, tool) in enumerate(run):
                    c = tool.get("content") or ""
                    if k == 0:
                        contents.append(c)
                    else:
                        contents.append(f"\n\n--- read continued (chunk {k}) ---\n\n{c}")
                merged_tool = dict(run[0][1])
                merged_tool["content"] = "".join(contents)
                out.append(first_asst)
                out.append(merged_tool)
                merged_runs += 1
                saved_msgs += (len(run) - 1) * 2
                i = j
                continue
        out.append(m)
        i += 1
    return out, merged_runs, saved_msgs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_in = 0
    n_runs = 0
    n_saved = 0
    msg_before = 0
    msg_after = 0

    with open(args.input) as f, out_path.open("w") as g:
        for line in f:
            r = json.loads(line)
            n_in += 1
            before = len(r["messages"])
            merged, runs, saved = merge_one_traj(r["messages"])
            r["messages"] = merged
            r["n_messages"] = len(merged)
            msg_before += before
            msg_after += len(merged)
            n_runs += runs
            n_saved += saved
            g.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"records: {n_in}")
    print(f"total msgs: {msg_before:,} → {msg_after:,} (saved {msg_before - msg_after:,})")
    print(f"merged runs: {n_runs}  msgs absorbed: {n_saved}")


if __name__ == "__main__":
    main()
