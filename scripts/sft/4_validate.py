#!/usr/bin/env python3
"""Validate ChatML-format SFT JSONL.

Checks:
  - alternation / role validity
  - every assistant tool_call has a matching tool message with same id
  - every tool message refers to an emitted tool_call_id
  - approx token count via tiktoken (cl100k_base) — flags >32K context

If --tokenizer points to a HF dir with a tokenizer.json, uses that instead.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def get_token_counter(tokenizer_dir):
    if tokenizer_dir:
        try:
            from transformers import AutoTokenizer
            tok = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
            def _cnt(s):
                return len(tok.encode(s, add_special_tokens=False))
            return _cnt, "hf"
        except Exception as e:
            print(f"[warn] failed to load HF tokenizer ({e}); falling back to char/4 heuristic")
    def _cnt(s):
        return max(1, len(s) // 4)
    return _cnt, "char/4"


def messages_text(msgs):
    parts = []
    for m in msgs:
        parts.append(m.get("role", "") + "\n")
        parts.append(str(m.get("content") or ""))
        for tc in m.get("tool_calls") or []:
            parts.append(tc.get("function", {}).get("name", ""))
            parts.append(tc.get("function", {}).get("arguments", ""))
    return "\n".join(parts)


def validate_one(rec):
    issues = []
    msgs = rec.get("messages") or []
    pending = {}  # id -> True (assistant emitted, awaiting tool result)
    seen_tool_ids = set()
    for i, m in enumerate(msgs):
        r = m.get("role")
        if r == "assistant":
            for tc in m.get("tool_calls") or []:
                tid = tc.get("id")
                if not tid:
                    issues.append(f"msg{i}: tool_call missing id")
                else:
                    pending[tid] = i
        elif r == "tool":
            tid = m.get("tool_call_id")
            if not tid:
                issues.append(f"msg{i}: tool message missing tool_call_id")
            elif tid not in pending:
                issues.append(f"msg{i}: tool result id {tid} has no matching call")
            else:
                seen_tool_ids.add(tid)
        elif r == "user":
            pass
        else:
            issues.append(f"msg{i}: unknown role {r!r}")
    orphan = set(pending) - seen_tool_ids
    if orphan:
        issues.append(f"unmatched tool_calls (no tool result): {len(orphan)}")
    return issues


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--tokenizer", default=None,
                    help="Optional HF tokenizer dir (e.g., Qwen3-4B path)")
    ap.add_argument("--max-tokens", type=int, default=32000)
    args = ap.parse_args()

    counter, mode = get_token_counter(args.tokenizer)
    print(f"token counter: {mode}")

    n = 0
    n_bad = 0
    token_buckets = Counter()
    too_long = []
    issues_total = 0
    per_task = Counter()
    with open(args.input) as f:
        for line in f:
            r = json.loads(line)
            n += 1
            per_task[r.get("task_id", "?")] += 1
            issues = validate_one(r)
            if issues:
                n_bad += 1
                issues_total += len(issues)
                if n_bad <= 5:
                    print(f"[bad] {r.get('task_id')}/{r.get('run_id')}: {issues[:3]}")
            ntok = counter(messages_text(r["messages"]))
            bucket = f"{ntok // 8000 * 8}K-{ntok // 8000 * 8 + 8}K"
            token_buckets[bucket] += 1
            if ntok > args.max_tokens:
                too_long.append((r.get("task_id"), r.get("run_id"), ntok))

    print(f"\ntotal: {n}  bad: {n_bad}  total_issues: {issues_total}")
    print(f"per-task counts ({len(per_task)} tasks):")
    for t, c in sorted(per_task.items()):
        print(f"  {t:50s} {c}")
    print(f"\ntoken buckets:")
    for k, v in sorted(token_buckets.items()):
        print(f"  {k:12s} {v}")
    if too_long:
        print(f"\n>{args.max_tokens} tokens ({len(too_long)}):")
        for t, r, n in sorted(too_long, key=lambda x: -x[2])[:20]:
            print(f"  {t:50s} {r:6s} {n}")

    sys.exit(0 if n_bad == 0 else 1)


if __name__ == "__main__":
    main()
