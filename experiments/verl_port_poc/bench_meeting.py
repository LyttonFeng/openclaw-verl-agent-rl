#!/usr/bin/env python3
"""Bench Qwen3-4B (base or +LoRA adapter) on meeting single-turn val tasks.

Reads /workspace/verl_port/data/meeting_inline_val.parquet, sends each row's
prompt to a running OpenAI-compatible API (vLLM serve), scores the response with
the same reward function used during training, and prints per-task + average.

Args:
  --base-url   OpenAI-compatible endpoint (default: http://127.0.0.1:8000/v1)
  --model      model name reported by the server (default: Qwen/Qwen3-4B)
  --max-tokens response length cap (default: 2048)

Env:
  DEEPSEEK_API_KEY  required for LLM judge
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd
import requests


def call_vllm(base_url: str, model: str, prompt_msgs: list, max_tokens: int) -> str:
    payload = {
        "model": model,
        "messages": prompt_msgs,
        "max_tokens": max_tokens,
        "temperature": 0.0,
    }
    r = requests.post(f"{base_url.rstrip('/')}/chat/completions", json=payload, timeout=600)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"] or ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", default="/workspace/verl_port/data/meeting_inline_val.parquet")
    ap.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--repo", default="/workspace/openclaw-verl-agent-rl")
    ap.add_argument("--out", default="/workspace/verl_port/bench_results.json")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(args.repo) / "rewards"))
    os.environ.setdefault("MEETING_TASKS_DIR", str(Path(args.repo) / "pinchbench_tasks" / "meeting_analysis"))
    from meeting_reward_single_turn import compute_score

    df = pd.read_parquet(args.parquet)
    print(f"bench {len(df)} tasks against {args.model} @ {args.base_url}")

    results = []
    for i, row in df.iterrows():
        prompt = list(row["prompt"])
        extra = dict(row["extra_info"])
        task_id = extra.get("task_id", row["reward_model"]["ground_truth"])
        ground_truth = row["reward_model"]["ground_truth"]
        print(f"\n[{i+1}/{len(df)}] {task_id}  (prompt {extra.get('transcript_chars', '?')} chars)")
        t0 = time.time()
        response = call_vllm(args.base_url, args.model, prompt, args.max_tokens)
        gen_s = time.time() - t0
        print(f"  generated {len(response)} chars in {gen_s:.1f}s")
        t0 = time.time()
        score = compute_score("pinchbench/meeting_analysis", response, ground_truth, extra)
        score_s = time.time() - t0
        print(f"  score={score['score']:.3f}  auto={score['automated_score']:.3f}  judge={score['judge_score']:.3f}  quality={score['quality_passed']} ({score_s:.1f}s)")
        results.append({
            "task_id": task_id,
            "response_chars": len(response),
            "gen_seconds": gen_s,
            "score_seconds": score_s,
            **score,
        })

    mean_score = sum(r["score"] for r in results) / len(results) if results else 0.0
    mean_auto = sum(r["automated_score"] for r in results) / len(results) if results else 0.0
    mean_judge = sum(r["judge_score"] for r in results) / len(results) if results else 0.0
    n_pass_quality = sum(1 for r in results if r["quality_passed"])

    summary = {
        "model": args.model,
        "n_tasks": len(results),
        "mean_score": mean_score,
        "mean_automated": mean_auto,
        "mean_judge": mean_judge,
        "n_quality_passed": n_pass_quality,
        "per_task": results,
    }
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"\n=== summary ===")
    print(f"  mean_score={mean_score:.3f}  auto={mean_auto:.3f}  judge={mean_judge:.3f}  quality_passed={n_pass_quality}/{len(results)}")
    print(f"  → {args.out}")


if __name__ == "__main__":
    main()
