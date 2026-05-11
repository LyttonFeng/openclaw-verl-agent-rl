"""Smoke test: end-to-end JiuwenClawAgentLoop with optional LoRA hot-load.

PREREQUISITES (run on the GPU pod, not locally):
  1. jiuwenclaw stack up and serving:
       bash /root/jiuwen_work/start_jw_pod.sh
     verify: curl -sf http://127.0.0.1:614/v1/models | head
  2. (optional) hot-load a veRL LoRA into the stack's vLLM:
       LORA_NAME=step16 LORA_PATH=/workspace/verl_port/ckpt_openclaw/global_step_16/actor/lora_adapter
       curl -s -X POST http://127.0.0.1:614/v1/load_lora_adapter \
         -H 'Content-Type: application/json' \
         -d "{\"lora_name\":\"$LORA_NAME\",\"lora_path\":\"$LORA_PATH\"}"
     verify: curl -s http://127.0.0.1:614/v1/models | grep "$LORA_NAME"

USAGE:
  python3 experiments/verl_port_poc/smoke_jiuwenclaw_agent_loop.py [--task-id TID] [--prompt TEXT]

What this verifies:
  - WS round-trip (chat.send → chat.final) completes within timeout
  - history.json is written under JIUWENCLAW_DATA_DIR
  - _history_to_messages + _build_response_from_history produce non-empty
    response_ids with mixed mask values (1 for assistant, 0 for tool results)
  - AgentLoopOutput is assembled cleanly

This does NOT exercise:
  - veRL training loop integration
  - LoRA weight sync from veRL hybrid engine → jiuwenclaw vLLM
    (we rely on curl hot-load above; the in-training sync hook is future work)
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))


def _norm_ids(out):
    if isinstance(out, dict) or hasattr(out, "input_ids"):
        return list(out["input_ids"])
    return list(out)


async def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--task-id", default="smoke_test")
    ap.add_argument(
        "--prompt",
        default=(
            "Write a Python function that returns the Fibonacci number at index n. "
            "Save it to a file called fib.py."
        ),
    )
    ap.add_argument("--ws-url", default=os.environ.get("JIUWENCLAW_WS_URL", "ws://127.0.0.1:611/ws"))
    ap.add_argument("--data-root", default=os.environ.get("JIUWENCLAW_DATA_DIR", str(Path.home() / ".jiuwenclaw")))
    ap.add_argument("--timeout", type=float, default=300.0)
    ap.add_argument("--tokenizer", default="Qwen/Qwen3-4B")
    args = ap.parse_args()

    os.environ["JIUWENCLAW_WS_URL"] = args.ws_url
    os.environ["JIUWENCLAW_DATA_DIR"] = args.data_root
    os.environ["JIUWENCLAW_TIMEOUT"] = str(args.timeout)

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("[smoke] transformers not installed; install it on the pod first", file=sys.stderr)
        return 2
    try:
        import websockets  # noqa: F401
    except ImportError:
        print("[smoke] websockets not installed: pip install websockets", file=sys.stderr)
        return 2

    from jiuwenclaw_agent_loop import JiuwenClawAgentLoop, JiuwenWSConfig

    loop = JiuwenClawAgentLoop.__new__(JiuwenClawAgentLoop)
    loop.cfg = JiuwenWSConfig.from_env()
    loop.tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    extra_info = {"task_id": args.task_id, "workspace_files": []}
    prompt_messages = [{"role": "user", "content": args.prompt}]

    print(f"[smoke] ws_url={loop.cfg.ws_url} data_root={loop.cfg.data_root}")
    print(f"[smoke] task_id={args.task_id} timeout={loop.cfg.timeout_seconds}s")
    print(f"[smoke] prompt={args.prompt!r}")

    out = await loop.run(sampling_params={}, prompt=prompt_messages, extra_info=extra_info)

    print("\n[smoke] === AgentLoopOutput ===")
    print(f"  status            = {out.extra_fields.get('status')}")
    print(f"  timed_out         = {out.extra_fields.get('timed_out')}")
    print(f"  session_id        = {out.extra_fields.get('session_id')}")
    print(f"  history_path      = {out.extra_fields.get('history_path')}")
    print(f"  prompt_ids        = {len(out.prompt_ids)} tokens")
    print(f"  response_ids      = {len(out.response_ids)} tokens")
    print(f"  response_mask=1   = {sum(out.response_mask)} tokens")
    print(f"  response_mask=0   = {len(out.response_mask) - sum(out.response_mask)} tokens")
    print(f"  num_turns         = {out.num_turns}")

    # Sanity checks
    failures = []
    if out.extra_fields.get("status") != "success":
        failures.append(f"status={out.extra_fields.get('status')} (expected success)")
    if len(out.response_ids) == 0:
        failures.append("response_ids empty — history.json parse failed?")
    if len(out.response_ids) != len(out.response_mask):
        failures.append("response_ids / response_mask length mismatch")
    if out.num_turns == 0:
        failures.append("num_turns=0 — no assistant turns detected in history")
    if sum(out.response_mask) == 0:
        failures.append("no mask=1 tokens — all assistant tokens missed")

    if failures:
        print("\n[smoke] FAILED:")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\n[smoke] PASS")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
