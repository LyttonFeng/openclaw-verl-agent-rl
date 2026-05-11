"""LoRA hot-sync from veRL checkpoint dir → jiuwenclaw vLLM.

Two roles:

  1. Library — import `sync_lora_to_jiuwen(...)` from a veRL trainer
     callback (e.g. on_save_checkpoint) to push the freshly-saved LoRA
     adapter into the jiuwenclaw stack's vLLM via /v1/load_lora_adapter.

  2. CLI — `python -m experiments.verl_port_poc.jiuwen_lora_sync
     --ckpt-dir /workspace/verl_port/ckpt_jw/global_step_5/actor/lora_adapter
     --lora-name jw-step5` for manual / cron-driven syncs while you're
     iterating on Tier 4 architecture.

Architectural context (read me before wiring this in!):
  This module supports **Path B**: jiuwenclaw runs its own vLLM, we
  periodically push new LoRA versions to it. With 2 GPUs total, Path B
  forces jiuwenclaw vLLM to fight veRL's hybrid-engine vLLM for memory —
  the OpenClaw v3 GRPO run on 2026-05-11 was OOM-killed exactly this way.
  Path B is only viable when GPUs are split (e.g. jiuwenclaw on GPU 0,
  veRL on GPU 1) AND each side fits in one card, OR you have ≥4 GPUs.

  **Path A** (preferred): jiuwenclaw's gateway → veRL's vLLM directly.
  veRL's hybrid engine already syncs FSDP weights to its own vLLM each
  rollout — no separate sync needed. This module is unused in Path A.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

DEFAULT_VLLM_BASE = os.environ.get("JIUWEN_VLLM_BASE", "http://127.0.0.1:614")


def _http_post_json(url: str, body: dict, timeout: float = 30.0) -> tuple[int, str]:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", errors="replace") if e.fp else str(e)


def _http_get_json(url: str, timeout: float = 10.0) -> tuple[int, str]:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as resp:
            return resp.status, resp.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as e:
        return e.code, str(e)


def sync_lora_to_jiuwen(
    lora_path: str | os.PathLike,
    lora_name: str,
    vllm_base: str = DEFAULT_VLLM_BASE,
    unload_existing: bool = True,
    timeout: float = 60.0,
    post_json: callable = _http_post_json,  # injected for tests
    get_json: callable = _http_get_json,    # injected for tests
) -> dict:
    """Push a LoRA adapter to jiuwenclaw's vLLM.

    Args:
      lora_path: directory containing adapter_config.json + adapter_model.safetensors.
        Path must be readable by the vLLM process (same host or shared FS).
      lora_name: unique adapter id; the jiuwenclaw gateway should use this as
        its `model` for /v1/chat/completions calls to take effect.
      vllm_base: e.g. http://127.0.0.1:614
      unload_existing: if a LoRA with the same name is already loaded, unload
        it first (vLLM rejects re-load of same name). Set False if you use
        version-stamped names per step.

    Returns:
      {"status": "ok"|"error", "lora_name": ..., "loaded_at": epoch_ts,
       "models": [...], "error": ?}

    Raises only on programmer errors (bad path); HTTP failures are reported
    in the returned dict so callers can decide to retry.
    """
    lp = Path(lora_path).resolve()
    if not (lp / "adapter_config.json").exists():
        raise FileNotFoundError(f"adapter_config.json missing under {lp}")

    result: dict = {"status": "error", "lora_name": lora_name, "models": []}

    if unload_existing:
        # Best-effort unload; ignore failures (adapter may not be loaded yet).
        code, body = post_json(
            f"{vllm_base}/v1/unload_lora_adapter",
            {"lora_name": lora_name},
            timeout,
        )
        logger.debug("unload %s -> %d %s", lora_name, code, body[:200])

    code, body = post_json(
        f"{vllm_base}/v1/load_lora_adapter",
        {"lora_name": lora_name, "lora_path": str(lp)},
        timeout,
    )
    if code != 200:
        result["error"] = f"load_lora_adapter HTTP {code}: {body[:500]}"
        return result

    # Verify it shows in /v1/models
    code2, body2 = get_json(f"{vllm_base}/v1/models", timeout=10.0)
    if code2 != 200:
        result["error"] = f"/v1/models HTTP {code2}: {body2[:200]}"
        return result
    try:
        models_doc = json.loads(body2)
        result["models"] = [m["id"] for m in models_doc.get("data", [])]
    except json.JSONDecodeError:
        result["error"] = f"non-JSON /v1/models: {body2[:200]}"
        return result

    if lora_name not in result["models"]:
        result["error"] = f"loaded but {lora_name!r} absent from /v1/models: {result['models']}"
        return result

    result["status"] = "ok"
    result["loaded_at"] = time.time()
    return result


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--ckpt-dir", required=True,
                    help="veRL LoRA dir (contains adapter_config.json)")
    ap.add_argument("--lora-name", required=True,
                    help="Unique name to register in jiuwenclaw vLLM")
    ap.add_argument("--vllm-base", default=DEFAULT_VLLM_BASE)
    ap.add_argument("--no-unload", action="store_true",
                    help="Skip unload-before-load (use for version-stamped names)")
    ap.add_argument("--timeout", type=float, default=60.0)
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    res = sync_lora_to_jiuwen(
        lora_path=args.ckpt_dir,
        lora_name=args.lora_name,
        vllm_base=args.vllm_base,
        unload_existing=not args.no_unload,
        timeout=args.timeout,
    )
    print(json.dumps(res, indent=2))
    return 0 if res["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
