#!/usr/bin/env python3
"""Mock trajectory gateway for jiuwenclaw RLOnlineRail.

Receives rail-v1 batches at POST /v1/gateway/upload/batch, appends each
PerTurnSample as a JSONL line keyed by trajectory_id. Our verl agent_loop
reads the JSONL to assemble per-turn (prompt_ids, completion_token_ids,
logprobs) instead of reverse-engineering history.json.

Pairs with USE_RL_ONLINE_RAIL=1 + TRAJECTORY_GATEWAY_URL=http://127.0.0.1:9000
in start_jw_headless.sh.

Usage:
  python mock_trajectory_gateway.py --port 9000 --out /tmp/jw_rail_v1
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import time
from pathlib import Path

from aiohttp import web

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)
logger = logging.getLogger("rail_gateway")


class GatewayState:
    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.batches_received = 0
        self.samples_written = 0
        self._lock = asyncio.Lock()

    async def write_batch(self, payload: dict) -> int:
        trajectory_id = str(payload.get("trajectory_id") or "unknown")
        samples = payload.get("samples") or []
        async with self._lock:
            path = self.out_dir / f"{trajectory_id}.jsonl"
            with path.open("a", encoding="utf-8") as f:
                for sample in samples:
                    f.write(json.dumps(sample, ensure_ascii=False) + "\n")
                    self.samples_written += 1
            self.batches_received += 1
            return len(samples)


async def handle_upload(request: web.Request) -> web.Response:
    state: GatewayState = request.app["state"]
    try:
        payload = await request.json()
    except Exception as exc:
        return web.json_response({"ok": False, "error": f"bad json: {exc}"}, status=400)
    n = await state.write_batch(payload)
    logger.info(
        "batch trajectory=%s samples=%d total_batches=%d total_samples=%d",
        payload.get("trajectory_id"),
        n,
        state.batches_received,
        state.samples_written,
    )
    return web.json_response({"ok": True, "samples_received": n})


async def handle_health(request: web.Request) -> web.Response:
    state: GatewayState = request.app["state"]
    return web.json_response(
        {
            "ok": True,
            "batches_received": state.batches_received,
            "samples_written": state.samples_written,
            "out_dir": str(state.out_dir),
            "uptime_s": time.time() - request.app["started_at"],
        }
    )


def make_app(out_dir: Path) -> web.Application:
    app = web.Application(client_max_size=64 * 1024 * 1024)
    app["state"] = GatewayState(out_dir)
    app["started_at"] = time.time()
    app.add_routes(
        [
            web.post("/v1/gateway/upload/batch", handle_upload),
            web.get("/health", handle_health),
        ]
    )
    return app


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument(
        "--out",
        default="/tmp/jw_rail_v1",
        help="Directory where per-trajectory JSONL files are written",
    )
    args = ap.parse_args()
    out_dir = Path(args.out)
    logger.info("rail gateway listening on %s:%d -> %s", args.host, args.port, out_dir)
    web.run_app(make_app(out_dir), host=args.host, port=args.port, print=None)


if __name__ == "__main__":
    main()
