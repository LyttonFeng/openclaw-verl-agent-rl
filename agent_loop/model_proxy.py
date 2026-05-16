"""
ModelProxy: HTTP reverse proxy that intercepts OpenClaw's LLM requests
and forwards them to veRL's vLLM inference engine.

Architecture:
  OpenClaw agent → POST /v1/chat/completions → ModelProxy (aiohttp)
  ModelProxy → asyncio.Queue → OpenClawAgentLoop → veRL server_manager.generate()
  veRL response → ModelProxy → OpenAI-format SSE stream → OpenClaw agent

Based on veRL's SWE-Agent recipe ModelProxy pattern.
"""

from __future__ import annotations

import asyncio
import errno
import json
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Optional

from aiohttp import web
from aiohttp.client_exceptions import ClientConnectionResetError

logger = logging.getLogger(__name__)


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; ignoring", name, raw)
        return None


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return None
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r; ignoring", name, raw)
        return None


def _is_client_disconnect(exc: BaseException) -> bool:
    """True when the HTTP peer closed the connection (tunnel timeout, agent cancel, etc.)."""
    if isinstance(exc, (ClientConnectionResetError, ConnectionResetError, BrokenPipeError)):
        return True
    if isinstance(exc, OSError) and exc.errno in (errno.EPIPE, errno.ECONNRESET, errno.ECONNABORTED):
        return True
    return False


@dataclass
class ModelRequest:
    """A single LLM request from OpenClaw, waiting for veRL to generate."""

    request_id: str
    messages: list[dict[str, Any]]
    temperature: float = 0.7
    top_p: Optional[float] = None
    top_k: Optional[int] = None
    max_tokens: int = 4096
    tools: Optional[list[dict]] = None
    tool_choice: Optional[Any] = None
    received_at: float = field(default_factory=time.time)
    response_event: asyncio.Event = field(default_factory=asyncio.Event)
    response_text: Optional[str] = None
    response_tool_calls: Optional[list[dict]] = None
    response_error: Optional[str] = None
    response_usage: Optional[dict] = None
    finish_reason: str = "stop"
    stream_queue: asyncio.Queue[dict[str, Any] | None] = field(default_factory=asyncio.Queue)


class ModelProxy:
    """HTTP proxy server that intercepts OpenClaw's LLM calls.

    Supports both streaming (SSE) and non-streaming responses.
    OpenClaw always sends stream=true, so streaming is the primary path.

    Lifecycle:
        proxy = ModelProxy()
        await proxy.start()       # binds to ephemeral port
        ...                       # OpenClaw sends requests to proxy.port
        req = await proxy.get_request()    # dequeue one request
        await proxy.send_response(req)     # unblock the HTTP handler
        ...
        await proxy.stop()
    """

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 0,
        timeout: float = 600.0,
        live_stream: bool = False,
    ):
        self.host = host
        self.port = port
        self.timeout = timeout
        self.live_stream = live_stream
        self._queue: asyncio.Queue[ModelRequest] = asyncio.Queue()
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

    async def start(self) -> int:
        self._app = web.Application()
        self._app.router.add_post("/v1/chat/completions", self._handle_chat_completion)
        self._app.router.add_get("/v1/models", self._handle_list_models)
        self._app.router.add_get("/health", self._handle_health)

        self._runner = web.AppRunner(self._app, access_log=None)
        await self._runner.setup()
        self._site = web.TCPSite(self._runner, self.host, self.port)
        await self._site.start()

        sockets = self._site._server.sockets  # type: ignore[union-attr]
        if sockets:
            self.port = sockets[0].getsockname()[1]

        logger.info("ModelProxy listening on %s:%d", self.host, self.port)
        return self.port

    async def stop(self) -> None:
        if self._runner:
            await self._runner.cleanup()
        logger.info("ModelProxy stopped")

    async def get_request(self, timeout: Optional[float] = None) -> ModelRequest:
        t = timeout or self.timeout
        return await asyncio.wait_for(self._queue.get(), timeout=t)

    async def send_response(self, req: ModelRequest) -> None:
        req.response_event.set()

    async def send_stream_chunk(
        self,
        req: ModelRequest,
        delta: dict[str, Any],
        finish_reason: str | None = None,
    ) -> None:
        await req.stream_queue.put({"delta": delta, "finish_reason": finish_reason})

    async def finish_stream(self, req: ModelRequest) -> None:
        await req.stream_queue.put(None)

    async def drain(self) -> None:
        """Drain any pending requests with error responses."""
        while not self._queue.empty():
            try:
                req = self._queue.get_nowait()
                req.response_error = "proxy shutting down"
                req.response_event.set()
            except asyncio.QueueEmpty:
                break

    # ── HTTP handlers ──

    async def _handle_chat_completion(self, request: web.Request) -> web.Response:
        try:
            body = await request.json()
        except json.JSONDecodeError:
            return web.json_response(
                {"error": {"message": "invalid JSON"}}, status=400
            )

        env_temperature = _env_float("PINCHBENCH_MODEL_TEMPERATURE")
        env_top_p = _env_float("PINCHBENCH_MODEL_TOP_P")
        env_top_k = _env_int("PINCHBENCH_MODEL_TOP_K")
        body_temperature = body.get("temperature") if "temperature" in body else None
        body_top_p = body.get("top_p")
        body_top_k = body.get("top_k")
        req = ModelRequest(
            request_id=f"proxy-{uuid.uuid4().hex[:12]}",
            messages=body.get("messages", []),
            temperature=env_temperature if env_temperature is not None else (body_temperature if body_temperature is not None else 0.7),
            top_p=env_top_p if env_top_p is not None else body_top_p,
            top_k=env_top_k if env_top_k is not None else body_top_k,
            max_tokens=body.get("max_tokens", body.get("max_completion_tokens", 4096)),
            tools=body.get("tools"),
            tool_choice=body.get("tool_choice"),
        )

        await self._queue.put(req)
        logger.info(
            "Queued request %s (%d messages, stream=%s, tools=%d)",
            req.request_id, len(req.messages), body.get("stream", False),
            len(body.get("tools") or []),
        )

        is_stream = body.get("stream", False)
        model_name = body.get("model", "verl-proxy")

        if is_stream and self.live_stream:
            return await self._live_stream_response(request, req, model_name)
        if is_stream:
            return await self._delayed_stream_response(request, req, model_name)

        try:
            await asyncio.wait_for(req.response_event.wait(), timeout=self.timeout)
        except asyncio.TimeoutError:
            return web.json_response(
                {"error": {"message": "generation timeout"}}, status=504
            )

        if req.response_error:
            return web.json_response(
                {"error": {"message": req.response_error}}, status=500
            )

        return self._json_response(req, model_name)

    async def _delayed_stream_response(
        self, http_request: web.Request, req: ModelRequest, model: str,
    ) -> web.Response:
        """Prepare SSE immediately, then send buffered generation when ready.

        veRL generation is not token-streaming, but OpenClaw expects activity on
        streaming requests. Sending the role chunk before generation prevents
        assistant idle timeouts without changing the final OpenAI-compatible
        tool/content chunks.
        """
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        created = int(time.time())

        async def _send(data: dict) -> None:
            payload = f"data: {json.dumps(data)}\n\n"
            await resp.write(payload.encode())

        def _chunk(delta: dict, finish_reason: str | None = None) -> dict:
            return {
                "id": req.request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }

        async def _keepalive() -> None:
            while not req.response_event.is_set():
                await asyncio.sleep(10)
                if not req.response_event.is_set():
                    await resp.write(b": keepalive\n\n")

        try:
            await resp.prepare(http_request)
            await _send(_chunk({"role": "assistant"}))
            keepalive_task = asyncio.create_task(_keepalive())
            try:
                await asyncio.wait_for(req.response_event.wait(), timeout=self.timeout)
            finally:
                keepalive_task.cancel()
                try:
                    await keepalive_task
                except asyncio.CancelledError:
                    pass

            if req.response_error:
                await _send(_chunk({"content": req.response_error}, finish_reason="stop"))
            elif req.response_tool_calls:
                content = req.response_text or ""
                if content:
                    await _send(_chunk({"content": content}))
                for i, tc in enumerate(req.response_tool_calls):
                    func = tc.get("function", {})
                    await _send(_chunk({"tool_calls": [{
                        "index": i,
                        "id": tc.get("id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": func.get("name", ""),
                            "arguments": func.get("arguments", "{}"),
                        },
                    }]}))
                await _send(_chunk({}, finish_reason="tool_calls"))
            else:
                content = req.response_text or ""
                if content:
                    await _send(_chunk({"content": content}))
                await _send(_chunk({}, finish_reason=req.finish_reason or "stop"))

            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp
        except asyncio.TimeoutError:
            try:
                await _send(_chunk({"content": "generation timeout"}, finish_reason="stop"))
                await resp.write(b"data: [DONE]\n\n")
                await resp.write_eof()
                return resp
            except BaseException:
                raise
        except asyncio.CancelledError:
            raise
        except BaseException as e:
            if not _is_client_disconnect(e):
                raise
            logger.warning(
                "SSE peer disconnected before delayed stream finished (request_id=%s): %s",
                req.request_id, e,
            )
            if getattr(resp, "prepared", False):
                try:
                    await resp.write_eof()
                except BaseException:
                    pass
                return resp
            return web.Response(status=499, text="Client Closed Request")

    async def _live_stream_response(
        self, http_request: web.Request, req: ModelRequest, model: str,
    ) -> web.Response:
        """Stream chunks as the upstream model produces them.

        This is used by ECS E2E testing where OpenClaw expects token/tool-call
        SSE activity while the model is generating. The default ModelProxy path
        remains buffered for veRL training compatibility.
        """
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )
        created = int(time.time())

        async def _send(data: dict) -> None:
            payload = f"data: {json.dumps(data)}\n\n"
            await resp.write(payload.encode())

        def _chunk(delta: dict, finish_reason: str | None = None) -> dict:
            return {
                "id": req.request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }

        try:
            await resp.prepare(http_request)
            while True:
                item = await asyncio.wait_for(req.stream_queue.get(), timeout=self.timeout)
                if item is None:
                    break
                await _send(_chunk(item.get("delta") or {}, item.get("finish_reason")))
            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp
        except asyncio.CancelledError:
            raise
        except BaseException as e:
            if not _is_client_disconnect(e):
                raise
            logger.warning(
                "SSE peer disconnected before live stream finished (request_id=%s): %s",
                req.request_id, e,
            )
            if getattr(resp, "prepared", False):
                try:
                    await resp.write_eof()
                except BaseException:
                    pass
                return resp
            return web.Response(status=499, text="Client Closed Request")

    def _json_response(self, req: ModelRequest, model: str) -> web.Response:
        message: dict[str, Any] = {"role": "assistant", "content": req.response_text or ""}
        if req.response_tool_calls:
            message["tool_calls"] = req.response_tool_calls

        body = {
            "id": req.request_id,
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": req.finish_reason}],
            "usage": req.response_usage or {
                "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0,
            },
        }
        return web.json_response(body)

    async def _stream_response(
        self, http_request: web.Request, req: ModelRequest, model: str,
    ) -> web.Response:
        """Return an SSE stream matching the OpenAI chat.completions streaming format.

        Follows the exact OpenAI streaming spec:
        - When tool_calls present: role chunk -> tool_call chunks (name+args split) -> finish(tool_calls)
        - When no tool_calls: role chunk -> content chunk -> finish(stop)
        - tool_calls and content are never mixed in the same delta
        """
        resp = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "X-Accel-Buffering": "no",
            },
        )

        created = int(time.time())
        has_tool_calls = bool(req.response_tool_calls)

        async def _send(data: dict) -> None:
            payload = f"data: {json.dumps(data)}\n\n"
            logger.debug("[SSE %s] %s", req.request_id, payload.strip()[:300])
            await resp.write(payload.encode())

        def _chunk(delta: dict, finish_reason: str | None = None) -> dict:
            c: dict = {
                "id": req.request_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
            }
            return c

        try:
            await resp.prepare(http_request)

            # Chunk 1: role
            await _send(_chunk({"role": "assistant"}))

            if has_tool_calls:
                for i, tc in enumerate(req.response_tool_calls):
                    func = tc.get("function", {})
                    args_str = func.get("arguments", "{}")
                    # Send each tool call as a single chunk with all fields.
                    # OpenClaw's streaming parser starts a new toolCall block
                    # when it sees a chunk with an id.
                    await _send(_chunk({"tool_calls": [{
                        "index": i,
                        "id": tc.get("id", f"call_{i}"),
                        "type": "function",
                        "function": {
                            "name": func.get("name", ""),
                            "arguments": args_str,
                        },
                    }]}))

                await _send(_chunk({}, finish_reason="tool_calls"))
            else:
                content = req.response_text or ""
                if content:
                    await _send(_chunk({"content": content}))
                await _send(_chunk({}, finish_reason="stop"))

            await resp.write(b"data: [DONE]\n\n")
            await resp.write_eof()
            return resp
        except asyncio.CancelledError:
            raise
        except BaseException as e:
            if not _is_client_disconnect(e):
                raise
            logger.warning(
                "SSE peer disconnected before stream finished (request_id=%s): %s",
                req.request_id, e,
            )
            if getattr(resp, "prepared", False):
                try:
                    await resp.write_eof()
                except BaseException:
                    pass
                return resp
            # prepare() failed (e.g. client gone before headers sent)
            return web.Response(status=499, text="Client Closed Request")

    async def _handle_list_models(self, request: web.Request) -> web.Response:
        return web.json_response({
            "object": "list",
            "data": [
                {
                    "id": "verl-proxy",
                    "object": "model",
                    "owned_by": "verl",
                }
            ],
        })

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.json_response({"status": "ok"})
