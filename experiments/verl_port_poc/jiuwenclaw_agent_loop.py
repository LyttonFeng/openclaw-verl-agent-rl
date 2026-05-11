"""JiuwenClawAgentLoop — veRL AgentLoopBase subclass driving rollouts via JiuwenClaw WS.

STATUS: SKELETON ONLY — not smoke-tested. Drafted overnight after the cross-
runtime LoRA bench revealed that OpenClaw-trained LoRAs degrade on JiuwenClaw
(see docs/verl_port/07_jiuwenclaw_runtime.md). The fix is training-side runtime
parity, which means swapping veRL's rollout agent from OpenClawAgentLoop to a
JiuwenClaw-based one so RL fine-tuning learns the actual JiuwenClaw tool
distribution.

Adapted from:
  - colleague's `run_pinchbench_jiuwenclaw.py::_execute_task` (WS protocol)
  - existing `OpenClawAgentLoop.run()` (token mask, AgentLoopOutput build)
  - verl.experimental.agent_loop.AgentLoopBase API (veRL 0.8)

Outstanding work before this can run:
  1. **ModelProxy wiring**: jiuwenclaw stack starts its own vLLM (port 614),
     but RL training needs jiuwenclaw rollouts to use veRL's hybrid-engine
     vLLM (which has the currently-training LoRA). Two paths:
       (a) After each grad step: dump LoRA -> hot-load to jiuwenclaw vLLM via
           POST /v1/load_lora_adapter (cheap, ~1s sync).
       (b) ModelProxy aiohttp bridge: jiuwenclaw gateway -> ModelProxy ->
           verl.server_manager.generate (no separate vLLM). Cleaner but more
           code.
     Recommend (a) for MVP since pinchbench-skill already had this pattern.
  2. **Workspace isolation**: jiuwenclaw uses a shared `~/.jiuwenclaw/agent/
     jiuwenclaw_workspace/`. Concurrent rollouts (n=2) need isolated dirs. Set
     `JIUWENCLAW_DATA_DIR=/tmp/jw_rollout_<session_id>` per rollout.
  3. **Token mask construction**: WS events don't expose model tokens directly.
     Need to re-tokenize the assistant turns from history.json with the chat
     template, then mark response_mask=1 only on assistant-generated tokens
     (tool result tokens stay 0).
  4. **AgentLoop registration**: add this class to a config.yaml that veRL
     loads via actor_rollout_ref.rollout.agent.agent_loop_config_path.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Lazy/guarded imports so pure-Python helpers (_history_to_messages,
# _build_response_from_history) can be unit-tested without veRL/websockets.
try:
    import websockets  # noqa: F401
except ImportError:
    websockets = None  # type: ignore[assignment]

try:
    from verl.experimental.agent_loop.agent_loop import (
        AgentLoopBase,
        AgentLoopOutput,
        AgentLoopMetrics,
    )
except ImportError:
    class AgentLoopBase:  # type: ignore[no-redef]
        def __init__(self, *args, **kwargs):
            pass
    AgentLoopOutput = None  # type: ignore[assignment, misc]
    AgentLoopMetrics = None  # type: ignore[assignment, misc]

logger = logging.getLogger(__name__)


@dataclass
class JiuwenWSConfig:
    """Connection + workspace config for a single JiuwenClawAgentLoop instance."""
    ws_url: str = "ws://127.0.0.1:611/ws"
    data_root: Path = Path.home() / ".jiuwenclaw"
    timeout_seconds: float = 600.0
    max_session_retries: int = 2

    @classmethod
    def from_env(cls) -> "JiuwenWSConfig":
        return cls(
            ws_url=os.environ.get("JIUWENCLAW_WS_URL", "ws://127.0.0.1:611/ws"),
            data_root=Path(
                os.environ.get("JIUWENCLAW_DATA_DIR") or (Path.home() / ".jiuwenclaw")
            ).resolve(),
            timeout_seconds=float(os.environ.get("JIUWENCLAW_TIMEOUT", "600")),
            max_session_retries=int(os.environ.get("JIUWENCLAW_MAX_RETRIES", "2")),
        )


def _history_path(data_root: Path, session_id: str) -> Path:
    return data_root / "agent" / "sessions" / session_id / "history.json"


def _load_history(path: Path) -> list[dict[str, Any]]:
    """Read jiuwenclaw session history.json, retrying on partial writes."""
    for _ in range(4):
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return payload
        except Exception:
            time.sleep(0.15)
    return []


async def _run_one_ws_session(
    *,
    ws_url: str,
    session_id: str,
    prompt: str,
    timeout_seconds: float,
    request_id: str,
) -> tuple[str, bool]:
    """Send one chat.send and wait for chat.final or chat.error.

    Returns: (status, timed_out)
    status ∈ {success, error, timeout}
    """
    start = time.time()
    try:
        async with websockets.connect(ws_url, max_size=30_000_000) as ws:
            req = {
                "type": "req",
                "id": request_id,
                "method": "chat.send",
                "params": {"session_id": session_id, "content": prompt},
            }
            await ws.send(json.dumps(req, ensure_ascii=False))
            while True:
                remaining = timeout_seconds - (time.time() - start)
                if remaining <= 0:
                    return "timeout", True
                raw = await asyncio.wait_for(ws.recv(), timeout=min(20.0, max(0.1, remaining)))
                data = json.loads(raw)
                t = data.get("type")
                if t == "res" and data.get("id") == request_id:
                    if not data.get("ok", False):
                        return "error", False
                    continue
                if t != "event":
                    continue
                payload = data.get("payload") or {}
                if str(payload.get("session_id") or "") != session_id:
                    continue
                ev = data.get("event")
                if ev == "chat.error":
                    return "error", False
                if ev == "chat.final":
                    return "success", False
    except asyncio.TimeoutError:
        return "timeout", True
    except Exception:
        return "error", False


class JiuwenClawAgentLoop(AgentLoopBase):
    """veRL AgentLoop using JiuwenClaw WebSocket as the rollout runtime.

    Per `.run()` call:
      1. Build an isolated session_id + per-rollout workspace under JIUWENCLAW_DATA_DIR.
      2. Copy `extra_info.workspace_files` into `<data_root>/agent/jiuwenclaw_workspace/`.
      3. Send chat.send via WS, wait for chat.final.
      4. Read history.json, extract assistant text + tool calls.
      5. Re-tokenize with chat template, build response_mask (mask=1 only on
         assistant-generated tokens; tool result tokens get mask=0).
      6. Return AgentLoopOutput.

    Reward is NOT computed here. veRL trainer calls reward_fn separately
    on the assembled trajectory (which knows the task_id from extra_info).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = JiuwenWSConfig.from_env()
        # Tokenizer is needed for re-tokenizing assistant turns + tool results
        # so we can build response_mask. AgentLoopBase exposes it via self.tokenizer.

    def _build_workspace(self, workspace_files: list[dict], target: Path) -> None:
        """Copy task workspace files into jiuwenclaw workspace dir.

        Per-rollout isolation: target should already be unique per session.
        Cleans non-system files first to avoid stale state from prior rollouts.
        """
        target.mkdir(parents=True, exist_ok=True)
        keep = {
            "AGENT.md", "AGENTS.md", "HEARTBEAT.md", "IDENTITY.md", "SOUL.md",
            "TOOLS.md", "USER.md", "agent-data.json", "day01", "extensions",
            "memory", "scripts", "skills",
        }
        for child in target.iterdir():
            if child.name in keep:
                continue
            (shutil.rmtree if child.is_dir() else lambda p: p.unlink(missing_ok=True))(child)
        for spec in (workspace_files or []):
            if "content" in spec:
                dst = target / spec.get("path") or spec.get("dest")
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.write_text(spec["content"], encoding="utf-8")
                continue
            src = Path(spec["source"]).expanduser()
            if not src.is_absolute():
                # Resolve relative to PINCHBENCH_DIR or skill root
                pb = os.environ.get("PINCHBENCH_DIR", "")
                if pb:
                    candidate = Path(pb) / "assets" / spec["source"]
                    if not candidate.exists():
                        candidate = Path(pb) / spec["source"]
                    src = candidate
            dst = target / spec["dest"]
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    @staticmethod
    def _history_to_messages(history: list[dict]) -> list[dict[str, Any]]:
        """Flatten jiuwenclaw event-stream into OpenAI-style chat messages.

        jiuwenclaw history.json is a flat list of event records, not chat messages.
        Each entry has {role, event_type, content, tool_call, result, ...}. We
        consume the stream and emit a canonical sequence of:
          - {role: user, content: str}
          - {role: assistant, content: str, tool_calls: [...]}
          - {role: tool, content: str, tool_call_id: str}

        Turn boundary heuristic: a `chat.tool_result` flushes the pending
        assistant turn (its accumulated text + tool_calls). This handles the
        common "delta → tool_call → tool_result → delta → ..." sequence and
        also handles parallel tool_calls within one assistant turn (multiple
        chat.tool_call events before the first chat.tool_result).
        Trailing assistant text (no tool_call) flushes at end-of-stream.

        Multiple chat.delta within one turn → last one wins (jiuwenclaw emits
        cumulative snapshots, not incremental chunks). chat.final overrides.
        """
        msgs: list[dict[str, Any]] = []
        pending_text = ""
        pending_calls: list[dict[str, Any]] = []

        def _flush_assistant() -> None:
            nonlocal pending_text, pending_calls
            if not pending_text and not pending_calls:
                return
            am: dict[str, Any] = {"role": "assistant", "content": pending_text or ""}
            if pending_calls:
                am["tool_calls"] = pending_calls
            msgs.append(am)
            pending_text = ""
            pending_calls = []

        for e in history:
            role = e.get("role")
            et = e.get("event_type")
            if role == "user":
                _flush_assistant()
                msgs.append({"role": "user", "content": e.get("content", "") or ""})
                continue
            if role != "assistant":
                continue
            if et in ("chat.delta", "chat.final"):
                text = e.get("content", "") or ""
                if text:
                    pending_text = text
            elif et == "chat.tool_call":
                tc = e.get("tool_call") or {}
                args = tc.get("arguments")
                if not isinstance(args, str):
                    args = json.dumps(args or {}, ensure_ascii=False)
                pending_calls.append({
                    "id": tc.get("tool_call_id") or "",
                    "type": "function",
                    "function": {"name": tc.get("name", ""), "arguments": args},
                })
            elif et == "chat.tool_result":
                _flush_assistant()
                result = e.get("result")
                if not isinstance(result, str):
                    result = json.dumps(result or "", ensure_ascii=False)
                msgs.append({
                    "role": "tool",
                    "content": result,
                    "tool_call_id": e.get("tool_call_id") or "",
                })
            # else: chat.usage_metadata / chat.tool_update / chat.usage_summary → skip
        _flush_assistant()
        return msgs

    def _build_response_from_history(
        self, history: list[dict], prompt_token_ids: list[int],
    ) -> tuple[list[int], list[int], int]:
        """Re-tokenize jiuwenclaw history into (response_ids, response_mask, num_turns).

        Strategy: cumulative `apply_chat_template` diff.
          1. Seed prev_full = template([user_msg], add_generation_prompt=True).
             This matches what run() built for prompt_token_ids.
          2. For each subsequent message, apply template up through it WITHOUT
             add_generation_prompt; new tokens are full[len(prev_full):].
          3. Assign mask=1 to assistant turn tokens, mask=0 to tool/user.

        The tokenizer's chat template is the source of truth for serialization
        (Hermes <tool_call>...</tool_call>, <tool_response>...</tool_response>,
        <think>...</think>, etc). This decouples us from jiuwenclaw's exact
        wire format and matches veRL actor's training tokenization.

        Note: `prompt_token_ids` is taken as ground truth for the prompt length;
        the assistant/tool turn tokens we emit are appended to it by the caller.
        If history[0] (user) re-tokenizes to a length different from
        prompt_token_ids, we log a warning — caller's prompt and our seed may
        be drifting (template/tools mismatch).
        """
        msgs = self._history_to_messages(history)
        if len(msgs) < 2 or msgs[0].get("role") != "user":
            return [], [], 0

        def _ids(messages: list[dict[str, Any]], add_gen: bool) -> list[int]:
            out = self.tokenizer.apply_chat_template(
                messages, add_generation_prompt=add_gen, tokenize=True,
            )
            # transformers ≥5 returns BatchEncoding; older returns list[int].
            if isinstance(out, dict) or hasattr(out, "input_ids"):
                out = out["input_ids"]
            return list(out)

        prev_full = _ids(msgs[:1], add_gen=True)
        if len(prev_full) != len(prompt_token_ids):
            logger.warning(
                "jiuwenclaw response_mask seed length mismatch: prompt_token_ids=%d "
                "vs template_reseed=%d (template/tools drift)",
                len(prompt_token_ids), len(prev_full),
            )

        response_ids: list[int] = []
        response_mask: list[int] = []
        num_turns = 0

        for i in range(1, len(msgs)):
            full = _ids(msgs[: i + 1], add_gen=False)
            # Common-prefix length; under stable templates k == len(prev_full).
            n = min(len(prev_full), len(full))
            k = 0
            while k < n and prev_full[k] == full[k]:
                k += 1
            turn_tokens = full[k:]
            role = msgs[i].get("role")
            if role == "assistant":
                mask_val = 1
                num_turns += 1
            else:
                mask_val = 0
            response_ids.extend(turn_tokens)
            response_mask.extend([mask_val] * len(turn_tokens))
            prev_full = full

        return response_ids, response_mask, num_turns

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        # AgentLoopBase fills `kwargs` from RLHFDataset row: prompt, extra_info,
        # data_source, reward_model, etc.
        prompt = kwargs["prompt"]
        extra_info = kwargs.get("extra_info") or {}
        task_id = extra_info.get("task_id") or "unknown"

        # Per-rollout session id for isolation
        session_id = f"jw_{task_id}_{uuid.uuid4().hex[:8]}"
        request_id = f"req_{uuid.uuid4().hex[:10]}"

        # Build isolated workspace
        jc_workspace = self.cfg.data_root / "agent" / "jiuwenclaw_workspace"
        workspace_files = extra_info.get("workspace_files") or []
        self._build_workspace(workspace_files, jc_workspace)

        # Apply chat template to get baseline prompt_ids (tokenizer comes from AgentLoopBase)
        prompt_messages = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        prompt_text = self.tokenizer.apply_chat_template(
            prompt_messages, add_generation_prompt=True, tokenize=False,
        )
        prompt_token_ids = self.tokenizer.encode(prompt_text, add_special_tokens=False)

        # Extract user prompt as plain string for chat.send (jiuwenclaw expects content str)
        user_content = ""
        for msg in prompt_messages:
            if msg.get("role") == "user":
                user_content = msg.get("content", "")
                break

        # Send WS request, wait for chat.final
        status, timed_out = await _run_one_ws_session(
            ws_url=self.cfg.ws_url,
            session_id=session_id,
            prompt=user_content,
            timeout_seconds=self.cfg.timeout_seconds,
            request_id=request_id,
        )

        # Read assembled transcript from history.json
        await asyncio.sleep(0.5)   # let jiuwenclaw flush final records
        history = _load_history(_history_path(self.cfg.data_root, session_id))

        response_ids, response_mask, num_turns = self._build_response_from_history(
            history, prompt_token_ids,
        )

        # reward_score is None — veRL trainer's reward_fn will compute it from
        # the full trajectory using compute_score (e.g., meeting_reward_single_turn
        # adapted for jiuwenclaw transcripts).
        return AgentLoopOutput(
            prompt_ids=prompt_token_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=None,
            reward_score=None,
            num_turns=num_turns,
            metrics=AgentLoopMetrics(),
            extra_fields={
                "session_id": session_id,
                "task_id": task_id,
                "status": status,
                "timed_out": timed_out,
                "history_path": str(_history_path(self.cfg.data_root, session_id)),
            },
        )
