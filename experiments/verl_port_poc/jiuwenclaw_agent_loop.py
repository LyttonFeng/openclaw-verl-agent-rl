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
    # Online RL Rail: when True, build (response_ids, response_mask, logprobs)
    # from rail-v1 PerTurnSample JSONL files written by colleague's RLOnlineRail
    # via TrajectoryUploader → mock_trajectory_gateway. Bypasses history.json
    # reverse-engineering. Requires USE_RL_ONLINE_RAIL=1 in the jiuwenclaw stack.
    use_rail_v1: bool = False
    rail_v1_dir: Path = Path("/tmp/jw_rail_v1")
    rail_v1_wait_seconds: float = 15.0

    @classmethod
    def from_env(cls) -> "JiuwenWSConfig":
        use_rail_v1 = os.environ.get("USE_RL_ONLINE_RAIL", "").strip().lower() in (
            "1", "true", "yes", "on",
        )
        return cls(
            ws_url=os.environ.get("JIUWENCLAW_WS_URL", "ws://127.0.0.1:611/ws"),
            data_root=Path(
                os.environ.get("JIUWENCLAW_DATA_DIR") or (Path.home() / ".jiuwenclaw")
            ).resolve(),
            timeout_seconds=float(os.environ.get("JIUWENCLAW_TIMEOUT", "600")),
            max_session_retries=int(os.environ.get("JIUWENCLAW_MAX_RETRIES", "2")),
            use_rail_v1=use_rail_v1,
            rail_v1_dir=Path(os.environ.get("RAIL_V1_DIR", "/tmp/jw_rail_v1")),
            rail_v1_wait_seconds=float(os.environ.get("RAIL_V1_WAIT_S", "15")),
        )


def _history_path(data_root: Path, session_id: str) -> Path:
    return data_root / "agent" / "sessions" / session_id / "history.json"


def _load_task_workspace_files(task_id: str) -> list[dict[str, str]]:
    """Read task YAML frontmatter to get input files list.

    Each task .md has frontmatter like:
        workspace_files:
          - source: meetings/foo-transcript.md
            dest: transcript.md

    Returns list of {"source": abs_path, "dest": basename} dicts.
    The dataset's extra_info is missing this field, so we resolve from task_id.
    """
    pinchbench_dir = os.environ.get("PINCHBENCH_DIR", "/workspace/openclaw-verl-agent-rl")
    task_file = Path(pinchbench_dir) / "pinchbench_tasks" / "meeting_analysis" / f"{task_id}.md"
    if not task_file.exists():
        return []
    try:
        content = task_file.read_text(encoding="utf-8")
        # Extract YAML frontmatter between --- markers
        if not content.startswith("---"):
            return []
        end = content.find("\n---", 4)
        if end < 0:
            return []
        fm = content[4:end]
        # Tiny YAML parser for workspace_files list — avoid pyyaml dep
        files: list[dict[str, str]] = []
        in_block = False
        cur: dict[str, str] = {}
        for line in fm.splitlines():
            if line.startswith("workspace_files:"):
                in_block = True
                continue
            if in_block:
                stripped = line.strip()
                if not line.startswith(" ") and stripped:
                    # Top-level key — exit block
                    in_block = False
                    if cur:
                        files.append(cur); cur = {}
                    continue
                if stripped.startswith("- source:"):
                    if cur:
                        files.append(cur); cur = {}
                    cur["source"] = stripped[len("- source:"):].strip()
                elif stripped.startswith("dest:"):
                    cur["dest"] = stripped[len("dest:"):].strip()
                elif stripped.startswith("source:"):
                    cur["source"] = stripped[len("source:"):].strip()
        if cur:
            files.append(cur)
        # Resolve source paths to absolute (relative to assets/)
        assets_dir = Path(pinchbench_dir) / "assets"
        resolved: list[dict[str, str]] = []
        for f in files:
            if "source" in f and "dest" in f:
                src = Path(f["source"])
                if not src.is_absolute():
                    src = assets_dir / f["source"]
                resolved.append({"source": str(src), "dest": f["dest"]})
        return resolved
    except Exception:
        return []


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


def _load_rail_v1_samples(
    rail_dir: Path, session_id: str, wait_seconds: float = 15.0
) -> list[dict[str, Any]]:
    """Find PerTurnSamples for a given session_id from gateway JSONL output.

    The gateway writes one file per trajectory_id (jiuwenclaw UUID). trajectory_id
    is jiuwenclaw-internal — we don't know it, so we scan files for matches on
    session_id. Poll with backoff because TrajectoryUploader is async (POST
    happens AFTER trajectory.run_evolution completes, may lag a few seconds).

    Returns samples sorted by step_index. Empty list if nothing found within
    wait_seconds.
    """
    if not rail_dir.exists():
        return []
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        matches: list[dict[str, Any]] = []
        try:
            for path in sorted(rail_dir.glob("*.jsonl")):
                try:
                    with path.open("r", encoding="utf-8") as f:
                        # Cheap pre-filter: skip files that don't mention our session_id
                        head = f.read(2048)
                        if session_id not in head and not head:
                            continue
                        f.seek(0)
                        for line in f:
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                sample = json.loads(line)
                            except Exception:
                                continue
                            if str(sample.get("session_id") or "") == session_id:
                                matches.append(sample)
                except FileNotFoundError:
                    continue
                except Exception:
                    continue
        except Exception:
            pass
        if matches:
            matches.sort(key=lambda s: int(s.get("step_index") or 0))
            return matches
        time.sleep(0.5)
    return []


def _build_response_from_rail_samples(
    samples: list[dict[str, Any]],
    eos_token_id: int,
) -> tuple[list[int], list[int], list[int], list[float], int]:
    """Build (prompt_ids, response_ids, response_mask, logprobs, num_turns) from rail samples.

    Strategy: prompt_ids comes from samples[0].prompt_ids (the *actual* prompt
    vLLM saw — train-inference consistent). Between turns, the model gets
    additional tokens (tool results, system reminders) which appear at the
    head of samples[i+1].prompt_ids beyond samples[i].prompt_ids +
    samples[i].response_tokens. We mark those as mask=0 (model didn't generate
    them) and the assistant tokens as mask=1.

    Args:
        samples: PerTurnSample list sorted by step_index.
        eos_token_id: emitted if samples is empty.

    Returns:
        prompt_ids: list[int]   the initial prompt vLLM consumed (turn 0)
        response_ids: list[int] all subsequent tokens: turn 0 response +
                                (tool gap + turn 1 response) + ...
        response_mask: list[int] 1 for assistant-generated, 0 for inter-turn
                                 tool/system tokens
        logprobs: list[float]    aligned with response_ids; 0.0 for mask=0 tokens
        num_turns: int           len(samples)
    """
    if not samples:
        return [eos_token_id], [eos_token_id], [0], [0.0], 0

    s0 = samples[0]
    prompt_ids: list[int] = [int(x) for x in (s0.get("prompt_ids") or [])]
    if not prompt_ids:
        # Hard fallback — sample missing prompt_ids; we can't reconstruct
        prompt_ids = [eos_token_id]

    response_ids: list[int] = []
    response_mask: list[int] = []
    logprobs: list[float] = []

    # End-of-turn cumulative token count (prompt + response so far)
    cum_after_turn: list[int] = []

    for i, sample in enumerate(samples):
        s_prompt = [int(x) for x in (sample.get("prompt_ids") or [])]
        s_resp = [int(x) for x in (sample.get("response_tokens") or [])]
        s_lp = [float(x) for x in (sample.get("logprobs") or [])]
        if len(s_lp) != len(s_resp):
            # Pad / truncate to match response length (rail may report top_logprobs
            # only; ensure 1:1 with response tokens, missing → 0.0)
            if len(s_lp) < len(s_resp):
                s_lp = s_lp + [0.0] * (len(s_resp) - len(s_lp))
            else:
                s_lp = s_lp[: len(s_resp)]

        if i > 0:
            # New tokens at head of this turn's prompt_ids beyond what model saw
            # at end of previous turn = tool result / system reminder gap.
            prev_end = cum_after_turn[i - 1]
            gap = s_prompt[prev_end:] if len(s_prompt) > prev_end else []
            if gap:
                response_ids.extend(gap)
                response_mask.extend([0] * len(gap))
                logprobs.extend([0.0] * len(gap))

        # This turn's assistant tokens
        response_ids.extend(s_resp)
        response_mask.extend([1] * len(s_resp))
        logprobs.extend(s_lp)

        cum_after_turn.append(len(s_prompt) + len(s_resp))

    if not response_ids:
        response_ids = [eos_token_id]
        response_mask = [0]
        logprobs = [0.0]

    return prompt_ids, response_ids, response_mask, logprobs, len(samples)


async def _run_one_ws_session(
    *,
    ws_url: str,
    session_id: str,
    prompt: str,
    timeout_seconds: float,
    request_id: str,
    connect_retries: int = 60,
    connect_retry_delay: float = 4.0,
) -> tuple[str, bool]:
    """Send one chat.send and wait for chat.final or chat.error.

    Retries the initial WS connect — veRL launches agent_loop workers
    BEFORE vLLM HTTP fully readies (Training Progress prints at 0/24 while
    CUDA graphs are still being captured), and the Path A launcher's
    headless jiuwenclaw only comes up after the `LLMServerManager:` line
    appears (~1–2 min in). So agent_loop's first run() can fire when
    nothing is listening on the WS port yet. Default: 60 retries × 4s =
    240s grace window covers the entire bootstrap.

    Returns: (status, timed_out)
    status ∈ {success, error, timeout}
    """
    start = time.time()
    # Retry WS connect (jiuwenclaw may still be coming up)
    ws = None
    last_connect_err: Optional[Exception] = None
    for attempt in range(connect_retries):
        try:
            ws = await websockets.connect(ws_url, max_size=30_000_000)
            break
        except Exception as e:  # noqa: BLE001
            last_connect_err = e
            if attempt < connect_retries - 1:
                await asyncio.sleep(connect_retry_delay)
    if ws is None:
        logger.warning("jiuwenclaw WS connect failed after %d retries to %s: %r",
                       connect_retries, ws_url, last_connect_err)
        return "error", False
    try:
        req = {
            "type": "req",
            "id": request_id,
            "method": "chat.send",
            # enable_memory=False alone is silently ignored (jiuwenclaw
            # gateway_normalize.py:128 requires all three flags). Setting the
            # avatar/group flags is what actually disables memory writes.
            "params": {
                "session_id": session_id,
                "content": prompt,
                "enable_memory": False,
                "group_digital_avatar": True,
                "is_group_chat": True,
            },
        }
        await ws.send(json.dumps(req, ensure_ascii=False))
        got_any_event = False
        while True:
            remaining = timeout_seconds - (time.time() - start)
            if remaining <= 0:
                return "timeout", True
            # Cold-start: jiuwenclaw needs 25-40s to load identity / memory
            # before first event. Old 20s cap caused fake-dead 假死 timeouts:
            # trajectory abort → status=timeout, history_len=0, 50% empty batch.
            # Use 120s before first event, 45s for inter-event gaps.
            inner_to = min(120.0 if not got_any_event else 45.0, max(0.1, remaining))
            raw = await asyncio.wait_for(ws.recv(), timeout=inner_to)
            got_any_event = True
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
    except Exception as e:  # noqa: BLE001
        logger.warning("jiuwenclaw WS session error: %r", e)
        return "error", False
    finally:
        try:
            await ws.close()
        except Exception:  # noqa: BLE001
            pass


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

    # Multi-stack parallelism: per-stack lock + round-robin selection. Each
    # stack has independent WS / workspace dir, so different stacks can run
    # concurrently. Within one stack, lock serializes rollouts that share
    # workspace. Stacks defined via env JIUWENCLAW_WS_URLS (comma list) and
    # JIUWENCLAW_DATA_DIRS (comma list, matched by index). Falls back to
    # single-stack mode if URLs list missing.
    _stack_locks: dict[int, asyncio.Lock] | None = None
    _stack_rr: int = 0
    _stacks: list[tuple[str, Path]] | None = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cfg = JiuwenWSConfig.from_env()
        # Tokenizer is needed for re-tokenizing assistant turns + tool results
        # so we can build response_mask. AgentLoopBase exposes it via self.tokenizer.
        if type(self)._stacks is None:
            urls = os.environ.get("JIUWENCLAW_WS_URLS", "").strip()
            dirs = os.environ.get("JIUWENCLAW_DATA_DIRS", "").strip()
            if urls and dirs:
                url_list = [u.strip() for u in urls.split(",") if u.strip()]
                dir_list = [Path(d.strip()).resolve() for d in dirs.split(",") if d.strip()]
                if len(url_list) == len(dir_list):
                    type(self)._stacks = list(zip(url_list, dir_list))
            # Fallback: single stack from JiuwenWSConfig.from_env (legacy mode)
            if type(self)._stacks is None:
                type(self)._stacks = [(self.cfg.ws_url, self.cfg.data_root)]
            type(self)._stack_locks = {i: asyncio.Lock() for i in range(len(type(self)._stacks))}
            logger.info("JiuwenClawAgentLoop multi-stack: %d stacks: %s",
                        len(type(self)._stacks), type(self)._stacks)

    def _pick_stack(self) -> tuple[int, str, Path, asyncio.Lock]:
        """Round-robin a stack for this rollout."""
        cls = type(self)
        idx = cls._stack_rr % len(cls._stacks)
        cls._stack_rr += 1
        ws_url, data_root = cls._stacks[idx]
        return idx, ws_url, data_root, cls._stack_locks[idx]

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

        # Serialize workspace prep + WS roundtrip on the shared jiuwenclaw
        # workspace dir. The stack server reads files from
        # ~/.jiuwenclaw/agent/jiuwenclaw_workspace/ (hardcoded) so concurrent
        # rollouts within one worker process would race. The lock makes this
        # safe at the cost of in-worker serial execution; parallelism still
        # comes from multiple AgentLoopWorker processes (each gets its own lock).
        # Pick a stack (round-robin). Each stack has independent WS + data_root.
        stack_idx, stack_ws_url, stack_data_root, stack_lock = self._pick_stack()
        jc_workspace = stack_data_root / "agent" / "jiuwenclaw_workspace"
        workspace_files = extra_info.get("workspace_files") or []
        # 训推一致 fix: dataset's extra_info doesn't carry workspace_files,
        # so fallback to loading from task YAML frontmatter (same source bench
        # harness uses). Otherwise model finds empty workspace, can't read
        # transcript.md, just does <think> reasoning forever → reward=0.
        if not workspace_files and task_id and task_id != "unknown":
            workspace_files = _load_task_workspace_files(task_id)
            if workspace_files:
                logger.info("loaded %d workspace_files from task %s YAML",
                            len(workspace_files), task_id)
        async with stack_lock:
            # 训推一致 fix: clean workspace BEFORE rollout to avoid leakage from
            # previous rollout's files (each judge call should see only this
            # rollout's writes).
            if jc_workspace.exists():
                for _it in jc_workspace.iterdir():
                    try:
                        if _it.is_file() or _it.is_symlink():
                            _it.unlink()
                        elif _it.is_dir():
                            shutil.rmtree(_it)
                    except Exception:
                        pass
            self._build_workspace(workspace_files, jc_workspace)
            status, timed_out = await _run_one_ws_session(
                ws_url=stack_ws_url,
                session_id=session_id,
                prompt=user_content,
                timeout_seconds=self.cfg.timeout_seconds,
                request_id=request_id,
            )
            await asyncio.sleep(0.5)   # let jiuwenclaw flush final records
            history = _load_history(_history_path(stack_data_root, session_id))
            # 训推一致 fix: snapshot workspace AFTER rollout, BEFORE releasing
            # lock (otherwise next rollout clears jc_workspace). Pass snapshot
            # path via extra_fields → reward.compute_score reads real files.
            ws_snapshot = stack_data_root / "agent" / "sessions" / session_id / "ws_snapshot"
            try:
                if ws_snapshot.exists():
                    shutil.rmtree(ws_snapshot)
                ws_snapshot.parent.mkdir(parents=True, exist_ok=True)
                if jc_workspace.exists():
                    shutil.copytree(jc_workspace, ws_snapshot)
                else:
                    ws_snapshot.mkdir(parents=True, exist_ok=True)
            except Exception as _e:
                logger.warning("workspace snapshot failed: %s", _e)
                ws_snapshot.mkdir(parents=True, exist_ok=True)

        # Path A: rail-v1 (preferred) — read PerTurnSamples from the gateway
        # output dir. Each sample is what vLLM actually saw + generated, so
        # token_ids / logprobs are train-inference consistent. No
        # reverse-engineering history.json's chunked event stream.
        # Path B: legacy history.json reverse-engineering (fallback).
        rail_samples: list[dict[str, Any]] = []
        rail_response_logprobs: list[float] = []
        if self.cfg.use_rail_v1:
            rail_samples = _load_rail_v1_samples(
                self.cfg.rail_v1_dir,
                session_id,
                wait_seconds=self.cfg.rail_v1_wait_seconds,
            )
            if rail_samples:
                eos_id = self.tokenizer.eos_token_id or 0
                (
                    prompt_token_ids,
                    response_ids,
                    response_mask,
                    rail_response_logprobs,
                    num_turns,
                ) = _build_response_from_rail_samples(rail_samples, eos_id)
                logger.info(
                    "rail-v1: built response from %d samples session=%s "
                    "prompt_len=%d response_len=%d num_turns=%d",
                    len(rail_samples), session_id,
                    len(prompt_token_ids), len(response_ids), num_turns,
                )
            else:
                logger.warning(
                    "rail-v1: no samples found for session=%s within %.1fs "
                    "(rail_dir=%s) — falling back to history.json",
                    session_id, self.cfg.rail_v1_wait_seconds, self.cfg.rail_v1_dir,
                )

        if not rail_samples:
            response_ids, response_mask, num_turns = self._build_response_from_history(
                history, prompt_token_ids,
            )

        # Truncate to veRL's max_response_length — otherwise long multi-turn
        # jiuwenclaw sessions can accumulate >12k tokens, breaking veRL's
        # _postprocess `torch.cat` which expects all rollouts padded to a
        # uniform shape. Mirrors OpenClawAgentLoop's `[:response_length]`.
        max_resp_len = int(self.rollout_config.response_length)
        if len(response_ids) > max_resp_len:
            logger.warning(
                "jiuwenclaw response_ids %d > max_response_length %d, truncating",
                len(response_ids), max_resp_len,
            )
            response_ids = response_ids[:max_resp_len]
            response_mask = response_mask[:max_resp_len]
            if rail_response_logprobs:
                rail_response_logprobs = rail_response_logprobs[:max_resp_len]

        # veRL's _agent_loop_postprocess (line 575) does tokenizer.pad on
        # response_ids and expects a tensor result. Empty list[int] makes pad
        # return {"input_ids": []} (still a list), which then fails .dim().
        # Emit a single EOS-as-placeholder so postprocess produces a valid
        # (all-padding) tensor. mask=0 so it contributes nothing to training.
        # Mirrors OpenClawAgentLoop's empty-response guard.
        if not response_ids:
            eos_id = self.tokenizer.eos_token_id or 0
            logger.warning(
                "jiuwenclaw run produced empty response_ids "
                "(status=%s, timed_out=%s, history_len=%d) — emitting EOS placeholder",
                status, timed_out, len(history),
            )
            response_ids = [eos_id]
            response_mask = [0]
        if not prompt_token_ids:
            eos_id = self.tokenizer.eos_token_id or 0
            prompt_token_ids = [eos_id]

        # reward_score is None — veRL trainer's reward_fn will compute it from
        # the full trajectory using compute_score (e.g., meeting_reward_single_turn
        # adapted for jiuwenclaw transcripts).
        # Pad logprobs to match response_ids length if rail-v1 was used.
        # rail_response_logprobs may be shorter than response_ids if truncation
        # / empty-rollout placeholders adjusted the latter — keep them aligned.
        response_logprobs = None
        if rail_response_logprobs:
            if len(rail_response_logprobs) < len(response_ids):
                rail_response_logprobs = rail_response_logprobs + [0.0] * (
                    len(response_ids) - len(rail_response_logprobs)
                )
            elif len(rail_response_logprobs) > len(response_ids):
                rail_response_logprobs = rail_response_logprobs[: len(response_ids)]
            response_logprobs = rail_response_logprobs

        return AgentLoopOutput(
            prompt_ids=prompt_token_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            response_logprobs=response_logprobs,
            reward_score=None,
            num_turns=num_turns,
            metrics=AgentLoopMetrics(),
            extra_fields={
                "session_id": session_id,
                "task_id": task_id,
                "status": status,
                "timed_out": timed_out,
                "history_path": str(_history_path(stack_data_root, session_id)),
                "stack_idx": stack_idx,
                # rail-v1 metadata (None when use_rail_v1=False or no samples found)
                "rail_v1_samples": len(rail_samples) if rail_samples else 0,
                "rail_v1_used": bool(rail_samples),
                # 训推一致 fix: pass real workspace path + transcript so reward
                # function (meeting_reward.py) reads actual files instead of
                # judging raw response_str dummy summary. veRL's reward_manager
                # merges extra_fields → extra_info before compute_score().
                "workspace_path": str(ws_snapshot),
                "transcript": history,
            },
        )
