"""
OpenClawAgentLoop: veRL agent loop that drives OpenClaw as the RL environment.

Architecture:
  1. Inherit from veRL's AgentLoopBase
  2. Start ModelProxy on ephemeral port
  3. SSH to ECS, run `openclaw agent --local` pointing LLM at ModelProxy
  4. Intercept each LLM request, use self.server_manager.generate() for token generation
  5. Convert tokens back to text, return via ModelProxy to OpenClaw
  6. After episode, run grading and compute process rewards
  7. Return AgentLoopOutput for veRL training

Environment variables:
  OPENCLAW_HOST, OPENCLAW_USER, OPENCLAW_SSH_KEY, OPENCLAW_PORT,
  OPENCLAW_WORKSPACE, MAX_TURNS (default 10; proxy-side model rounds per episode),
  PINCHBENCH_DIR, REWARD_MODE,
  PRM_VLLM_BASE_URL, PRM_MODEL, PRM_API_KEY,
  OPENCLAW_WEB_SEARCH_SKILLS, OPENCLAW_WEB_FETCH_SKILLS

  PINCHBENCH_RL_INJECT_TOOL_FORMAT_SUFFIX — Set to 1/true to append the legacy
  ``<tool_call>`` XML instructions to the first system message (training-only shim).
  **Default: unset/false** so rollout matches PinchBench/OpenClaw → vLLM (no extra
  system suffix), consistent with benchmark inference.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import sys
import shlex
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopMetrics,
    AgentLoopOutput,
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
if not logger.handlers:
    _h = logging.StreamHandler()
    _h.setLevel(logging.DEBUG)
    logger.addHandler(_h)

# ── Per-task EMA baseline for turn rewards ────────────────────────────────────
# Mirrors the same logic in reward_manager.py but lives here because
# turn-reward normalization must happen before token assignment, in this process.
# The two EMA dicts are independent (different Ray workers) but converge similarly.
_turn_task_ema: dict[str, float] = {}
_turn_task_ema_var: dict[str, float] = {}
_TURN_EMA_ALPHA = float(os.environ.get("PINCHBENCH_TASK_EMA_ALPHA", "0.1"))
_TURN_EMA_INIT = float(os.environ.get("PINCHBENCH_TASK_EMA_INIT", "0.5"))
_TURN_EMA_VAR_INIT = float(os.environ.get("PINCHBENCH_TASK_EMA_VAR_INIT", "0.0"))


def _normalize_turn_rewards(task_id: str, per_turn_rewards: list[float]) -> list[float]:
    """Normalize per-turn rewards with task-specific EMA mean and variance.

    Tracks mean per-turn reward (not sum) so the baseline is independent of
    episode length. Uses sqrt(EMA variance + 1.0), not sqrt(var + eps), to avoid
    sparse-reward scale explosions.

    - Always-failing task (raw_mean → 0, baseline → 0): advantage → 0, no gradient.
    - Improving task (raw_mean > baseline): positive advantages, learning signal.
    - Consistently high task (raw_mean ≈ baseline): advantages → 0, converged.
    """
    if not per_turn_rewards:
        return per_turn_rewards
    raw_mean = sum(per_turn_rewards) / len(per_turn_rewards)
    if task_id not in _turn_task_ema:
        _turn_task_ema[task_id] = _TURN_EMA_INIT
        _turn_task_ema_var[task_id] = _TURN_EMA_VAR_INIT
    baseline = _turn_task_ema[task_id]
    variance = _turn_task_ema_var.get(task_id, _TURN_EMA_VAR_INIT)
    scale = (variance + 1.0) ** 0.5
    centered = raw_mean - baseline
    _turn_task_ema[task_id] = (1.0 - _TURN_EMA_ALPHA) * baseline + _TURN_EMA_ALPHA * raw_mean
    _turn_task_ema_var[task_id] = (1.0 - _TURN_EMA_ALPHA) * variance + _TURN_EMA_ALPHA * (centered ** 2)

    normalized = [(r - baseline) / scale for r in per_turn_rewards]
    logger.debug(
        "[EMA] task=%s raw_mean=%.3f baseline=%.3f var=%.3f scale=%.3f normalized_sum=%.3f",
        task_id, raw_mean, baseline, variance, scale, sum(normalized),
    )
    return normalized

DEFAULT_WEB_SEARCH_SKILLS = (
    "web_search",
    "ddg-search",
    "search-web",
    "web-search-free",
    "ddg-web-search",
    "dashscope-web-search",
    "local-web-search-skill",
    "websearch",
)
DEFAULT_WEB_FETCH_SKILLS = (
    "web_fetch",
    "web-fetch",
    "web-fetch-markdown",
    "clean-web-fetch",
    "jina-web-fetcher",
    "safe-smart-web-fetch",
    "webfetch",
)


def _task_markdown_dir(repo_root: str | Path) -> Path:
    root = Path(repo_root)
    for name in ("pinchbench_tasks", "tasks"):
        candidate = root / name
        if candidate.is_dir():
            return candidate
    return root / "tasks"

TASK18_TEACHER_HINT = (
    "When the task includes an .xlsx workbook, do not use the read tool on the .xlsx file. "
    "Use exec with Python plus pandas/openpyxl first to inspect workbook sheets, columns, and rows. "
    "Use read only for plain-text files such as .csv, .md, or .txt."
)


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}


def _task18_teacher_hint_enabled() -> bool:
    return _env_flag("PINCHBENCH_TASK18_TEACHER_HINT")


def _task18_teacher_hint_prob() -> float:
    raw = os.environ.get("PINCHBENCH_TASK18_TEACHER_HINT_PROB", "0.35").strip()
    try:
        return max(0.0, min(1.0, float(raw)))
    except ValueError:
        logger.warning("Invalid PINCHBENCH_TASK18_TEACHER_HINT_PROB=%r, fallback to 0.35", raw)
        return 0.35


def _maybe_inject_task18_teacher_hint(
    task_id: str, raw_prompt: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if task_id != "task_18_spreadsheet_summary" or not _task18_teacher_hint_enabled():
        return raw_prompt
    if random.random() > _task18_teacher_hint_prob():
        return raw_prompt
    injected = [dict(m) for m in raw_prompt]
    for idx in range(len(injected) - 1, -1, -1):
        if injected[idx].get("role") != "user":
            continue
        content = injected[idx].get("content", "")
        if TASK18_TEACHER_HINT in content:
            return injected
        injected[idx]["content"] = f"{TASK18_TEACHER_HINT}\n\n{content}"
        logger.warning(
            "[task18-teacher-hint] injected rollout hint into task_18 user prompt (prob=%.2f)",
            _task18_teacher_hint_prob(),
        )
        return injected
    return injected


def _rsync_bin() -> str:
    """Resolve rsync in Ray workers whose PATH may not include /usr/bin."""
    found = shutil.which("rsync")
    if found:
        return found
    fallback = Path("/usr/bin/rsync")
    if fallback.exists():
        return str(fallback)
    return "rsync"


def _ecs_retry_attempts() -> int:
    return max(1, int(os.environ.get("PINCHBENCH_ECS_RETRY_ATTEMPTS", "3")))


def _run_subprocess_with_retry(
    cmd: list[str],
    *,
    label: str,
    timeout: float,
    attempts: Optional[int] = None,
) -> subprocess.CompletedProcess[str]:
    total = attempts or _ecs_retry_attempts()
    last: Optional[subprocess.CompletedProcess[str]] = None
    for attempt in range(1, total + 1):
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            result = subprocess.CompletedProcess(cmd, 124, "", str(exc))
        last = result
        if result.returncode == 0:
            return result
        logger.warning(
            "%s failed (attempt %d/%d, rc=%s): %s",
            label,
            attempt,
            total,
            result.returncode,
            (result.stderr or result.stdout or "").strip()[:300],
        )
        if attempt < total:
            time.sleep(min(2 * attempt, 8))
    assert last is not None
    return last


def _resolve_pinchbench_dir() -> str:
    """Repo root with ``tasks/`` and ``assets/`` (for ECS workspace seeding)."""
    pb = os.environ.get("PINCHBENCH_DIR", "").strip()
    if pb:
        return pb
    here = Path(__file__).resolve()
    inferred = here.parents[2]  # rl/agent_loop/this.py -> repo root
    if (inferred / "tasks").is_dir() and (inferred / "assets").is_dir():
        return str(inferred)
    return ""


TASK16_NOISE_PATH_SUFFIXES = (
    "agents.md",
    "bootstrap.md",
    "soul.md",
    "user.md",
    "identity.md",
    "heartbeat.md",
    "tools.md",
    "memory.md",
    "report.md",
    "inbox.md",
    "emails.md",
    "emails.txt",
    "inbox_emails.txt",
    "event_map.md",
    "report_schema.md",
    "triax_report.md",
    "tripe_report.md",
)


def _task16_is_inbox_path(path: str) -> bool:
    p = path.strip().lower()
    return p.startswith("inbox/email_")


def _task16_is_noise_path(path: str) -> bool:
    p = path.strip().lower()
    if not p or _task16_is_inbox_path(p):
        return False
    if p == "memory" or p.startswith("memory/"):
        return True
    return any(p == suffix or p.endswith(f"/{suffix}") for suffix in TASK16_NOISE_PATH_SUFFIXES)


def _task16_episode_tags(trajectory: list[dict[str, Any]], workspace_path: str) -> dict[str, Any]:
    inbox_reads = 0
    non_task_reads = 0
    first_read_paths: list[str] = []
    report_written = False
    assistant_read_only_turns = 0

    for turn in trajectory:
        if turn.get("role") != "assistant":
            continue
        reads = []
        for tc in turn.get("tool_calls") or []:
            fn = tc.get("function") if isinstance(tc, dict) else None
            name = ""
            args = ""
            if isinstance(fn, dict):
                name = str(fn.get("name", ""))
                args = fn.get("arguments", "")
            else:
                name = str(tc.get("name", ""))
                args = tc.get("arguments", "")
            if name == "read":
                try:
                    parsed = json.loads(args) if isinstance(args, str) else (args or {})
                except Exception:
                    parsed = {}
                path = str(parsed.get("path", ""))
                if path:
                    reads.append(path)
            if name == "write":
                try:
                    parsed = json.loads(args) if isinstance(args, str) else (args or {})
                except Exception:
                    parsed = {}
                path = str(parsed.get("path", "")).lower()
                if path.endswith("triage_report.md"):
                    report_written = True
        if reads and not report_written:
            assistant_read_only_turns += 1
        if reads and len(first_read_paths) < 5:
            first_read_paths.extend(reads[: max(0, 5 - len(first_read_paths))])
        inbox_reads += sum(1 for path in reads if _task16_is_inbox_path(path))
        non_task_reads += sum(1 for path in reads if _task16_is_noise_path(path))

    tags = {
        "non_task_read_count": non_task_reads,
        "inbox_read_count": inbox_reads,
        "first_read_paths": first_read_paths,
        "first_reads_are_task_files": bool(first_read_paths) and all(
            _task16_is_inbox_path(path) for path in first_read_paths
        ),
        "report_written": report_written or (Path(workspace_path) / "triage_report.md").exists(),
        "read_only_turns": assistant_read_only_turns,
    }
    tags["failure_tags"] = [
        tag
        for tag, ok in (
            ("task_grounding_under_workspace_noise", tags["non_task_read_count"] > 0),
            ("no_report", not tags["report_written"]),
            ("read_loop", tags["read_only_turns"] >= 3),
        )
        if ok
    ]
    return tags


@dataclass
class OpenClawConfig:
    host: str = "localhost"
    user: str = "root"
    ssh_key: str = str(Path.home() / ".ssh" / "id_ed25519")
    ssh_port: int = 22
    workspace_base: str = "/tmp/pinchbench"
    pinchbench_dir: str = ""
    reward_mode: str = "self-judge"
    proxy_bind_host: str = "0.0.0.0"
    agent_timeout: float = 240.0
    max_turns: int = 10
    # turn>0 且 OpenClaw 请求 messages<=2 连续达到此次数则判为卡死；0=关闭该检测
    stuck_retry_threshold: int = 2
    prm_vllm_base_url: str = "http://localhost:9090/v1"
    prm_model: str = "Qwen/Qwen3-4B"
    prm_api_key: str = "dummy"
    remote_activate_cmd: str = ""

    @classmethod
    def from_env(cls) -> "OpenClawConfig":
        pb = _resolve_pinchbench_dir()
        if pb and not os.environ.get("PINCHBENCH_DIR", "").strip():
            logger.warning("PINCHBENCH_DIR unset; using inferred repo root %s", pb)
        return cls(
            host=os.environ.get("OPENCLAW_HOST", "localhost"),
            user=os.environ.get("OPENCLAW_USER", "root"),
            ssh_key=os.environ.get("OPENCLAW_SSH_KEY", str(Path.home() / ".ssh" / "id_ed25519")),
            ssh_port=int(os.environ.get("OPENCLAW_PORT", "22")),
            workspace_base=os.environ.get("OPENCLAW_WORKSPACE", "/tmp/pinchbench"),
            pinchbench_dir=pb,
            reward_mode=os.environ.get("REWARD_MODE", "self-judge"),
            proxy_bind_host=os.environ.get("PROXY_BIND_HOST", "0.0.0.0"),
            agent_timeout=float(os.environ.get("AGENT_TIMEOUT", "240")),
            max_turns=int(os.environ.get("MAX_TURNS", "10")),
            stuck_retry_threshold=int(os.environ.get("OPENCLAW_STUCK_RETRY_THRESHOLD", "2")),
            prm_vllm_base_url=os.environ.get("PRM_VLLM_BASE_URL", "http://localhost:9090/v1"),
            prm_model=os.environ.get("PRM_MODEL", "Qwen/Qwen3-4B"),
            prm_api_key=os.environ.get("PRM_API_KEY", "dummy"),
            remote_activate_cmd=os.environ.get("OPENCLAW_REMOTE_ACTIVATE_CMD", ""),
        )


class OpenClawAgentLoop(AgentLoopBase):
    """veRL agent loop for PinchBench via OpenClaw."""

    def __init__(self, config: Optional[dict] = None, **kwargs):
        super().__init__(**kwargs)
        if config and isinstance(config, dict):
            self.oc_config = OpenClawConfig(**config)
        else:
            self.oc_config = OpenClawConfig.from_env()
        reasoning_enabled = os.environ.get("OPENCLAW_MODEL_REASONING", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not reasoning_enabled:
            self.apply_chat_template_kwargs = dict(getattr(self, "apply_chat_template_kwargs", {}) or {})
            self.apply_chat_template_kwargs.setdefault("enable_thinking", False)

    def _ensure_pinchbench_dir(self) -> None:
        """Ray / Hydra may leave pinchbench_dir empty; seeding needs tasks + assets paths."""
        if self.oc_config.pinchbench_dir.strip():
            return
        pb = _resolve_pinchbench_dir()
        if pb:
            self.oc_config.pinchbench_dir = pb
            os.environ.setdefault("PINCHBENCH_DIR", pb)
            logger.warning("pinchbench_dir was empty; set to %s for workspace seeding", pb)

    def _use_benchmark_runtime(self, task_id: str) -> bool:
        raw = os.environ.get("PINCHBENCH_AGENT_LOOP_RUNTIME", "auto").strip().lower()
        if raw in {"benchmark", "bench", "lib_agent"}:
            return True
        if raw in {"legacy", "agent_loop", "custom"}:
            return False
        return task_id == "task_16_email_triage"

    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        """Run one PinchBench episode.

        kwargs contains dataset fields: raw_prompt, extra_info, etc.
        """
        self._ensure_pinchbench_dir()
        from .model_proxy import ModelProxy, ModelRequest
        from .trajectory import TrajectoryReconstructor, TurnRecord, compact_turn_history

        print(f"[OpenClawAgentLoop.run] START kwargs keys={list(kwargs.keys())}", flush=True)
        logger.info("[run] START kwargs keys=%s, sampling_params=%s", list(kwargs.keys()), sampling_params)

        raw_prompt = kwargs.get("raw_prompt", [])
        extra_info = kwargs.get("extra_info", {})
        task_id = extra_info.get("task_id", "unknown")
        raw_prompt = _maybe_inject_task18_teacher_hint(task_id, raw_prompt)
        task_prompt = ""
        if raw_prompt:
            last_user = [m for m in raw_prompt if m.get("role") == "user"]
            if last_user:
                task_prompt = last_user[-1].get("content", "")

        print(f"[OpenClawAgentLoop.run] task_id={task_id}, host={self.oc_config.host}, prompt_len={len(task_prompt)}", flush=True)
        logger.info("[run] task_id=%s, host=%s, prompt_len=%d", task_id, self.oc_config.host, len(task_prompt))

        session_id = f"rl-{uuid.uuid4().hex[:8]}"
        workspace_rel = self._episode_workspace_rel(task_id, session_id)

        proxy = ModelProxy(host=self.oc_config.proxy_bind_host, port=0)
        proxy_port = await proxy.start()
        print(f"[OpenClawAgentLoop.run] ModelProxy started on port {proxy_port}", flush=True)
        logger.info("[run] ModelProxy started on port %d", proxy_port)

        all_prompt_ids: list[int] = []
        all_response_ids: list[int] = []
        all_response_mask: list[int] = []
        all_response_logprobs: list[float] = []
        turns: list[TurnRecord] = []
        messages: list[dict] = []
        turn_count = 0
        retry_count = 0

        async def _compact_messages_by_turn(
            messages_in: list[dict[str, Any]],
            max_prompt_tokens: int,
            tools: Optional[list[dict]] = None,
        ) -> list[dict[str, Any]]:
            """Trim oldest complete turns until the prompt fits the context budget.

            IMPORTANT: Uses LIFO (drop newest turns) instead of FIFO to preserve
            early turns where critical tool calls (like exec) often appear.
            """
            if max_prompt_tokens <= 0 or not messages_in:
                return messages_in

            assistant_indices = [
                idx for idx, msg in enumerate(messages_in) if msg.get("role") == "assistant"
            ]
            if not assistant_indices:
                return messages_in

            prefix = messages_in[:assistant_indices[0]]
            turn_starts = assistant_indices[:]
            candidate = list(messages_in)

            # LIFO: drop from the END instead of the beginning
            while turn_starts:
                prompt_ids = await self.apply_chat_template(candidate, tools=tools)
                if len(prompt_ids) <= max_prompt_tokens:
                    return candidate

                # Drop the LAST turn instead of the first
                if len(turn_starts) == 1:
                    # Only one turn left, can't drop more
                    logger.warning(
                        "[run] Cannot compact further: only 1 turn remains, %d tokens > %d limit",
                        len(prompt_ids),
                        max_prompt_tokens,
                    )
                    return candidate

                drop_start = turn_starts[-1]
                logger.warning(
                    "[run] Compacting NEWEST turn (LIFO): drop messages[%d:] to fit %d tokens (current=%d)",
                    drop_start,
                    max_prompt_tokens,
                    len(prompt_ids),
                )
                candidate = candidate[:drop_start]
                turn_starts = turn_starts[:-1]

            return candidate

        def _trim_trajectory_to_last_turns(
            trajectory: list[dict[str, Any]],
            keep_assistant_turns: int,
        ) -> list[dict[str, Any]]:
            if keep_assistant_turns <= 0 or not trajectory:
                return trajectory[-1:] if trajectory else []
            assistant_positions = [
                idx for idx, msg in enumerate(trajectory) if msg.get("role") == "assistant"
            ]
            if len(assistant_positions) <= keep_assistant_turns:
                return trajectory
            cut_index = assistant_positions[-keep_assistant_turns]
            return trajectory[cut_index:]

        def _max_prompt_tokens() -> int:
            explicit = os.environ.get("PINCHBENCH_AGENT_MAX_PROMPT_TOKENS", "").strip()
            if explicit:
                try:
                    return max(0, int(explicit))
                except ValueError:
                    logger.warning("Invalid PINCHBENCH_AGENT_MAX_PROMPT_TOKENS=%r", explicit)
            for attr in ("max_prompt_length", "prompt_length"):
                value = getattr(self.rollout_config, attr, None)
                if value:
                    try:
                        return max(0, int(value))
                    except (TypeError, ValueError):
                        pass
            try:
                max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "0") or 0)
                response_len = int(getattr(self.rollout_config, "response_length", 0) or 0)
                if max_model_len > response_len:
                    return max(0, max_model_len - response_len - 512)
            except (TypeError, ValueError):
                pass
            return 0

        def _effective_generate_budget(
            prompt_len: int,
            used_response_len: int,
            requested_max_tokens: int,
        ) -> int:
            """Clamp generation to the remaining response/model-context budget.

            Validation can compact history down to a single turn and still end up
            with a prompt that nearly fills the model context window. In that case
            vLLM rejects max_tokens=0. Guard here instead of letting the request
            reach vLLM with an invalid sampling config.
            """
            response_cap = int(getattr(self.rollout_config, "response_length", 0) or 0)
            response_remaining = max(0, response_cap - used_response_len) if response_cap > 0 else 0

            try:
                max_model_len = int(os.environ.get("VLLM_MAX_MODEL_LEN", "0") or 0)
            except (TypeError, ValueError):
                max_model_len = 0
            model_remaining = max(0, max_model_len - prompt_len) if max_model_len > 0 else 0

            caps = [cap for cap in (response_remaining, model_remaining, requested_max_tokens) if cap > 0]
            if not caps:
                return 0
            return min(caps)

        t_start = time.time()
        openclaw_proc: Optional[asyncio.subprocess.Process] = None
        benchmark_runtime_task: Optional[asyncio.Task] = None
        benchmark_execution_result: Optional[dict[str, Any]] = None
        reverse_tunnel_proc: Optional[asyncio.subprocess.Process] = None
        metrics: dict[str, Any] = {}
        use_benchmark_runtime = self._use_benchmark_runtime(task_id)

        try:
            if use_benchmark_runtime:
                proxy_url = f"http://127.0.0.1:{proxy_port}/v1"
                if self.oc_config.host not in ("localhost", "127.0.0.1"):
                    reverse_tunnel_proc = await self._start_reverse_tunnel(proxy_port)
                logger.info("[run] Starting benchmark runtime for task=%s", task_id)
                benchmark_runtime_task = asyncio.create_task(asyncio.to_thread(
                    self._execute_with_benchmark_runtime,
                    task_id,
                    task_prompt,
                    session_id,
                    proxy_url,
                    extra_info,
                    workspace_rel,
                ))
            elif self.oc_config.host in ("localhost", "127.0.0.1"):
                logger.info("[run] Starting local OpenClaw for task=%s", task_id)
                openclaw_proc = await self._start_local_openclaw(
                    task_prompt, session_id, f"http://127.0.0.1:{proxy_port}/v1", task_id, extra_info, workspace_rel,
                )
            else:
                logger.info("[run] Starting remote OpenClaw on %s for task=%s", self.oc_config.host, task_id)
                openclaw_proc = await self._start_remote_openclaw(
                    task_prompt, session_id, proxy_port, task_id, extra_info, workspace_rel,
                )
            if openclaw_proc is not None:
                logger.info("[run] OpenClaw started pid=%s, waiting for first request...", openclaw_proc.pid)
            else:
                logger.info("[run] Benchmark runtime started, waiting for first proxy request...")

            async def _drain_oc_stream(stream, label, tid):
                while True:
                    line = await stream.readline()
                    if not line:
                        break
                    logger.debug("[OC %s/%s] %s", tid, label, line.decode(errors="replace").rstrip())

            if openclaw_proc is not None:
                asyncio.get_event_loop().create_task(_drain_oc_stream(openclaw_proc.stdout, "out", task_id))
                asyncio.get_event_loop().create_task(_drain_oc_stream(openclaw_proc.stderr, "err", task_id))

            while turn_count < self.oc_config.max_turns:
                try:
                    logger.info("[run] Waiting for proxy request (turn=%d, timeout=%.0fs)...", turn_count, self.oc_config.agent_timeout)
                    proxy_task = asyncio.ensure_future(proxy.get_request())
                    runtime_wait_task = (
                        asyncio.ensure_future(openclaw_proc.wait())
                        if openclaw_proc is not None
                        else asyncio.ensure_future(asyncio.shield(benchmark_runtime_task))
                    )
                    done, pending = await asyncio.wait(
                        [proxy_task, runtime_wait_task],
                        timeout=self.oc_config.agent_timeout,
                        return_when=asyncio.FIRST_COMPLETED,
                    )
                    for p in pending:
                        p.cancel()

                    if not done:
                        logger.warning("Proxy timeout at turn %d", turn_count)
                        break

                    if runtime_wait_task in done and proxy_task not in done:
                        if openclaw_proc is not None:
                            logger.info("OpenClaw exited (code=%s) before sending request", openclaw_proc.returncode)
                        else:
                            logger.info("Benchmark runtime exited before sending request")
                        break

                    req: ModelRequest = proxy_task.result()
                    logger.info("[run] Got proxy request turn=%d messages=%d", turn_count, len(req.messages))
                except asyncio.CancelledError:
                    logger.warning("Cancelled at turn %d", turn_count)
                    break

                if openclaw_proc is not None and openclaw_proc.returncode is not None:
                    logger.info("OpenClaw exited (code=%s)", openclaw_proc.returncode)
                    req.response_error = "agent process exited"
                    await proxy.send_response(req)
                    break

                n_msg = len(req.messages)
                thr = self.oc_config.stuck_retry_threshold
                if thr > 0 and turn_count > 0 and n_msg <= 2 and retry_count >= thr:
                    logger.warning(
                        "OpenClaw stuck in retry loop (turn=%d, retries=%d, threshold=%d), breaking",
                        turn_count, retry_count, thr,
                    )
                    req.response_error = "retry loop detected"
                    await proxy.send_response(req)
                    break
                if thr > 0 and turn_count > 0 and n_msg <= 2:
                    retry_count += 1
                else:
                    retry_count = 0

                chat_messages = self._prepare_messages(req.messages, req.tools)
                max_prompt_tokens = _max_prompt_tokens()
                if max_prompt_tokens > 0:
                    chat_messages = await _compact_messages_by_turn(
                        chat_messages,
                        max_prompt_tokens=max_prompt_tokens,
                        tools=req.tools,
                    )
                try:
                    logger.info("[run] Applying chat template (messages=%d, tools=%s)...", len(chat_messages), bool(req.tools))
                    prompt_token_ids = await self.apply_chat_template(chat_messages, tools=req.tools)
                    logger.info("[run] Chat template done, prompt_ids=%d", len(prompt_token_ids))
                except Exception as e:
                    logger.error("Chat template failed: %s", e)
                    req.response_error = str(e)
                    await proxy.send_response(req)
                    break

                if turn_count == 0:
                    if max_prompt_tokens > 0 and len(prompt_token_ids) > max_prompt_tokens:
                        logger.warning(
                            "[run] Turn-0 prompt (%d tokens) exceeds max_prompt_tokens (%d); truncating from left",
                            len(prompt_token_ids), max_prompt_tokens,
                        )
                        prompt_token_ids = list(prompt_token_ids[-max_prompt_tokens:])
                    all_prompt_ids = list(prompt_token_ids)
                elif not all_prompt_ids:
                    all_prompt_ids = list(prompt_token_ids)

                response_length = self.rollout_config.response_length
                if len(all_response_ids) >= response_length:
                    logger.warning(
                        "[run] Response budget exhausted before generation "
                        "(turn=%d, used=%d, limit=%d); ending episode.",
                        turn_count,
                        len(all_response_ids),
                        response_length,
                    )
                    break

                sampling_max_tokens = int(sampling_params.get("max_tokens", response_length) or response_length)
                try:
                    openclaw_max_tokens = int(getattr(req, "max_tokens", 0) or 0)
                except (TypeError, ValueError):
                    openclaw_max_tokens = 0
                requested_max_tokens = (
                    min(sampling_max_tokens, openclaw_max_tokens)
                    if openclaw_max_tokens > 0
                    else sampling_max_tokens
                )
                effective_max_tokens = _effective_generate_budget(
                    prompt_len=len(prompt_token_ids),
                    used_response_len=len(all_response_ids),
                    requested_max_tokens=requested_max_tokens,
                )
                if effective_max_tokens <= 0:
                    logger.warning(
                        "[run] Generation budget exhausted before vLLM call "
                        "(turn=%d, prompt=%d, used_response=%d, response_limit=%d, requested=%d); ending episode.",
                        turn_count,
                        len(prompt_token_ids),
                        len(all_response_ids),
                        response_length,
                        requested_max_tokens,
                    )
                    break

                gen_sampling_params = dict(sampling_params)
                gen_sampling_params["max_tokens"] = effective_max_tokens

                logger.info(
                    "[run] Calling server_manager.generate (turn=%d, prompt_ids=%d, max_tokens=%d, openclaw_req_max=%s, sampling_max=%s)...",
                    turn_count,
                    len(prompt_token_ids),
                    effective_max_tokens,
                    openclaw_max_tokens or None,
                    sampling_max_tokens,
                )
                gen_output = await self.server_manager.generate(
                    request_id=uuid.uuid4().hex,
                    prompt_ids=prompt_token_ids,
                    sampling_params=gen_sampling_params,
                )
                logger.info("[run] Generate done, got %d tokens", len(gen_output.token_ids) if gen_output.token_ids else 0)

                response_ids = list(gen_output.token_ids)
                response_logprobs = list(gen_output.log_probs) if gen_output.log_probs else [0.0] * len(response_ids)

                response_text = self.tokenizer.decode(response_ids, skip_special_tokens=False)
                clean_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

                # Env tokens = everything added by environment between turns
                if turn_count > 0:
                    env_token_ids = prompt_token_ids[len(all_prompt_ids) + len(all_response_ids):]
                    all_response_ids.extend(env_token_ids)
                    all_response_mask.extend([0] * len(env_token_ids))
                    all_response_logprobs.extend([0.0] * len(env_token_ids))

                all_response_ids.extend(response_ids)
                all_response_mask.extend([1] * len(response_ids))
                all_response_logprobs.extend(response_logprobs)

                turns.append(TurnRecord(
                    turn_index=turn_count,
                    messages=chat_messages,
                    prompt_ids=prompt_token_ids,
                    response_ids=response_ids,
                    response_text=clean_text,
                    response_logprobs=response_logprobs,
                ))

                tool_calls = self._parse_tool_calls(response_text)

                content_for_openclaw = self._strip_tool_tags(clean_text)
                logger.info("[run] Turn %d: tool_calls=%s, finish=%s, content_len=%d",
                            turn_count, bool(tool_calls),
                            "tool_calls" if tool_calls else "stop",
                            len(content_for_openclaw))
                if tool_calls:
                    for i, tc in enumerate(tool_calls):
                        f = tc.get("function", {})
                        logger.info("[run]   tc[%d] id=%s name=%s args_len=%d args_preview=%.200s",
                                    i, tc.get("id"), f.get("name"), len(f.get("arguments", "")),
                                    f.get("arguments", ""))
                else:
                    logger.info("[run]   no_tool_call content_preview=%.160s", content_for_openclaw[:160] if content_for_openclaw else "")
                logger.debug("[run] raw_text_preview=%.300s", response_text[:300] if response_text else "")

                req.response_text = content_for_openclaw
                req.response_tool_calls = tool_calls
                req.finish_reason = "tool_calls" if tool_calls else "stop"
                await proxy.send_response(req)

                messages.append({"role": "assistant", "content": clean_text, "tool_calls": tool_calls or []})
                turn_count += 1

                total_resp_len = len(all_response_ids)
                resp_budget = self.rollout_config.response_length
                if total_resp_len > resp_budget > 0:
                    logger.warning(
                        "[run] Response buffer exceeded budget after turn %d (%d > %d); compacting history",
                        turn_count,
                        total_resp_len,
                        resp_budget,
                    )
                    compacted = compact_turn_history(
                        turns=turns,
                        response_ids=all_response_ids,
                        response_mask=all_response_mask,
                        response_logprobs=all_response_logprobs,
                        max_response_tokens=resp_budget,
                    )
                    turns = compacted.turns
                    all_response_ids = compacted.response_ids
                    all_response_mask = compacted.response_mask
                    all_response_logprobs = compacted.response_logprobs
                    logger.info(
                        "[run] Compaction dropped %d turns / %d tokens; kept %d tokens",
                        compacted.dropped_turns,
                        compacted.dropped_tokens,
                        len(all_response_ids),
                    )

            if openclaw_proc and openclaw_proc.returncode is None:
                try:
                    await asyncio.wait_for(openclaw_proc.wait(), timeout=30)
                except asyncio.TimeoutError:
                    openclaw_proc.kill()
                    await openclaw_proc.wait()
            if benchmark_runtime_task is not None:
                try:
                    benchmark_execution_result = await asyncio.wait_for(
                        benchmark_runtime_task,
                        timeout=float(os.environ.get("PINCHBENCH_BENCHMARK_RUNTIME_JOIN_TIMEOUT", "60")),
                    )
                except asyncio.TimeoutError:
                    logger.warning("Benchmark runtime did not finish after model loop")

        finally:
            if reverse_tunnel_proc is not None and reverse_tunnel_proc.returncode is None:
                reverse_tunnel_proc.terminate()
                try:
                    await asyncio.wait_for(reverse_tunnel_proc.wait(), timeout=5)
                except asyncio.TimeoutError:
                    reverse_tunnel_proc.kill()
                    await reverse_tunnel_proc.wait()
            await proxy.drain()
            await proxy.stop()

        t_gen = time.time() - t_start
        metrics["generate_sequences"] = t_gen
        metrics["num_preempted"] = -1

        # Compute reward
        if benchmark_execution_result is not None:
            transcript_raw = list(benchmark_execution_result.get("transcript") or [])
            task_workspace_path = str(benchmark_execution_result.get("workspace") or "")
            terminal_success = self._run_grading_execution_result(
                task_id,
                benchmark_execution_result,
                extra_info=extra_info,
            )
        else:
            transcript_raw = self._load_openclaw_transcript(session_id, task_id)
            task_workspace_path = str(Path(self.oc_config.workspace_base) / workspace_rel) if self.oc_config.workspace_base else ""
            terminal_success = self._run_grading(
                task_id,
                transcript_raw,
                workspace_rel=workspace_rel,
                extra_info=extra_info,
            )
        terminal_reward_raw = 1.0 if terminal_success else 0.0  # {0,+1}: no negative gradient for failures
        terminal_reward_weight = float(os.environ.get("PINCHBENCH_TERMINAL_REWARD_WEIGHT", "0.3"))
        terminal_reward = terminal_reward_weight * terminal_reward_raw

        trajectory_for_reward = self._transcript_to_messages(transcript_raw) or messages
        if turns:
            trajectory_for_reward = _trim_trajectory_to_last_turns(
                trajectory_for_reward,
                keep_assistant_turns=len(turns),
            )
        episode_tags = _task16_episode_tags(trajectory_for_reward, task_workspace_path) if task_id == "task_16_email_triage" else {}
        per_turn_rewards = await self._compute_rewards(
            trajectory_for_reward, terminal_success, task_id, task_prompt,
            workspace_path=task_workspace_path,
            extra_info=extra_info,
        )
        if turns and len(per_turn_rewards) > len(turns):
            per_turn_rewards = per_turn_rewards[-len(turns):]
        elif turns and len(per_turn_rewards) < len(turns):
            per_turn_rewards = [0.0] * (len(turns) - len(per_turn_rewards)) + per_turn_rewards
        # per_turn_rewards already includes terminal_reward on the last turn
        # (added by compute_episode_rewards_async), so do NOT add it again.
        # Normalize per-task EMA so tasks with different success rates don't
        # pollute each other's advantage estimates via the global veRL baseline.
        per_turn_rewards = _normalize_turn_rewards(task_id, per_turn_rewards)
        total_reward = sum(per_turn_rewards)
        process_reward = total_reward - terminal_reward

        if task_id == "task_18_spreadsheet_summary":
            try:
                from rl.agent_loop.task18_event_reward.reward_task18_event_only import (
                    extract_tool_calls as _extract_t18_tool_calls,
                    task18_event_breakdown as _task18_event_breakdown,
                )

                assistant_turns = [t for t in trajectory_for_reward if t.get("role") == "assistant"]
                for idx, turn in enumerate(assistant_turns):
                    prev_turns = assistant_turns[:idx]
                    tool_result = None
                    turn_pos = trajectory_for_reward.index(turn)
                    for t in trajectory_for_reward[turn_pos + 1 :]:
                        if t.get("role") == "tool":
                            tool_result = t.get("content", "")
                            break
                        if t.get("role") == "assistant":
                            break
                    events = _task18_event_breakdown(turn, trajectory_for_reward[:turn_pos], tool_result)
                    calls = _extract_t18_tool_calls(turn)
                    if calls or events:
                        logger.info(
                            "[task18-turn] idx=%d tool_calls=%s events=%s turn_reward=%.3f",
                            idx,
                            [
                                {
                                    "name": c.get("name", ""),
                                    "args": str(c.get("arguments", ""))[:160],
                                }
                                for c in calls
                            ],
                            events,
                            per_turn_rewards[idx] if idx < len(per_turn_rewards) else 0.0,
                        )
            except Exception as e:
                logger.warning("[task18-turn] reward event logging failed: %s", e)

        logger.info(
            "Reward: total=%.2f process=%.2f terminal=%.1f per_turn=%s turns=%d mode=%s",
            total_reward,
            process_reward,
            terminal_reward,
            per_turn_rewards,
            turn_count,
            self.oc_config.reward_mode,
        )
        if episode_tags:
            logger.info("[task16-episode-tags] %s", episode_tags)

        # Assign per-turn rewards to model-generated assistant spans. The old
        # im_end-only mode is still available via PINCHBENCH_REWARD_ASSIGNMENT.
        # terminal_reward=0 here because it's already in per_turn_rewards.
        reward_at_tokens = self._assign_rewards(
            all_response_ids, all_response_mask, per_turn_rewards, 0.0,
        )
        if os.environ.get("PINCHBENCH_REWARD_DEBUG", "0").lower() in {"1", "true", "yes", "on"}:
            reward_nonzero = [
                (idx, round(value, 4))
                for idx, value in enumerate(reward_at_tokens)
                if value != 0.0
            ]
            logger.info(
                "Reward token alignment: task=%s assignment=%s turn_rewards=%s nonzero=%s response_len=%d mask_model_tokens=%d",
                task_id,
                os.environ.get("PINCHBENCH_REWARD_ASSIGNMENT", "turn_broadcast"),
                per_turn_rewards,
                reward_nonzero[:20],
                len(all_response_ids),
                sum(all_response_mask),
            )

        response_length = self.rollout_config.response_length

        # Ensure non-empty ids for veRL postprocessing (tokenizer.pad needs valid input)
        if not all_prompt_ids:
            eos_id = self.tokenizer.eos_token_id or 0
            all_prompt_ids = [eos_id]
        if not all_response_ids:
            eos_id = self.tokenizer.eos_token_id or 0
            all_response_ids = [eos_id]
            all_response_mask = [0]
            all_response_logprobs = [0.0]

        output = AgentLoopOutput(
            prompt_ids=all_prompt_ids,
            response_ids=all_response_ids[:response_length],
            response_mask=all_response_mask[:response_length],
            response_logprobs=all_response_logprobs[:response_length] if all_response_logprobs else None,
            reward_score=total_reward,
            num_turns=turn_count * 2 + 1,
            metrics=AgentLoopMetrics(
                generate_sequences=t_gen,
                tool_calls=turn_count,
                num_preempted=-1,
            ),
            extra_fields={
                "turn_scores": per_turn_rewards,
                "total_reward": total_reward,
                "process_reward": process_reward,
                "tool_rewards": reward_at_tokens[:response_length],
                "task_id": task_id,
                "terminal_success": terminal_success,
                "terminal_reward": terminal_reward,
                "reward_mode": self.oc_config.reward_mode,
                "trajectory": trajectory_for_reward,
                "prompt_group": extra_info.get("prompt_group", ""),
                "reward_rubric": extra_info.get("reward_rubric", {}),
                "episode_tags": episode_tags,
            },
        )
        if not use_benchmark_runtime:
            self._cleanup_remote_episode(task_id, session_id, workspace_rel)
        return output

    # ── OpenClaw process management ──

    @staticmethod
    def _episode_workspace_rel(task_id: str, session_id: str) -> str:
        safe_task = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in task_id)
        safe_session = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in session_id)
        return f"{safe_task}/{safe_session}"

    @staticmethod
    def _remote_openclaw_home(session_id: str) -> str:
        safe_session = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in session_id)
        return f"/tmp/pinchbench/openclaw_home/{safe_session}"

    async def _start_reverse_tunnel(self, proxy_port: int) -> asyncio.subprocess.Process:
        proc = await asyncio.create_subprocess_exec(
            "ssh",
            "-N",
            "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-o", "ExitOnForwardFailure=yes",
            "-R", f"{proxy_port}:127.0.0.1:{proxy_port}",
            "-i", self.oc_config.ssh_key,
            "-p", str(self.oc_config.ssh_port),
            f"{self.oc_config.user}@{self.oc_config.host}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await asyncio.sleep(1.0)
        if proc.returncode is not None:
            raise RuntimeError(f"reverse tunnel exited early with code {proc.returncode}")
        logger.info("Started benchmark-runtime reverse tunnel pid=%s port=%d", proc.pid, proxy_port)
        return proc

    def _benchmark_model_id(self) -> str:
        for name in ("PINCHBENCH_BENCHMARK_MODEL_ID", "OPENCLAW_MODEL", "MODEL", "VERL_MODEL"):
            value = os.environ.get(name, "").strip()
            if value:
                return value.rsplit("/", 1)[-1]
        return "Qwen3-4B"

    def _benchmark_task_for_episode(
        self,
        task_id: str,
        task_prompt: str,
        extra_info: Optional[dict[str, Any]],
    ):
        scripts_dir = Path(self.oc_config.pinchbench_dir) / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from lib_tasks import Task, TaskLoader, resolve_task_markdown_path

        tasks_dir = _task_markdown_dir(self.oc_config.pinchbench_dir)
        task_file = resolve_task_markdown_path(tasks_dir, task_id)
        base_task = TaskLoader(tasks_dir).load_task(task_file)
        workspace_files = self._workspace_files_from_extra_info(extra_info) or list(base_task.workspace_files)
        prompt = task_prompt or base_task.prompt
        return Task(
            task_id=base_task.task_id,
            name=base_task.name,
            category=base_task.category,
            grading_type=base_task.grading_type,
            timeout_seconds=base_task.timeout_seconds,
            workspace_files=workspace_files,
            prompt=prompt,
            expected_behavior=base_task.expected_behavior,
            grading_criteria=base_task.grading_criteria,
            automated_checks=base_task.automated_checks,
            llm_judge_rubric=base_task.llm_judge_rubric,
            grading_weights=base_task.grading_weights,
            file_path=base_task.file_path,
            frontmatter=dict(base_task.frontmatter or {}),
        )

    def _execute_with_benchmark_runtime(
        self,
        task_id: str,
        task_prompt: str,
        session_id: str,
        proxy_url: str,
        extra_info: Optional[dict[str, Any]],
        workspace_rel: str,
    ) -> dict[str, Any]:
        scripts_dir = Path(self.oc_config.pinchbench_dir) / "scripts"
        if str(scripts_dir) not in sys.path:
            sys.path.insert(0, str(scripts_dir))
        from lib_agent import ensure_agent_exists, execute_openclaw_task

        task = self._benchmark_task_for_episode(task_id, task_prompt, extra_info)
        agent_id = f"rl-bench-{task_id}-{session_id[:8]}"
        model_id = self._benchmark_model_id()
        benchmark_run_id = f"{uuid.uuid4().int % 10**16:016d}-{session_id}"
        run_head = benchmark_run_id.split("-", 1)[0]
        run_root = Path(os.environ.get("PINCHBENCH_RUN_ROOT", "/tmp/pinchbench"))
        workspace = run_root / run_head / "agent_workspace"
        os.environ.setdefault("PINCHBENCH_OPENCLAW_CONTEXT_WINDOW", os.environ.get("VLLM_MAX_MODEL_LEN", "32768"))
        os.environ.setdefault("PINCHBENCH_OPENCLAW_MAX_TOKENS", os.environ.get("PINCHBENCH_TASK16_MAX_TOKENS_PER_TURN", "4096"))
        os.environ.setdefault("OPENCLAW_MODEL_REASONING", "0")
        os.environ.setdefault("PINCHBENCH_NO_BANNER", "1")
        old_remote_home = os.environ.get("PINCHBENCH_REMOTE_OPENCLAW_HOME")
        if old_remote_home is None:
            os.environ["PINCHBENCH_REMOTE_OPENCLAW_HOME"] = self._remote_openclaw_home(session_id)
        ensure_agent_exists(
            agent_id,
            model_id,
            workspace,
            base_url=proxy_url,
            api_key="dummy",
        )
        output_dir_raw = os.environ.get("PINCHBENCH_BENCHMARK_TRANSCRIPT_DIR", "").strip()
        output_dir = Path(output_dir_raw) if output_dir_raw else None
        try:
            result = execute_openclaw_task(
                task=task,
                agent_id=agent_id,
                model_id=model_id,
                run_id=benchmark_run_id,
                timeout_multiplier=float(os.environ.get("PINCHBENCH_BENCHMARK_TIMEOUT_MULTIPLIER", "1.0")),
                skill_dir=Path(self.oc_config.pinchbench_dir),
                output_dir=output_dir,
                verbose=os.environ.get("PINCHBENCH_BENCHMARK_RUNTIME_VERBOSE", "0").strip().lower()
                in {"1", "true", "yes", "on"},
            )
        finally:
            if old_remote_home is None:
                os.environ.pop("PINCHBENCH_REMOTE_OPENCLAW_HOME", None)
            else:
                os.environ["PINCHBENCH_REMOTE_OPENCLAW_HOME"] = old_remote_home
        logger.info(
            "Benchmark runtime finished task=%s status=%s workspace=%s transcript=%d",
            task_id,
            result.get("status"),
            result.get("workspace"),
            len(result.get("transcript") or []),
        )
        return result

    async def _start_local_openclaw(
        self,
        prompt: str,
        session_id: str,
        proxy_url: str,
        task_id: str,
        extra_info: Optional[dict[str, Any]] = None,
        workspace_rel: Optional[str] = None,
    ) -> asyncio.subprocess.Process:
        agent_id = f"rl-{task_id}-{session_id[:8]}"
        workspace = Path(self.oc_config.workspace_base) / (workspace_rel or self._episode_workspace_rel(task_id, session_id))
        self._setup_agent_local(agent_id, proxy_url, workspace, task_id=task_id, task_prompt=prompt)
        self._prepare_workspace(task_id, workspace, extra_info=extra_info)

        proc = await asyncio.create_subprocess_exec(
            "openclaw", "agent", "--agent", agent_id,
            "--session-id", session_id, "--message", prompt, "--local",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            cwd=str(workspace),
        )
        logger.info("Started local OpenClaw pid=%s task=%s", proc.pid, task_id)
        return proc

    async def _start_remote_openclaw(
        self,
        prompt: str,
        session_id: str,
        proxy_port: int,
        task_id: str,
        extra_info: Optional[dict[str, Any]] = None,
        workspace_rel: Optional[str] = None,
    ) -> asyncio.subprocess.Process:
        agent_id = f"rl-{task_id}-{session_id[:8]}"
        workspace = f"{self.oc_config.workspace_base}/{workspace_rel or self._episode_workspace_rel(task_id, session_id)}"
        remote_home = self._remote_openclaw_home(session_id)

        self._preflight_remote_skill_pool(task_id=task_id, task_prompt=prompt)
        self._prepare_remote_workspace(task_id, workspace, extra_info=extra_info)

        # SSH reverse tunnel: ECS localhost:<proxy_port> -> RunPod localhost:<proxy_port>
        # RunPod doesn't expose arbitrary ports to the public internet,
        # so OpenClaw on ECS connects to its own localhost via the tunnel.
        local_proxy_url = f"http://127.0.0.1:{proxy_port}/v1"

        setup_cmd = self._build_remote_setup(
            agent_id,
            local_proxy_url,
            workspace,
            task_id,
            openclaw_home=remote_home,
            disable_skills=self._disable_default_skills_for_task(task_id, prompt),
        )
        escaped_prompt = prompt.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
        run_cmd = (
            f"openclaw agent --agent {agent_id} --session-id {session_id} "
            f'--message "{escaped_prompt}" --local'
        )
        remote_prefix = self._build_remote_activate_prefix()
        remote_command = " && ".join(filter(None, [
            remote_prefix,
            setup_cmd,
            f"cd {workspace} && {run_cmd}",
        ]))

        proc = await asyncio.create_subprocess_exec(
            "ssh", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=10",
            "-o", "ExitOnForwardFailure=yes",
            "-R", f"{proxy_port}:127.0.0.1:{proxy_port}",
            "-i", self.oc_config.ssh_key, "-p", str(self.oc_config.ssh_port),
            f"{self.oc_config.user}@{self.oc_config.host}",
            remote_command,
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        logger.info(
            "Started remote OpenClaw on %s pid=%s task=%s (reverse tunnel :%d)",
            self.oc_config.host, proc.pid, task_id, proxy_port,
        )
        return proc

    def _preflight_remote_skill_pool(self, *, task_id: str, task_prompt: str) -> None:
        """Fail fast when a web-heavy task cannot see the expected skill pool."""
        requirements = self._web_skill_requirements(task_id=task_id, task_prompt=task_prompt)
        if not requirements:
            return

        remote_prefix = self._build_remote_activate_prefix()
        remote_cmd = "openclaw skills list --eligible --json"
        if remote_prefix:
            remote_cmd = f"{remote_prefix} && {remote_cmd}"

        logger.info(
            "Preflighting ECS OpenClaw skills for task=%s (search=%s fetch=%s)",
            task_id,
            requirements["require_search"],
            requirements["require_fetch"],
        )
        result = _run_subprocess_with_retry(
            [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=10",
                "-i", self.oc_config.ssh_key,
                "-p", str(self.oc_config.ssh_port),
                f"{self.oc_config.user}@{self.oc_config.host}",
                remote_cmd,
            ],
            label="ECS skill preflight",
            timeout=30,
        )
        if result.returncode != 0:
            raise RuntimeError(
                "ECS OpenClaw skill preflight failed "
                f"(exit={result.returncode}): {result.stderr.strip() or result.stdout.strip()}"
            )

        try:
            payload = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(
                f"ECS OpenClaw skill preflight returned non-JSON output: {exc}"
            ) from exc

        ready_skills = {
            item.get("name")
            for item in payload.get("skills", [])
            if isinstance(item, dict) and item.get("eligible") and not item.get("disabled")
        }
        web_search_aliases = self._skill_aliases(
            "OPENCLAW_WEB_SEARCH_SKILLS",
            DEFAULT_WEB_SEARCH_SKILLS,
        )
        web_fetch_aliases = self._skill_aliases(
            "OPENCLAW_WEB_FETCH_SKILLS",
            DEFAULT_WEB_FETCH_SKILLS,
        )

        search_match = self._first_matching_alias(ready_skills, web_search_aliases)
        fetch_match = self._first_matching_alias(ready_skills, web_fetch_aliases)

        logger.info(
            "ECS OpenClaw skill pool: ready=%s, web_search=%s, web_fetch=%s",
            sorted(ready_skills),
            search_match or "missing",
            fetch_match or "missing",
        )

        missing_required: list[str] = []
        if requirements["require_search"] and not search_match:
            missing_required.append(
                "web_search (acceptable names: " + ", ".join(web_search_aliases) + ")"
            )
        if requirements["require_fetch"] and not fetch_match:
            missing_required.append(
                "web_fetch (acceptable names: " + ", ".join(web_fetch_aliases) + ")"
            )
        if missing_required:
            raise RuntimeError(
                "ECS OpenClaw skill pool mismatch for "
                f"{task_id}: missing required capability(s): {', '.join(missing_required)}. "
                f"Ready skills: {', '.join(sorted(ready_skills)) or '<none>'}. "
                "If you intentionally renamed the skills, remap them with "
                "OPENCLAW_WEB_SEARCH_SKILLS / OPENCLAW_WEB_FETCH_SKILLS."
            )

        if requirements["recommend_fetch"] and not fetch_match:
            logger.warning(
                "Task %s prefers a web_fetch-style skill, but none is ready. "
                "This is not fatal, but it may weaken the trajectory relative to the GT reference.",
                task_id,
            )

    def _web_skill_requirements(self, *, task_id: str, task_prompt: str) -> Optional[dict[str, bool]]:
        text = f"{task_id}\n{task_prompt}".lower()

        stock_task = (
            "stock price" in text and "apple" in text
        ) or "stock_report.txt" in text
        if stock_task:
            return {
                "require_search": True,
                "require_fetch": True,
                "recommend_fetch": True,
            }

        market_research_task = (
            "observability and apm" in text
            or "application performance monitoring" in text
            or "competitive landscape analysis" in text
            or "market research" in text
        )
        if market_research_task:
            return {
                "require_search": True,
                "require_fetch": False,
                "recommend_fetch": True,
            }

        polymarket_task = (
            "polymarket" in text
            or "prediction markets" in text
            or "trending market" in text
            or "polymarket_briefing" in text
        )
        if polymarket_task:
            return {
                "require_search": True,
                "require_fetch": False,
                "recommend_fetch": True,
            }

        return None

    def _skill_aliases(self, env_name: str, defaults: tuple[str, ...]) -> list[str]:
        raw = os.environ.get(env_name, "").strip()
        if raw:
            aliases = [item.strip() for item in raw.split(",") if item.strip()]
            if aliases:
                return aliases
        return list(defaults)

    @staticmethod
    def _first_matching_alias(ready_skills: set[str], aliases: list[str]) -> Optional[str]:
        for alias in aliases:
            if alias in ready_skills:
                return alias
        return None

    def _setup_agent_local(
        self,
        agent_id: str,
        proxy_url: str,
        workspace: Path,
        *,
        task_id: str,
        task_prompt: str,
    ) -> None:
        workspace.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["openclaw", "agents", "delete", agent_id, "--force"],
            capture_output=True, text=True, check=False,
        )
        subprocess.run(
            ["openclaw", "agents", "add", agent_id,
             "--model", "verl/verl-proxy", "--workspace", str(workspace), "--non-interactive"],
            capture_output=True, text=True, check=False,
        )
        self._patch_agent_skills_local(
            agent_id,
            disable_skills=self._disable_default_skills_for_task(task_id, task_prompt),
        )
        agent_dir = Path.home() / ".openclaw" / "agents" / agent_id / "agent"
        agent_dir.mkdir(parents=True, exist_ok=True)
        models = {
            "mode": "replace",
            "providers": {"verl": {
                "baseUrl": proxy_url, "apiKey": "dummy", "api": "openai-completions",
                "models": [{"id": "verl-proxy", "name": "verl-proxy", "reasoning": False}],
            }},
            "defaultProvider": "verl", "defaultModel": "verl/verl-proxy",
        }
        (agent_dir / "models.json").write_text(json.dumps(models, indent=2), "utf-8")
        auth = {
            "version": 1,
            "profiles": {
                "verl-default": {
                    "type": "api_key",
                    "key": "dummy",
                    "provider": "verl",
                }
            },
        }
        (agent_dir / "auth-profiles.json").write_text(json.dumps(auth, indent=2), "utf-8")
        sessions_file = Path.home() / ".openclaw" / "agents" / agent_id / "sessions" / "sessions.json"
        if sessions_file.exists():
            sessions_file.unlink()

    def _disable_default_skills_for_task(self, task_id: str, task_prompt: str = "") -> bool:
        raw = os.environ.get("PINCHBENCH_DISABLE_DEFAULT_SKILLS", "1").strip().lower()
        if raw not in {"1", "true", "yes", "on"}:
            return False
        return self._web_skill_requirements(task_id=task_id, task_prompt=task_prompt) is None

    def _patch_agent_skills_local(self, agent_id: str, *, disable_skills: bool) -> None:
        if not disable_skills:
            return
        path = Path.home() / ".openclaw" / "openclaw.json"
        try:
            data = json.loads(path.read_text("utf-8"))
            agents = data.get("agents", {}).get("list", [])
            normalized = agent_id.replace(":", "-").lower()
            for entry in agents:
                if not isinstance(entry, dict):
                    continue
                if entry.get("name") == agent_id or entry.get("id") == agent_id or str(entry.get("id", "")).lower() == normalized:
                    entry["skills"] = []
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", "utf-8")
            os.replace(tmp, path)
        except Exception as exc:
            logger.warning("Failed to disable default skills for local agent %s: %s", agent_id, exc)

    def _build_remote_setup(
        self,
        agent_id: str,
        proxy_url: str,
        workspace: str,
        task_id: str,
        *,
        openclaw_home: str,
        disable_skills: bool = False,
    ) -> str:
        import base64
        models_json = json.dumps({
            "mode": "replace",
            "providers": {"verl": {
                "baseUrl": proxy_url, "apiKey": "dummy", "api": "openai-completions",
                "models": [{"id": "verl-proxy", "name": "verl-proxy", "reasoning": False}],
            }},
            "defaultProvider": "verl", "defaultModel": "verl/verl-proxy",
        }, indent=2)
        auth_json = json.dumps({
            "version": 1,
            "profiles": {
                "verl-default": {
                    "type": "api_key",
                    "key": "dummy",
                    "provider": "verl",
                }
            },
        }, indent=2)
        b64_models = base64.b64encode(models_json.encode()).decode()
        b64_auth = base64.b64encode(auth_json.encode()).decode()
        lock_file = "/tmp/openclaw_agents.lock"
        proxy_models_url = f"{proxy_url}/models"
        proxy_models_url_q = shlex.quote(proxy_models_url)
        # openclaw may normalize agent_id on `agents add` (truncates last char into `id`,
        # keeps original as `name`). Match by `name` to get the correct agentDir.
        get_agent_dir_py = (
            f'python3 -c "'
            f'import json; '
            f'd=json.load(open(\\\"$HOME/.openclaw/openclaw.json\\\")); '
            f'ms=[a for a in d[\\\"agents\\\"][\\\"list\\\"] if a.get(\\\"name\\\")==\\\"{agent_id}\\\" or a.get(\\\"id\\\")==\\\"{agent_id}\\\"]; '
            f'print(ms[0][\\\"agentDir\\\"]) if ms else print(\\\"$HOME/.openclaw/agents/{agent_id}/agent\\\")'
            f'"'
        )
        proxy_smoke_test_py = shlex.quote(
            (
                "import json, sys, urllib.request; "
                f"url={proxy_models_url!r}; "
                "data=json.load(urllib.request.urlopen(url, timeout=15)); "
                "ids=[item.get('id') for item in data.get('data', []) if isinstance(item, dict)]; "
                "print('[OC-SETUP] proxy_models=', ids); "
                "sys.exit(0 if 'verl-proxy' in ids else 2)"
            )
        )
        disable_skills_py = shlex.quote(
            (
                "import json, os, pathlib; "
                "p=pathlib.Path(os.path.expanduser('~/.openclaw/openclaw.json')); "
                "d=json.loads(p.read_text()) if p.exists() else {}; "
                "lst=d.get('agents',{}).get('list',[]); "
                f"aid={agent_id!r}; norm=aid.replace(':','-').lower(); "
                "\nfor e in lst:\n"
                "    if isinstance(e, dict) and (e.get('name')==aid or e.get('id')==aid or str(e.get('id','')).lower()==norm):\n"
                "        e['skills']=[]\n"
                "p.write_text(json.dumps(d, indent=2, ensure_ascii=False)+'\\n')"
            )
        )
        scrub_cmd = " && ".join(
            [f"rm -rf {shlex.quote(workspace)}/{shlex.quote(rel)}" for rel in self._workspace_scrub_paths()]
        )
        home_q = shlex.quote(openclaw_home)
        return " && ".join([
            f"rm -rf {home_q}",
            f"mkdir -p {workspace} {home_q}/.openclaw",
            f"export HOME={home_q} OPENCLAW_HOME={home_q}",
            (
                "python3 -c "
                + shlex.quote(
                    "import json, os, pathlib; "
                    "p=pathlib.Path(os.environ['HOME'])/'.openclaw'/'openclaw.json'; "
                    "p.parent.mkdir(parents=True, exist_ok=True); "
                    "p.write_text(json.dumps({"
                    "'gateway': {'port': 18789, 'bind': 'loopback'}, "
                    "'agents': {'defaults': {'skipBootstrap': True, 'skills': []}}"
                    "}, indent=2)+'\\n')"
                )
            ),
            "echo '[OC-SETUP] HOME='$HOME",
            "echo '[OC-SETUP] openclaw.json:' && cat $HOME/.openclaw/openclaw.json",
            "echo '[OC-SETUP] openclaw_bin='$(command -v openclaw)",
            "openclaw --version 2>/dev/null || true",
            f"(flock {lock_file} openclaw agents delete {agent_id} --force >/dev/null 2>&1 || true)",
            f"(flock {lock_file} openclaw agents add {agent_id} --model verl/verl-proxy --workspace {workspace} --non-interactive 2>&1 || echo '[WARN] openclaw agents add failed for {agent_id}' >&2)",
            *([f"python3 -c {disable_skills_py}", f"echo '[OC-SETUP] disabled default skills for {agent_id}'"] if disable_skills else []),
            scrub_cmd,
            f"_adir=$({get_agent_dir_py})",
            f"mkdir -p $_adir",
            f"echo {b64_models} | base64 -d > $_adir/models.json",
            f"echo {b64_auth} | base64 -d > $_adir/auth-profiles.json",
            f"rm -f $_adir/../sessions/sessions.json",
            f"echo '[OC-SETUP] agent_dir='$_adir",
            "echo '[OC-SETUP] models.json:' && cat $_adir/models.json",
            "echo '[OC-SETUP] auth-profiles.json:' && cat $_adir/auth-profiles.json",
            f"echo '[OC-SETUP] proxy_models_url={proxy_models_url}'",
            f"python3 -c {proxy_smoke_test_py} || (echo '[OC-SETUP] proxy smoke test failed for {proxy_models_url_q}' >&2; exit 97)",
        ])

    def _build_remote_activate_prefix(self) -> str:
        """Build an optional shell prefix for ECS-side OpenClaw activation.

        The default is empty because the ECS install is expected to expose the
        `openclaw` CLI directly on PATH. If a site-specific activation command is
        required, pass it via OPENCLAW_REMOTE_ACTIVATE_CMD.
        """
        if self.oc_config.remote_activate_cmd.strip():
            return self.oc_config.remote_activate_cmd.strip()
        remote_bin_dir = os.environ.get("OPENCLAW_REMOTE_BIN_DIR", "").strip()
        if remote_bin_dir:
            return f'export PATH="{remote_bin_dir}:$PATH"'
        return ""

    def _cleanup_remote_episode(self, task_id: str, session_id: str, workspace_rel: str) -> None:
        if self.oc_config.host in ("localhost", "127.0.0.1"):
            return
        cleanup_agent = os.environ.get("PINCHBENCH_CLEANUP_REMOTE_AGENT", "1").strip().lower() in {
            "1", "true", "yes", "on",
        }
        cleanup_workspace = os.environ.get("PINCHBENCH_CLEANUP_REMOTE_WORKSPACE", "1").strip().lower() in {
            "1", "true", "yes", "on",
        }
        if not cleanup_agent and not cleanup_workspace:
            return

        agent_id = f"rl-{task_id}-{session_id[:8]}"
        workspace = f"{self.oc_config.workspace_base}/{workspace_rel}"
        remote_home = self._remote_openclaw_home(session_id)
        parts: list[str] = []
        if cleanup_agent:
            parts.append(
                f"export HOME={shlex.quote(remote_home)} OPENCLAW_HOME={shlex.quote(remote_home)}; "
                f"openclaw agents delete {shlex.quote(agent_id)} --force >/dev/null 2>&1 || true"
            )
        if cleanup_workspace:
            parts.append(f"rm -rf {shlex.quote(workspace)} {shlex.quote(remote_home)}")
        remote_prefix = self._build_remote_activate_prefix()
        remote_cmd = " ; ".join(parts)
        if remote_prefix:
            remote_cmd = f"{remote_prefix} && {remote_cmd}"
        result = _run_subprocess_with_retry(
            [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=10",
                "-i", self.oc_config.ssh_key,
                "-p", str(self.oc_config.ssh_port),
                f"{self.oc_config.user}@{self.oc_config.host}",
                remote_cmd,
            ],
            label="ECS episode cleanup",
            timeout=30,
        )
        if result.returncode != 0:
            logger.warning(
                "Remote episode cleanup failed for %s/%s: %s %s",
                task_id,
                session_id,
                result.stderr.strip(),
                result.stdout.strip(),
            )

    @staticmethod
    def _workspace_scrub_paths() -> list[str]:
        # OpenClaw can materialize generic bootstrap / memory scaffold files in the
        # workspace root. They are not part of benchmark task fixtures and can pull
        # the agent off-task (e.g. reading MEMORY.md / BOOTSTRAP.md instead of inbox/).
        return [
            "BOOTSTRAP.md",
            "SOUL.md",
            "USER.md",
            "IDENTITY.md",
            "HEARTBEAT.md",
            "TOOLS.md",
            "AGENTS.md",
            "MEMORY.md",
            "REPORT.md",
            "INBOX.md",
            "emails.md",
            "emails.txt",
            "inbox_emails.txt",
            "event_map.md",
            "report_schema.md",
            "triax_report.md",
            "tripe_report.md",
            "memory",
            ".openclaw",
            ".git",
        ]

    def _scrub_workspace(self, workspace: Path) -> None:
        try:
            for rel in self._workspace_scrub_paths():
                target = workspace / rel
                if target.is_dir():
                    shutil.rmtree(target, ignore_errors=True)
                elif target.exists():
                    target.unlink(missing_ok=True)
        except Exception as e:
            logger.warning("Local workspace scrub failed for %s: %s", workspace, e)

    def _workspace_files_from_extra_info(self, extra_info: Optional[dict[str, Any]]) -> list[dict[str, Any]]:
        if not isinstance(extra_info, dict):
            return []
        files = extra_info.get("workspace_files")
        if isinstance(files, (list, tuple)) or (hasattr(files, "tolist") and not isinstance(files, dict)):
            if hasattr(files, "tolist"):
                files = files.tolist()
            return [f for f in files if isinstance(f, dict)]
        if isinstance(files, dict):
            normalized: list[dict[str, Any]] = []
            for path, content in files.items():
                if isinstance(path, str):
                    normalized.append({"path": path, "content": "" if content is None else str(content)})
            return normalized
        return []

    def _write_workspace_files(self, workspace: Path, files: list[dict[str, Any]]) -> None:
        for f in files:
            if "content" in f:
                dest = workspace / str(f["path"])
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(str(f["content"]))
            elif "source" in f:
                src = Path(self.oc_config.pinchbench_dir) / "assets" / str(f["source"])
                dest = workspace / str(f.get("dest", f["source"]))
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(src.read_bytes())

    def _prepare_workspace(
        self,
        task_id: str,
        workspace: Path,
        extra_info: Optional[dict[str, Any]] = None,
    ) -> None:
        if not self.oc_config.pinchbench_dir:
            return
        try:
            dynamic_files = self._workspace_files_from_extra_info(extra_info)
            if dynamic_files:
                if workspace.exists():
                    shutil.rmtree(workspace)
                workspace.mkdir(parents=True, exist_ok=True)
                self._write_workspace_files(workspace, dynamic_files)
                self._scrub_workspace(workspace)
                logger.info(
                    "Seeded dynamic workspace_files for %s -> %s (%d files)",
                    task_id,
                    workspace,
                    len(dynamic_files),
                )
                return

            import sys
            scripts_dir = Path(self.oc_config.pinchbench_dir) / "scripts"
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            from lib_tasks import TaskLoader, resolve_task_markdown_path
            tasks_dir = _task_markdown_dir(self.oc_config.pinchbench_dir)
            loader = TaskLoader(tasks_dir)
            task_file = resolve_task_markdown_path(tasks_dir, task_id)
            if not task_file.exists():
                logger.error(
                    "Task markdown not found for %s (expected %s); cannot seed workspace_files",
                    task_id,
                    task_file,
                )
                return
            task = loader.load_task(task_file)
            if task is None:
                return
            if workspace.exists():
                shutil.rmtree(workspace)
            workspace.mkdir(parents=True, exist_ok=True)
            self._write_workspace_files(workspace, list(getattr(task, "workspace_files", [])))
            self._scrub_workspace(workspace)
        except Exception as e:
            logger.error("Workspace prep failed: %s", e)

    def _prepare_remote_workspace(
        self,
        task_id: str,
        workspace: str,
        extra_info: Optional[dict[str, Any]] = None,
    ) -> None:
        """Write task `workspace_files` onto ECS before OpenClaw runs.

        Local mode calls `_prepare_workspace` on the training host; remote mode
        seeds a temp tree locally and rsyncs it to the remote workspace.

        Tasks with empty ``workspace_files`` still get ``rm -rf <workspace> &&
        mkdir -p`` on ECS each episode so outputs (e.g. ``stock_report.txt``) do
        not linger across rollouts — aligned with benchmark ``prepare_task_workspace``.
        """
        if not self.oc_config.pinchbench_dir:
            return
        try:
            with tempfile.TemporaryDirectory(prefix="pinchbench_seed_") as td:
                local_root = Path(td) / task_id
                self._prepare_workspace(task_id, local_root, extra_info=extra_info)
                if not local_root.is_dir():
                    return
                has_files = any(p.is_file() for p in local_root.rglob("*"))
                ws = shlex.quote(workspace)
                reset = _run_subprocess_with_retry(
                    [
                        "ssh",
                        "-o", "StrictHostKeyChecking=no",
                        "-o", "ConnectTimeout=10",
                        "-i", self.oc_config.ssh_key,
                        "-p", str(self.oc_config.ssh_port),
                        f"{self.oc_config.user}@{self.oc_config.host}",
                        f"rm -rf {ws} && mkdir -p {ws}",
                    ],
                    label="ECS workspace reset",
                    timeout=30,
                )
                if reset.returncode != 0:
                    logger.error(
                        "Remote workspace reset failed for %s: %s %s",
                        task_id, reset.stderr, reset.stdout,
                    )
                    return
                if not has_files:
                    scrub_cmd = " && ".join(
                        [f"rm -rf {shlex.quote(workspace)}/{shlex.quote(rel)}" for rel in self._workspace_scrub_paths()]
                    )
                    _run_subprocess_with_retry(
                        [
                            "ssh",
                            "-o", "StrictHostKeyChecking=no",
                            "-o", "ConnectTimeout=10",
                            "-i", self.oc_config.ssh_key,
                            "-p", str(self.oc_config.ssh_port),
                            f"{self.oc_config.user}@{self.oc_config.host}",
                            scrub_cmd,
                        ],
                        timeout=30,
                        label="ECS empty workspace scrub",
                    )
                    logger.info(
                        "Remote workspace reset (empty seed / no workspace_files) for %s -> %s",
                        task_id,
                        workspace,
                    )
                    return
                rsync_cmd = [
                    _rsync_bin(),
                    "-az",
                    "--timeout=60",
                    "-e",
                    (
                        f"ssh -o StrictHostKeyChecking=no "
                        f"-i {self.oc_config.ssh_key} -p {self.oc_config.ssh_port}"
                    ),
                    f"{local_root}/",
                    f"{self.oc_config.user}@{self.oc_config.host}:{workspace}/",
                ]
                sync = _run_subprocess_with_retry(
                    rsync_cmd,
                    label="ECS workspace rsync",
                    timeout=120,
                )
                if sync.returncode != 0:
                    logger.error(
                        "Remote workspace rsync failed for %s: %s %s",
                        task_id, sync.stderr, sync.stdout,
                    )
                else:
                    scrub_cmd = " && ".join(
                        [f"rm -rf {shlex.quote(workspace)}/{shlex.quote(rel)}" for rel in self._workspace_scrub_paths()]
                    )
                    scrub = _run_subprocess_with_retry(
                        [
                            "ssh",
                            "-o", "StrictHostKeyChecking=no",
                            "-o", "ConnectTimeout=10",
                            "-i", self.oc_config.ssh_key,
                            "-p", str(self.oc_config.ssh_port),
                            f"{self.oc_config.user}@{self.oc_config.host}",
                            scrub_cmd,
                        ],
                        timeout=30,
                        label="ECS workspace scrub",
                    )
                    if scrub.returncode != 0:
                        logger.warning(
                            "Remote workspace scrub failed for %s: %s %s",
                            task_id, scrub.stderr, scrub.stdout,
                        )
                    logger.info("Seeded remote workspace for %s -> %s", task_id, workspace)
        except Exception as e:
            logger.error("Remote workspace prep failed for %s: %s", task_id, e)

    # ── Transcript / Grading ──

    def _load_openclaw_transcript(self, session_id: str, task_id: str) -> list[dict]:
        agents_dir = Path.home() / ".openclaw" / "agents"
        if not agents_dir.exists():
            return []
        prefix = f"rl-{task_id}"
        for agent_dir in agents_dir.iterdir():
            if not agent_dir.name.startswith(prefix):
                continue
            sessions_dir = agent_dir / "sessions"
            if not sessions_dir.exists():
                continue
            jsonl_files = sorted(sessions_dir.glob("*.jsonl"), key=lambda p: p.stat().st_mtime, reverse=True)
            if jsonl_files:
                transcript = []
                for line in jsonl_files[0].read_text("utf-8").splitlines():
                    line = line.strip()
                    if line:
                        try:
                            transcript.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                return transcript
        return []

    def _transcript_to_messages(self, transcript: list[dict]) -> list[dict]:
        msgs = []
        for entry in transcript:
            if entry.get("type") != "message":
                continue
            msg = entry.get("message", {})
            role = msg.get("role")
            if role in ("user", "assistant", "tool", "system"):
                msgs.append({
                    "role": role, "content": msg.get("content", ""),
                    "tool_calls": msg.get("tool_calls", []),
                })
        return msgs

    def _run_grading(
        self,
        task_id: str,
        transcript: list[dict],
        workspace_rel: Optional[str] = None,
        extra_info: Optional[dict[str, Any]] = None,
    ) -> bool:
        if not self.oc_config.pinchbench_dir:
            return False
        try:
            import sys, subprocess
            scripts_dir = Path(self.oc_config.pinchbench_dir) / "scripts"
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            from lib_tasks import TaskLoader, resolve_task_markdown_path
            from lib_grading import grade_task
            tasks_dir = _task_markdown_dir(self.oc_config.pinchbench_dir)
            loader = TaskLoader(tasks_dir)
            task_file = resolve_task_markdown_path(tasks_dir, task_id)
            if not task_file.exists():
                return False
            task = loader.load_task(task_file)
            if task is None:
                return False

            workspace_rel = workspace_rel or task_id
            workspace = Path(self.oc_config.workspace_base) / workspace_rel
            workspace.mkdir(parents=True, exist_ok=True)

            # Sync workspace files from ECS (files are created there by OpenClaw)
            if self.oc_config.host not in ("localhost", "127.0.0.1"):
                ssh_key = self.oc_config.ssh_key or "/root/.ssh/id_ed25519"
                remote_ws = f"{self.oc_config.workspace_base}/{workspace_rel}/"
                rsync_io_timeout = os.environ.get("PINCHBENCH_WORKSPACE_RSYNC_IO_TIMEOUT", "60")
                rsync_proc_timeout = float(os.environ.get("PINCHBENCH_WORKSPACE_RSYNC_PROC_TIMEOUT", "120"))
                try:
                    sync = _run_subprocess_with_retry(
                        [_rsync_bin(), "-az", f"--timeout={rsync_io_timeout}",
                         "-e", f"ssh -o StrictHostKeyChecking=no -i {ssh_key} -p {self.oc_config.ssh_port}",
                         f"{self.oc_config.user}@{self.oc_config.host}:{remote_ws}",
                         str(workspace) + "/"],
                        label="ECS workspace fetch for grading",
                        timeout=rsync_proc_timeout,
                    )
                    if sync.returncode != 0:
                        logger.warning(
                            "rsync workspace failed for %s: rc=%s stderr=%s stdout=%s",
                            task_id,
                            sync.returncode,
                            sync.stderr.strip(),
                            sync.stdout.strip(),
                        )
                    else:
                        logger.info("Synced workspace from ECS for %s", task_id)
                except Exception as e:
                    logger.warning("rsync workspace failed for %s: %s", task_id, e)

            execution_result = {
                "transcript": transcript,
                "workspace": str(workspace),
                "status": "success",
            }

            reward_rubric = extra_info.get("reward_rubric") if isinstance(extra_info, dict) else None
            if (
                task_id == "task_16_email_triage"
                and isinstance(reward_rubric, dict)
                and os.environ.get("PINCHBENCH_PER_INSTANCE_VERIFIER", "").strip().lower()
                in {"1", "true", "yes", "on"}
            ):
                from rl.agent_loop.per_instance_verifier import verify_task16_per_instance

                threshold = float(os.environ.get("PINCHBENCH_PER_INSTANCE_VERIFIER_THRESHOLD", "0.72"))
                result = verify_task16_per_instance(
                    workspace_path=workspace,
                    reward_rubric=reward_rubric,
                    threshold=threshold,
                    min_coverage=float(os.environ.get("PINCHBENCH_PER_INSTANCE_MIN_COVERAGE", "0.90")),
                    min_priority=float(os.environ.get("PINCHBENCH_PER_INSTANCE_MIN_PRIORITY", "0.80")),
                    min_category=float(os.environ.get("PINCHBENCH_PER_INSTANCE_MIN_CATEGORY", "0.75")),
                    min_required_fields=float(os.environ.get("PINCHBENCH_PER_INSTANCE_MIN_REQUIRED_FIELDS", "0.75")),
                )
                logger.info(
                    "Per-instance grading %s: score=%.3f passed=%s threshold=%.3f breakdown=%s notes=%s",
                    task_id,
                    result.score,
                    result.passed,
                    threshold,
                    result.breakdown,
                    result.notes,
                )
                return result.passed

            skill_dir = Path(self.oc_config.pinchbench_dir)
            from lib_grading import resolve_judge_backend_from_env, preflight_judge_connection
            judge_cfg = resolve_judge_backend_from_env(
                default_backend="api",
                default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            if judge_cfg["judge_backend"] == "api" and judge_cfg["judge_api_key"]:
                preflight_judge_connection(
                    judge_model=str(judge_cfg["judge_model"]),
                    judge_backend=str(judge_cfg["judge_backend"]),
                    judge_base_url=str(judge_cfg["judge_base_url"]),
                    judge_api_key=str(judge_cfg["judge_api_key"]),
                    timeout_seconds=30.0,
                )
            result = grade_task(
                task=task,
                execution_result=execution_result,
                skill_dir=skill_dir,
                judge_model=str(judge_cfg["judge_model"]),
                judge_backend=str(judge_cfg["judge_backend"]),
                judge_base_url=judge_cfg["judge_base_url"],
                judge_api_key=judge_cfg["judge_api_key"],
            )
            score = getattr(result, "score", 0.0)
            logger.info("Grading %s: score=%.2f", task_id, score)
            return score >= 0.5
        except Exception as e:
            logger.error("Grading failed for %s: %s", task_id, e)
            return False

    def _run_grading_execution_result(
        self,
        task_id: str,
        execution_result: dict[str, Any],
        extra_info: Optional[dict[str, Any]] = None,
    ) -> bool:
        if not self.oc_config.pinchbench_dir:
            return False
        try:
            scripts_dir = Path(self.oc_config.pinchbench_dir) / "scripts"
            if str(scripts_dir) not in sys.path:
                sys.path.insert(0, str(scripts_dir))
            from lib_tasks import TaskLoader, resolve_task_markdown_path
            from lib_grading import grade_task, resolve_judge_backend_from_env, preflight_judge_connection

            tasks_dir = _task_markdown_dir(self.oc_config.pinchbench_dir)
            task_file = resolve_task_markdown_path(tasks_dir, task_id)
            if not task_file.exists():
                return False
            task = TaskLoader(tasks_dir).load_task(task_file)

            reward_rubric = extra_info.get("reward_rubric") if isinstance(extra_info, dict) else None
            if (
                task_id == "task_16_email_triage"
                and isinstance(reward_rubric, dict)
                and os.environ.get("PINCHBENCH_PER_INSTANCE_VERIFIER", "").strip().lower()
                in {"1", "true", "yes", "on"}
            ):
                from rl.agent_loop.per_instance_verifier import verify_task16_per_instance

                result = verify_task16_per_instance(
                    workspace_path=Path(str(execution_result.get("workspace", ""))),
                    reward_rubric=reward_rubric,
                    threshold=float(os.environ.get("PINCHBENCH_PER_INSTANCE_VERIFIER_THRESHOLD", "0.72")),
                    min_coverage=float(os.environ.get("PINCHBENCH_PER_INSTANCE_MIN_COVERAGE", "0.90")),
                    min_priority=float(os.environ.get("PINCHBENCH_PER_INSTANCE_MIN_PRIORITY", "0.80")),
                    min_category=float(os.environ.get("PINCHBENCH_PER_INSTANCE_MIN_CATEGORY", "0.75")),
                    min_required_fields=float(os.environ.get("PINCHBENCH_PER_INSTANCE_MIN_REQUIRED_FIELDS", "0.75")),
                )
                logger.info("Per-instance grading %s: score=%.3f passed=%s", task_id, result.score, result.passed)
                return result.passed

            judge_cfg = resolve_judge_backend_from_env(
                default_backend="api",
                default_base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            )
            if judge_cfg["judge_backend"] == "api" and judge_cfg["judge_api_key"]:
                preflight_judge_connection(
                    judge_model=str(judge_cfg["judge_model"]),
                    judge_backend=str(judge_cfg["judge_backend"]),
                    judge_base_url=str(judge_cfg["judge_base_url"]),
                    judge_api_key=str(judge_cfg["judge_api_key"]),
                    timeout_seconds=30.0,
                )
            result = grade_task(
                task=task,
                execution_result=execution_result,
                skill_dir=Path(self.oc_config.pinchbench_dir),
                judge_model=str(judge_cfg["judge_model"]),
                judge_backend=str(judge_cfg["judge_backend"]),
                judge_base_url=judge_cfg["judge_base_url"],
                judge_api_key=judge_cfg["judge_api_key"],
            )
            score = getattr(result, "score", 0.0)
            logger.info("Benchmark-runtime grading %s: score=%.2f status=%s", task_id, score, execution_result.get("status"))
            return score >= 0.5
        except Exception as e:
            logger.error("Benchmark-runtime grading failed for %s: %s", task_id, e)
            return False

    # ── Reward computation ──

    def _get_vllm_base_url(self) -> str:
        """Get actual vLLM HTTP URL from server_manager (Ray-assigned dynamic port)."""
        try:
            addrs = list(self.server_manager._server_id_to_handle.keys())
            if addrs:
                url = f"http://{addrs[0]}/v1"
                logger.info("Detected vLLM URL: %s", url)
                return url
        except Exception:
            pass
        return self.oc_config.prm_vllm_base_url

    def _get_prm_base_url(self) -> str:
        """Choose judge endpoint without leaking external judges to rollout vLLM."""
        if self.oc_config.reward_mode == "self-judge":
            return self._get_vllm_base_url()
        return self.oc_config.prm_vllm_base_url

    async def _compute_rewards(
        self, trajectory: list[dict], terminal_success: bool, task_id: str, task_prompt: str,
        workspace_path: str = "",
        extra_info: Optional[dict[str, Any]] = None,
    ) -> list[float]:
        try:
            reward_module_override = os.environ.get("PINCHBENCH_REWARD_MODULE_OVERRIDE", "").strip()
            if reward_module_override:
                import importlib.util
                import sys

                module_path = Path(reward_module_override).expanduser().resolve()
                spec = importlib.util.spec_from_file_location(
                    "pinchbench_reward_override_module", module_path
                )
                if spec is None or spec.loader is None:
                    raise RuntimeError(f"invalid reward override module: {module_path}")
                module = importlib.util.module_from_spec(spec)
                sys.modules["pinchbench_reward_override_module"] = module
                spec.loader.exec_module(module)
                compute_episode_rewards_async = module.compute_episode_rewards_async
            else:
                from .reward import compute_episode_rewards_async
            vllm_url = self._get_prm_base_url()
            try:
                return await compute_episode_rewards_async(
                    trajectory, terminal_success, task_id,
                    task_prompt=task_prompt,
                    mode=self.oc_config.reward_mode,
                    vllm_base_url=vllm_url,
                    judge_model=self.oc_config.prm_model,
                    judge_api_key=self.oc_config.prm_api_key,
                    workspace_path=workspace_path,
                    extra_info=extra_info or {},
                )
            except TypeError as exc:
                if "extra_info" not in str(exc):
                    raise
                return await compute_episode_rewards_async(
                    trajectory, terminal_success, task_id,
                    task_prompt=task_prompt,
                    mode=self.oc_config.reward_mode,
                    vllm_base_url=vllm_url,
                    judge_model=self.oc_config.prm_model,
                    judge_api_key=self.oc_config.prm_api_key,
                    workspace_path=workspace_path,
                )
        except Exception as e:
            logger.error("Reward computation failed: %s", e)
            return []

    def _assign_rewards(
        self, response_ids: list[int], response_mask: list[int],
        per_turn_rewards: list[float], terminal_reward: float,
    ) -> list[float]:
        rewards = [0.0] * len(response_ids)
        if not response_ids:
            return rewards

        assignment = os.environ.get("PINCHBENCH_REWARD_ASSIGNMENT", "turn_broadcast").strip().lower()
        if assignment in {"turn_broadcast", "broadcast", "span"}:
            self._assign_rewards_to_turn_spans(rewards, response_mask, per_turn_rewards)
            if terminal_reward:
                self._assign_scalar_to_model_tokens(rewards, response_mask, terminal_reward)
            return rewards

        im_end_id = self.tokenizer.convert_tokens_to_ids("<|im_end|>")
        model_im_ends = [
            i for i, tid in enumerate(response_ids)
            if tid == im_end_id and i < len(response_mask) and response_mask[i] == 1
        ]

        n = min(len(per_turn_rewards), len(model_im_ends))
        for k in range(n):
            rewards[model_im_ends[k]] = per_turn_rewards[k]

        if n < len(per_turn_rewards):
            leftover = sum(per_turn_rewards[n:])
            rewards[-1] += leftover

        rewards[-1] += terminal_reward
        return rewards

    @staticmethod
    def _model_token_spans(response_mask: list[int]) -> list[tuple[int, int]]:
        spans: list[tuple[int, int]] = []
        start: int | None = None
        for idx, mask in enumerate(response_mask):
            if mask == 1 and start is None:
                start = idx
            elif mask != 1 and start is not None:
                spans.append((start, idx))
                start = None
        if start is not None:
            spans.append((start, len(response_mask)))
        return spans

    def _assign_rewards_to_turn_spans(
        self,
        rewards: list[float],
        response_mask: list[int],
        per_turn_rewards: list[float],
    ) -> None:
        spans = [
            (max(0, start), min(len(rewards), end))
            for start, end in self._model_token_spans(response_mask[: len(rewards)])
            if start < len(rewards) and end > start
        ]
        if not spans:
            if rewards and per_turn_rewards:
                rewards[-1] += float(sum(per_turn_rewards))
            return

        n = min(len(per_turn_rewards), len(spans))
        for idx in range(n):
            value = float(per_turn_rewards[idx])
            start, end = spans[idx]
            length = max(1, end - start)
            per_token = value / length
            for pos in range(start, end):
                rewards[pos] += per_token

        if n < len(per_turn_rewards):
            leftover = float(sum(per_turn_rewards[n:]))
            start, end = spans[-1]
            length = max(1, end - start)
            per_token = leftover / length
            for pos in range(start, end):
                rewards[pos] += per_token

    def _assign_scalar_to_model_tokens(
        self,
        rewards: list[float],
        response_mask: list[int],
        scalar_reward: float,
    ) -> None:
        positions = [idx for idx, mask in enumerate(response_mask) if mask == 1]
        if not positions:
            if rewards:
                rewards[-1] += scalar_reward
            return
        per_token = scalar_reward / len(positions)
        for pos in positions:
            rewards[pos] += per_token

    # ── Utilities ──

    def _parse_tool_calls(self, text: str) -> Optional[list[dict]]:
        import re
        tool_calls = []
        def add_call(name: str, arguments: dict) -> None:
            tool_calls.append({
                "id": f"call_{uuid.uuid4().hex[:8]}",
                "type": "function",
                "function": {
                    "name": name,
                    "arguments": json.dumps(arguments, ensure_ascii=False),
                },
            })

        for match in re.finditer(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.DOTALL):
            try:
                tc = json.loads(match.group(1))
                add_call(tc.get("name", ""), tc.get("arguments", {}))
            except json.JSONDecodeError:
                pass
        # Qwen3.5 XML tool-call format used by vLLM qwen3_xml parser:
        # <tool_call><function=read><parameter=path>file.txt</parameter></function></tool_call>
        for block in re.finditer(r'<tool_call>\s*(.*?)\s*</tool_call>', text, re.DOTALL):
            body = block.group(1)
            if body.lstrip().startswith("{"):
                continue
            fn = re.search(r'<function=([A-Za-z_][\w.-]*)>\s*(.*?)\s*</function>', body, re.DOTALL)
            if not fn:
                continue
            args: dict[str, str] = {}
            for param in re.finditer(r'<parameter=([A-Za-z_][\w.-]*)>\s*(.*?)\s*</parameter>', fn.group(2), re.DOTALL):
                args[param.group(1)] = param.group(2).strip()
            add_call(fn.group(1), args)
        return tool_calls if tool_calls else None

    def _prepare_messages(self, messages: list[dict], tools: list[dict] | None) -> list[dict]:
        """Prepare OpenClaw messages for Qwen's apply_chat_template.

        1. Flatten content-parts format (list of dicts) to plain strings,
           since Qwen's template silently drops list-format content.
        2. Inject <tool_call> format instructions so the model knows to output
           parseable XML tags when calling tools.
        """
        out = []
        for msg in messages:
            msg = dict(msg)
            content = msg.get("content")
            if isinstance(content, list):
                parts = []
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        parts.append(part.get("text", ""))
                msg["content"] = "\n".join(parts)
            out.append(msg)

        # Default OFF: PinchBench benchmark uses OpenClaw → vLLM without this shim.
        # Opt-in for old experiments: PINCHBENCH_RL_INJECT_TOOL_FORMAT_SUFFIX=1
        if tools and os.environ.get("PINCHBENCH_RL_INJECT_TOOL_FORMAT_SUFFIX", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        ):
            TOOL_FORMAT_SUFFIX = (
                "\n\n# Output Format for Tool Calls\n"
                "When you need to call a tool, output the call inside <tool_call></tool_call> XML tags:\n"
                "<tool_call>\n"
                '{"name": "<function-name>", "arguments": {<args-json-object>}}\n'
                "</tool_call>\n"
                "You may call multiple tools by using multiple <tool_call> blocks."
            )
            for msg in out:
                if msg.get("role") == "system":
                    content = msg.get("content", "")
                    if isinstance(content, str) and "<tool_call>" not in content:
                        msg["content"] = content + TOOL_FORMAT_SUFFIX
                    break

        return out

    def _strip_tool_tags(self, text: str) -> str:
        """Strip <tool_call>...</tool_call> and <think>...</think> from text for OpenClaw content field."""
        import re
        text = re.sub(r'<tool_call>.*?</tool_call>', '', text, flags=re.DOTALL)
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        return text.strip()

    def _get_public_ip(self) -> str:
        """Get public IP (kept for fallback/debugging)."""
        try:
            import urllib.request
            return urllib.request.urlopen("https://ifconfig.me", timeout=5).read().decode().strip()
        except Exception:
            return "127.0.0.1"
