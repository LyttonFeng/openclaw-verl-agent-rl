"""DSv4-Pro judges the quality of an agent's swarm policy.

Given (extracted plan text, transcript summary, terminal score), the judge
rates four dimensions of the policy and its execution:

  1. decomposition_quality  — is the plan well-scoped for the task?
  2. verification           — does the plan include a check step?
  3. adherence              — did execution follow the stated plan?
  4. efficiency             — no over-decomposition for simple tasks?

Returns a swarm_policy_score in [0, 1] plus per-dimension breakdown.

This is independent of terminal_score so it adds a separate signal that has
variance even when terminal scores collapse to similar values (the failure
mode that killed MR2 round_02).
"""

from __future__ import annotations

import json
import logging
import os
import re
import time
import urllib.request
from dataclasses import dataclass
from typing import Any, Optional

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "deepseek-v4-pro"
DEFAULT_BASE_URL = "https://api.deepseek.com/v1"
DEFAULT_TIMEOUT = 120.0
MAX_RETRIES = 3
RETRY_BACKOFF = 1.5


@dataclass
class SwarmJudgeResult:
    score: float                    # composite swarm_policy_score in [0, 1]
    breakdown: dict[str, float]     # per-dimension
    notes: str
    raw_response: str
    error: Optional[str] = None


_SYSTEM = (
    "You are a rigorous judge of an AI agent's *team policy* on a long-context "
    "task. You evaluate the policy plan separately from whether the task was "
    "ultimately scored high. Always return valid JSON only."
)


_USER_TMPL = """You will see one agent rollout on a task. Your job is to rate the QUALITY OF ITS OBSERVED SWARM/TEAM POLICY (how it actually structured its work, judged from execution traces), independent of whether the final answer happened to be correct.

# Task description
{task_brief}

# Intended swarm style (the prompt template the agent was given)
template_id: {template_id}
{template_desc}

# The agent's emitted plan (if it wrote one)
{plan_text}

# Observed execution (this is the primary signal — judge based on what the agent actually did)
- terminal score: {terminal_score:.3f}
- total tool calls by Lead: {n_tool_calls}
- direct read calls by Lead: {n_reads}
- read offset range: {coverage_hint}
- distinct files written: {n_files_written}
- evidence of re-read after draft (any read with offset overlapping earlier reads): {has_reread}
- evidence of intermediate notes / drafts (writes to non-final files): {has_intermediate_files}
- **sub-agent dispatches** (delegations to a frozen Qwen3-4B helper): {n_subagent_calls}
- sub-agent task descriptions: {subagent_tasks}

DECOMPOSITION SIGNAL: A Lead that dispatches sub-agents IS decomposing the work — the
sub-agents do the reading and return compact summaries. So "0 reads by Lead but
N sub-agent dispatches" = STRONG decomposition (Lead delegated everything).
"0 reads by Lead AND 0 sub-agent dispatches" = no work attempted.

# Rate the OBSERVED policy on four dimensions, each 0.0 to 1.0:

1. **decomposition_quality** — Did the agent's actual execution show meaningful decomposition for this task? Multiple distinct reads covering different parts = good decomposition. Single-pass when task is multi-aspect = poor.

2. **verification** — Did the agent re-read the source after drafting? Did it cross-check between writes? Score 1.0 if explicit re-read pattern is visible. 0.5 if partial. 0.0 if no verification at all.

3. **adherence** — Did the observed behavior match the intended swarm style (template_id)? Score high if execution aligns with the template's expected pattern. Score low if agent ignored the template style.

4. **efficiency** — Was effort spent on the task (not wasted)? Score low ONLY if there's clear duplicate / redundant work (e.g., reading the same section twice for no reason, dispatching identical sub-tasks). Do NOT penalize thorough work or multiple sub-agents that each cover distinct aspects — that's good decomposition, not waste.

Return ONLY this JSON object (no markdown, no prose):

{{
  "decomposition_quality": 0.0,
  "verification": 0.0,
  "adherence": 0.0,
  "efficiency": 0.0,
  "swarm_policy_score": 0.0,
  "notes": "1-2 sentences about the observed policy"
}}

"swarm_policy_score" must equal the unweighted mean of the four dimension scores.
"""


def _extract_json(text: str) -> Optional[dict[str, Any]]:
    text = text.strip()
    if not text:
        return None
    # direct parse
    try:
        d = json.loads(text)
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        pass
    # code fence
    m = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    if m:
        try:
            d = json.loads(m.group(1))
            return d if isinstance(d, dict) else None
        except json.JSONDecodeError:
            pass
    # balanced brace
    s = text.find("{")
    e = text.rfind("}")
    if s >= 0 and e > s:
        try:
            d = json.loads(text[s : e + 1])
            return d if isinstance(d, dict) else None
        except json.JSONDecodeError:
            pass
    return None


def _call_api(prompt: str, *, model: str, base_url: str, api_key: str, timeout: float) -> tuple[str, Optional[str]]:
    """Returns (text, error_str_or_None)."""
    body = json.dumps(
        {
            "model": model,
            "messages": [
                {"role": "system", "content": _SYSTEM},
                {"role": "user", "content": prompt},
            ],
            "temperature": 0.0,
            "max_tokens": 4096,  # DSv4-Pro's reasoning can eat 2-3k tokens; need headroom
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions",
        data=body,
        headers={"Content-Type": "application/json", "Authorization": f"Bearer {api_key}"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        msg = data.get("choices", [{}])[0].get("message", {})
        # DSv4-Pro frequently returns its analysis in `reasoning_content` and
        # leaves `content` empty when max_tokens is hit. Fall back to
        # reasoning_content so the judge still produces output.
        text = msg.get("content") or msg.get("reasoning_content") or ""
        return text, None
    except Exception as e:
        return "", f"{type(e).__name__}: {e}"


def judge_swarm_policy(
    *,
    task_brief: str,
    plan_text: str,
    terminal_score: float,
    template_id: str = "unknown",
    template_desc: str = "",
    n_tool_calls: int = 0,
    n_reads: int = 0,
    coverage_hint: str = "",
    n_files_written: int = 0,
    has_reread: bool = False,
    has_intermediate_files: bool = False,
    n_subagent_calls: int = 0,
    subagent_tasks: list[str] | None = None,
    model: str = DEFAULT_MODEL,
    base_url: str = DEFAULT_BASE_URL,
    api_key: Optional[str] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> SwarmJudgeResult:
    """Judge one rollout's swarm policy from OBSERVED behavior.

    plan_text is optional (used as extra context if agent emitted one). The
    judge primarily looks at execution traces: tool call counts, read patterns,
    file writes, and whether observed behavior matches the intended template.
    """
    api_key = api_key or os.environ.get("DEEPSEEK_API_KEY") or ""
    if not api_key:
        return SwarmJudgeResult(
            score=0.0,
            breakdown={},
            notes="",
            raw_response="",
            error="no DEEPSEEK_API_KEY",
        )

    sub_tasks_str = "; ".join(subagent_tasks or []) if subagent_tasks else "(none)"
    prompt = _USER_TMPL.format(
        task_brief=task_brief[:800],
        template_id=template_id,
        template_desc=template_desc[:200],
        plan_text=(plan_text[:1500] if plan_text else "(agent did not emit an explicit <plan> block — judge from observed behavior only)"),
        terminal_score=terminal_score,
        n_tool_calls=n_tool_calls,
        n_reads=n_reads,
        coverage_hint=coverage_hint or "n/a",
        n_files_written=n_files_written,
        has_reread="yes" if has_reread else "no",
        has_intermediate_files="yes" if has_intermediate_files else "no",
        n_subagent_calls=n_subagent_calls,
        subagent_tasks=sub_tasks_str,
    )

    last_err = ""
    for attempt in range(MAX_RETRIES):
        text, err = _call_api(prompt, model=model, base_url=base_url, api_key=api_key, timeout=timeout)
        if err:
            last_err = err
            logger.warning("swarm judge attempt %d/%d API error: %s", attempt + 1, MAX_RETRIES, err)
        else:
            parsed = _extract_json(text)
            if parsed and "swarm_policy_score" in parsed:
                bk = {
                    k: float(parsed.get(k, 0.0))
                    for k in ("decomposition_quality", "verification", "adherence", "efficiency")
                    if k in parsed
                }
                score = float(parsed["swarm_policy_score"])
                # clamp
                score = max(0.0, min(1.0, score))
                return SwarmJudgeResult(
                    score=score,
                    breakdown=bk,
                    notes=str(parsed.get("notes", "")),
                    raw_response=text,
                    error=None,
                )
            last_err = f"parse_failed (text head='{text[:120]}')"
            logger.warning("swarm judge attempt %d/%d parse failed", attempt + 1, MAX_RETRIES)
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_BACKOFF ** attempt)

    return SwarmJudgeResult(
        score=0.0,
        breakdown={},
        notes="",
        raw_response=text if "text" in dir() else "",
        error=f"judge_exhausted_retries: {last_err}",
    )


# ---------------------------------------------------------------------------


def composite_reward(
    terminal_score: float,
    swarm_policy_score: float,
    gamma: float = 0.4,
) -> float:
    """Linear composite. Keep terminal as the primary signal."""
    return (1.0 - gamma) * terminal_score + gamma * swarm_policy_score


if __name__ == "__main__":
    # smoke test (requires DEEPSEEK_API_KEY)
    logging.basicConfig(level=logging.INFO)
    res = judge_swarm_policy(
        task_brief="Extract every vote / motion from a long city council transcript.",
        plan_text="strategy: decompose-2-parallel\nsubtasks: [scan votes L0-3000, scan votes L3000-end]",
        terminal_score=0.5,
        n_tool_calls=15,
        n_reads=12,
        coverage_hint="L0-5000",
    )
    print(json.dumps({"score": res.score, "breakdown": res.breakdown, "notes": res.notes, "error": res.error}, indent=2))
