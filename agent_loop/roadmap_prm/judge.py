"""
Per-turn judge: DSv4-flash scores a single agent turn against a roadmap.

Returns one of {-1, 0, +1} along with milestone hit / pitfall name / reason.
"""

from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any

from .schema import Roadmap
from .trajectory import Turn, render_history


JUDGE_SYSTEM = """You are a strict process supervisor evaluating whether an AI agent's CURRENT TURN advances toward completing a task. You do NOT evaluate the final result — only this single turn's contribution.

You have:
- A roadmap describing milestones and pitfalls for the task.
- The full task prompt the agent received.
- Compressed history of all previous turns.
- The current turn (thinking + tool_calls + tool_results + text).

Score this turn:
  +1  Made CLEAR meaningful progress on a milestone, AND no pitfall is triggered.
        Examples: wrote the expected output file (after reading source); extracted
        NEW concrete data visible in this turn (specific names, dates, amounts);
        correctly noted an issue and changed approach. NOT just reading a file —
        reading the input transcript is baseline behavior, see BASELINE section.
   0  Default for routine, baseline, or ambiguous turns. Examples: bare read of
        the input file; re-reading a file already read; brief acknowledgement;
        listing files; minor formatting; thinking-only turn that is not excessive.
        When uncertain between +1 and 0, ALWAYS choose 0.
  -1  Regressed. Any pitfall in the roadmap is triggered, OR the turn hallucinates,
        loops, wastes budget, or moves away from the goal.

EVALUATION ORDER (do this every turn, in order):
  STEP 1: Walk through EVERY pitfall in the roadmap. For each one, check whether
          its conditions are satisfied by THIS turn (paying attention to "LAST
          turn", "prior turn must have", "no X has occurred yet" clauses — use
          the FINAL TURN flag and trajectory history to verify these). If ANY
          pitfall fires, score -1 immediately, name the pitfall, set milestone
          to "none". Pitfalls ALWAYS win — even if a milestone also fires.
  STEP 2: Only if NO pitfall fires, check milestones — but apply the
          BASELINE-BEHAVIOR filter below before awarding +1.
  STEP 3: Otherwise score 0.

BASELINE BEHAVIORS (default to score=0 even if a milestone superficially matches):

These are tasks where reading the source file and producing some output are
the minimum expected behavior. Routine actions every agent performs are NOT
worth +1 — they are the floor, not progress.

  - Reading the input transcript file is BASELINE. An M1_read_* / M1_first_read /
    M1_transcript_parsed milestone does NOT fire +1 just because a read happened.
    Only score +1 if this turn does something MORE than read: extract concrete
    structured data, parse content into a meaningful form (e.g. name a specific
    attendee, date, decision visible in the turn). A bare read with no extraction
    or downstream use scores 0, not +1.
  - Listing files in the workspace, brief acknowledgements, a single tool_call
    with no extraction or transformation: score 0.
  - Re-reading a file already read in a prior turn: score 0.
  - Thinking that does not produce a tool_call or substantive text: score 0
    (only -1 if excessive per the roadmap rule).

Milestone hits that DO reliably warrant +1:
  - Writing the expected output file (M5_write_* / M6_* / M7_file_written family),
    on the FIRST turn the write happens, AND a prior successful read exists.
  - Extracting NEW concrete data the task requires that is visibly present in
    the current turn's text or tool_result (e.g. listing specific attendee
    names from the transcript, identifying dates / dollar amounts / decisions).
  - Correctly noticing an error and changing approach.

When uncertain between +1 and 0, ALWAYS pick 0. Sparse high-confidence
positive signal is strongly preferred over dense weak signal — most turns
should score 0, with +1 reserved for clear progress events and -1 reserved
for clear pitfall hits.

CRITICAL RULES:
1. **Pitfall sweep is mandatory.** Do not skip STEP 1. The temptation is to see
   an obvious milestone hit (like "first read of source") and stop there. But
   in 1-turn trajectories, that same turn often triggers a "no output written
   by end of trajectory" pitfall — both can be true, pitfall wins.
2. **Check PRECONDITIONS literally.** If a milestone or pitfall description
   contains "AND a prior turn has done X" or "AND this is the LAST turn", verify
   by looking at the FINAL TURN flag and trajectory history. If the
   precondition is not met, that rule does NOT fire.
3. **First-time only for milestones.** A milestone fires +1 only on the FIRST
   turn that satisfies it. Subsequent re-satisfactions score 0.
4. **Long post-write text counts as a pitfall.** If the main output was already
   produced via a tool call in a prior turn, AND this turn produces another
   long block of assistant text (>500 chars) that paraphrases / restates /
   re-derives the output, score -1 and name the relevant pitfall.
5. **Thinking alone.** Thinking paired with a tool call is fine. Thinking ALONE
   (no tool call, no substantive text) only scores -1 if the thinking is
   excessive (>2500-3000 chars per the roadmap).
6. **When unsure, choose 0.** Only +1 for clear first-time progress with NO
   pitfall match. Only -1 for a clear pitfall match.

WORKED EXAMPLES (illustrating STEP 1 precedence):

EXAMPLE A — single-turn aborted trajectory:
  Trajectory has exactly 1 turn (FINAL TURN: YES).
  Turn 0: tool_call read('input.md') with successful tool_result; no other action.
  Roadmap defines M1_read = first read of input, P1_no_output_written = LAST
  turn AND no write has occurred.
  Both M1 conditions and P1 conditions are satisfied. STEP 1 says pitfalls
  win. The correct answer is:
    {"score": -1, "milestone": "none", "pitfall": "P1_no_output_written",
     "reason": "Final turn with no write tool_call anywhere in trajectory."}
  WRONG (do not do this):
    {"score": 1, "milestone": "M1_read", ...}  -- skipped STEP 1 sweep.

EXAMPLE B — long write after a successful read:
  Trajectory has 3 turns. Turn 0 read input (39000 chars, is_error: false).
  Turn 1 (this turn) writes output.md with 11000 chars of structured tables.
  Roadmap has P2_write_without_reading defined as: write occurs AND no prior
  successful read tool_call exists.
  A prior read tool_call with is_error: false on the input file DOES exist
  (turn 0). So P2 does NOT fire, regardless of how polished the written
  content looks. Check P2 by literal tool history only, never by content
  style. The correct answer is +1 (write milestone fires) or 0 — never -1
  with P2.

EXAMPLE C — short trailing text after a write:
  Turn 1 wrote output.md. This turn produces 87 chars of assistant text and
  no tool call. P3_post_write_rambling requires text > 500 chars. 87 < 500,
  so P3 does NOT fire. Correct: 0 (neutral closing).

Output STRICT JSON (no extra text, no markdown):
{"score": <-1|0|1>, "milestone": "<milestone_id or 'none'>", "pitfall": "<pitfall_id or 'none'>", "reason": "<one short sentence>"}
"""


@dataclass
class JudgeResult:
    score: int  # -1, 0, +1
    milestone: str  # milestone id or "none"
    pitfall: str  # pitfall id or "none"
    reason: str
    raw: str = ""  # raw judge output for debugging


def _get_judge_config() -> tuple[str, str, str]:
    """Returns (model, base_url, api_key) for DSv4-flash."""
    api_key = os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY (or OPENAI_API_KEY) must be set")
    base_url = os.environ.get("ROADMAP_JUDGE_BASE_URL", "https://api.deepseek.com/v1")
    model = os.environ.get("ROADMAP_JUDGE_MODEL", "deepseek-v4-flash")
    return model, base_url, api_key


def _call_chat_completion(messages: list[dict], temperature: float = 0.0, max_tokens: int = 200) -> str:
    model, base_url, api_key = _get_judge_config()
    payload = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": False,
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base_url}/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        },
        method="POST",
    )
    last_err: Exception | None = None
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                body = resp.read().decode("utf-8")
            obj = json.loads(body)
            return obj["choices"][0]["message"]["content"]
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as e:
            last_err = e
            time.sleep(2 ** attempt)
    raise RuntimeError(f"Judge API call failed after retries: {last_err}")


_JSON_RE = re.compile(r"\{[^{}]*\}")


def _parse_judge_output(raw: str) -> JudgeResult:
    raw_stripped = raw.strip()
    # Strip markdown code fence if present
    if raw_stripped.startswith("```"):
        raw_stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_stripped, flags=re.MULTILINE)
    # Try direct parse
    try:
        obj = json.loads(raw_stripped)
    except json.JSONDecodeError:
        m = _JSON_RE.search(raw_stripped)
        if not m:
            return JudgeResult(score=0, milestone="none", pitfall="none",
                               reason=f"PARSE_FAIL: {raw[:120]}", raw=raw)
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return JudgeResult(score=0, milestone="none", pitfall="none",
                               reason=f"PARSE_FAIL: {raw[:120]}", raw=raw)

    sc = obj.get("score", 0)
    if isinstance(sc, str):
        sc = int(sc.replace("+", ""))
    sc = max(-1, min(1, int(sc)))
    return JudgeResult(
        score=sc,
        milestone=str(obj.get("milestone", "none")),
        pitfall=str(obj.get("pitfall", "none")),
        reason=str(obj.get("reason", "")),
        raw=raw,
    )


def judge_turn(
    roadmap: Roadmap,
    task_prompt: str,
    history_turns: list[Turn],
    current_turn_idx: int,
) -> JudgeResult:
    """
    Score a single turn given the roadmap, task prompt, and full trajectory.
    history_turns[current_turn_idx] is the turn being scored.
    """
    if current_turn_idx >= len(history_turns):
        raise IndexError(f"current_turn_idx {current_turn_idx} >= {len(history_turns)}")

    history_text = render_history(history_turns, current_turn_idx)
    current_text = history_turns[current_turn_idx].to_judge_text()
    total_turns = len(history_turns)
    is_final = current_turn_idx == total_turns - 1
    position_line = (
        f"Position: turn {current_turn_idx} of {total_turns} total. "
        f"FINAL TURN: {'YES' if is_final else 'NO'}."
    )

    user_msg = f"""# Roadmap

{roadmap.to_judge_text()}

# Task Prompt (what the agent was asked to do)

{task_prompt}

# Trajectory History (previous turns)

{history_text}

# CURRENT TURN to score

{position_line}

{current_text}

# Now score this turn. Output strict JSON only.
# Reminder: pitfalls with a "LAST turn" condition only fire when FINAL TURN: YES above.
# Reminder: milestones with a "final turn" condition similarly require FINAL TURN: YES."""

    raw = _call_chat_completion(
        messages=[
            {"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": user_msg},
        ],
        temperature=0.0,
        max_tokens=200,
    )
    return _parse_judge_output(raw)


def judge_trajectory(
    roadmap: Roadmap,
    task_prompt: str,
    turns: list[Turn],
    max_turns: int | None = None,
) -> list[JudgeResult]:
    """Score every turn in a trajectory."""
    n = len(turns) if max_turns is None else min(max_turns, len(turns))
    results = []
    for i in range(n):
        results.append(judge_turn(roadmap, task_prompt, turns, i))
    return results


# ─── Terminal-completion gate (design pivot 2026-05-06) ─────────────────────────────
# Goal: only run process-supervision (per-turn PRM) on rollouts that FAILED
# the terminal goal — "only help the lost trajectories". Judge decides this
# binary directly; no numerical threshold in code.

TERMINAL_COMPLETION_SYSTEM = """You judge whether an AI agent FAILED to accomplish the task's terminal goal.

Your job is to identify "lost trajectories" — the only ones that get process
supervision in training. Successful and partially-successful work should pass
through untouched.

Default verdict: mostly_done = TRUE. Only flip to false when there is clear
evidence the agent got lost — for example, the deliverable is missing, empty,
or completely off-topic; the agent gave up or kept apologizing without
producing output; or the work shows no real attempt at the main asks.
Substantive output with minor flaws or imperfections still counts as
mostly_done = true.

A separate terminal grader has already scored this trajectory; you'll see that
score below. Treat it as one signal among others — rely primarily on your own
read of the deliverable, but let the score sanity-check your gut.

When uncertain → mostly_done = true. Bias toward letting trajectories pass.

Output strict JSON:
{"mostly_done": <true|false>, "reason": "<one short sentence>"}"""


def judge_terminal_completion(
    task_prompt: str,
    final_turn_text: str,
    terminal_score: float | None = None,
) -> tuple[bool, str]:
    """Returns (mostly_done, reason). On parse/API failure, returns (True, error_msg)
    so the caller skips per-turn judging — safe default biased toward "let it pass"
    rather than risk amplifying success on a successful trajectory."""
    score_line = (
        f"\n\nTerminal grader score: {terminal_score:.3f} / 1.0"
        if terminal_score is not None else ""
    )
    user_msg = f"""# Task Prompt

{task_prompt}

# Agent's Final Turn / Final Output

{final_turn_text}{score_line}

# Now judge: did the agent mostly accomplish the terminal goal? Output strict JSON only."""
    try:
        raw = _call_chat_completion(
            messages=[
                {"role": "system", "content": TERMINAL_COMPLETION_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.0,
            max_tokens=120,
        )
    except Exception as e:
        # Safe default on API failure: pass-through (mostly_done=true). We'd rather
        # lose PRM signal on a lost trajectory than risk amplifying a successful one.
        return True, f"API_ERROR: {type(e).__name__}: {str(e)[:100]}"

    raw_stripped = raw.strip()
    if raw_stripped.startswith("```"):
        raw_stripped = re.sub(r"^```(?:json)?\s*|\s*```$", "", raw_stripped, flags=re.MULTILINE)
    try:
        obj = json.loads(raw_stripped)
    except json.JSONDecodeError:
        m = _JSON_RE.search(raw_stripped)
        if not m:
            return True, f"PARSE_FAIL: {raw[:120]}"
        try:
            obj = json.loads(m.group(0))
        except json.JSONDecodeError:
            return True, f"PARSE_FAIL: {raw[:120]}"
    # Default to mostly_done=true if key missing — same bias as elsewhere.
    return bool(obj.get("mostly_done", True)), str(obj.get("reason", ""))
