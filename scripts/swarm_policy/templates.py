"""Prompt templates that bias a Lead agent toward different swarm/team
policies, where sub-agent work is delegated via `exec subagent.sh`.

The Lead is the only trainable model. Each sub-agent invocation spawns a
frozen Qwen3-4B base instance in a separate OpenClaw session, reads a slice
of the source, and returns ONLY its final summary text. The Lead's context
sees the summary as exec tool output.

Four templates span the swarm-decision spectrum so K=4 rollouts per task
give the GRPO group meaningful variance:

  M1_lone:     no sub-agent (control / lower bound — Lead does everything itself)
  M2_split:    pre-decompose into 2 slices, dispatch one sub-agent per slice
  M3_review:   sub-agent does primary extraction, Lead verifies by re-reading
  M4_iter:     dispatch sub-agents, inspect summaries, dispatch more if gaps remain
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


SUBAGENT_TOOL_DOC = """

---
**SUB-AGENT TOOL AVAILABLE**

You have a sub-agent helper accessible via the `exec` tool. It is a separate
Qwen3-4B agent with its own OpenClaw session. You can dispatch it focused
sub-tasks to keep your own context lean.

Usage:
    exec subagent.sh "<sub-task description>" <source_file> [<offset>] [<length>]

The sub-agent will read the source (full file or specified slice), do the
sub-task with its own read/grep/exec, and return ONLY its final summary text
as the exec stdout. The summary is the only thing you will see — the
sub-agent's internal reasoning and tool calls are hidden.

Best practice:
- Use sub-agent for chunked / specialized investigation. The sub will return
  detailed structured results (verbatim quotes preserved); your context is
  65k tokens, so don't worry about size.
- Give precise sub-task descriptions and (when possible) offset/length to
  bound the sub's work.
- If a result is incomplete or wrong, dispatch the sub-agent again with a
  refined task description.
"""

QWEN_SIMPLE_TOOL_DOC = """

---
You have one helper agent.

To ask the helper to read the meeting transcript and return notes, run:

    exec subagent.sh "<focused task>" <source_file>

The helper returns notes only. It does not write files.
You must write the final required output file yourself.

Use one of these helper requests when it matches the task:
- Stakeholder analysis: Extract all stakeholders, organizations, roles, interests, concerns, positions, and direct evidence.
- Council votes: Extract every agenda item, motion, vote result, dissent, abstention, recusal, and final outcome.
- Speaker summary: Extract each speaker, affiliation, topic, position, request, and supporting quote.
- Tech action items: Extract every action item, owner, deadline, decision, dependency, risk, and status change.
- Sentiment analysis: Extract each speaker or group, sentiment polarity, sentiment target, reason, and supporting quote.
"""


_QWEN_SIMPLE_SUFFIX = QWEN_SIMPLE_TOOL_DOC + """

Follow this workflow exactly:

1. Find the exact output filename required by the user task.
2. Call the helper once with the focused request that matches the task type.
3. After the helper returns, read the transcript only to verify missing or uncertain facts.
4. Write the required output file yourself using the exact filename from the task.
5. Check that the output file exists and is non-empty.

Hard rules:
- Helper calls: maximum 2.
- Transcript read calls after helper returns: maximum 4.
- Do not use offset or length arguments.
- Do not write a separate plan block.
- Do not answer only in chat.
- If some facts remain uncertain, write the best supported answer instead of continuing to read.
- The task is not complete until the required file is written.
"""


@dataclass(frozen=True)
class SwarmTemplate:
    template_id: str
    description: str
    expected_features: list[str]
    prompt_suffix: str


_SKILL_FULL_SUFFIX = SUBAGENT_TOOL_DOC + """

**Strategy for this task** — do these actions in order:

1. NOW: call `exec subagent.sh "<one narrow extraction task>" <source_file>` with a specific extraction goal. Example:
   `subagent.sh "Extract every motion and vote outcome with verbatim dissenters" transcript.md`

2. Read the sub-agent's structured result. If gaps remain, dispatch ONE more sub-agent for the missing piece.

3. Spot-check ONE claim by reading the source directly.

4. Write the final deliverable file (the task tells you the exact filename).

Do not think long before step 1 — dispatch your first sub-agent immediately.
"""


_RELAY_SUFFIX = SUBAGENT_TOOL_DOC + """

**Strategy for this task** — you are a dispatcher, NOT an analyst. Do exactly these 2 steps:

1. NOW: call `exec subagent.sh "<paste the FULL task description here verbatim>" <source_file>`.
   The sub-agent will read the source, do all analysis, and return the complete deliverable content.

2. Take the sub-agent's output and write it DIRECTLY to the deliverable file (the task tells you the exact filename). Do NOT edit, summarize, or re-analyze — just write it as-is.

IMPORTANT: Do NOT read the source file yourself. Do NOT think about the task. Just dispatch and relay.
"""

_EVOLVED_V1_SUFFIX = SUBAGENT_TOOL_DOC + """

exec subagent.sh "<full task>" <source_file>
Write its summary to the output file. Do nothing else.
"""

_EVOLVED_V2_SUFFIX = SUBAGENT_TOOL_DOC + """

Find the transcript file and output file names from the user request.
Run: `exec subagent.sh "<full user request verbatim>" <transcript_file>`
Write its output to the specified output file. No other steps.
"""

_EVOLVED_V3_SUFFIX = SUBAGENT_TOOL_DOC + """

/no_think

You are a DISPATCHER. Do NOT think. Do NOT reason. Do exactly 2 actions, in order:

ACTION 1 — dispatch once:
  exec subagent.sh "<paste the full task request verbatim>" <transcript_file>

ACTION 2 — the instant the sub-agent returns, WRITE ITS OUTPUT to the output
file named in the task (use the write/create-file tool). Copy the sub-agent's
text in full. This write is mandatory — it is the only thing that scores.

Rules: no <think>. No re-reading the transcript. No extra exec calls. No
analysis. If you are unsure of the output filename, use the exact name the
task specifies. WRITE THE FILE — that is the whole job.
"""


_EVOLVED_V4_SUFFIX = SUBAGENT_TOOL_DOC + """

**You are a DISPATCHER. The sub-agent does the analysis; you only relay its output to a file.**

STEP 1 — dispatch ONCE:
    exec subagent.sh "<paste the full task request verbatim>" <transcript_file>

  This `exec` call BLOCKS and returns the sub-agent's complete result as its
  output. The sub may take 1-3 minutes — that is expected. Wait for it.

STEP 2 — if (and only if) the exec result is the line
    `Command still running (session <SID>, pid <PID>). Use process ...`
  then the sub is finishing in the background. This is NORMAL, NOT an error.
  Do EXACTLY this:
    - Read <SID> = the two-word name right after "session" and before the comma
      (e.g. for "session crisp-trail, pid 165409" the session id is `crisp-trail`).
      It is NOT the pid and NOT "<SID>-<PID>".
    - Call the `process` tool with action `poll` and that sessionId to WAIT for
      completion. If it is still running, poll again. Keep polling until you get
      the sub's finished output.
    - NEVER start a second `exec subagent.sh`. NEVER re-dispatch. One sub only.

STEP 3 — write the sub-agent's output verbatim to the deliverable file named in
  the task (use the write/create-file tool). Copy it in full, do not edit or
  re-analyze. This file write is the ONLY thing that scores — it is mandatory.

Do NOT read the transcript yourself. Do NOT do your own analysis. Dispatch,
wait (poll if needed), write the file. That is the whole job.
"""


_EVOLVED_V7_SUFFIX = SUBAGENT_TOOL_DOC + """

/no_think

**You are a DISPATCHER. Act immediately. Your FIRST output MUST be the `exec`
tool call below — do NOT write any plan, prose, or thinking before it. If your
first message contains text but no tool call, you have FAILED the task.**

STEP 1 — dispatch ONCE (this is your very first action):
    exec subagent.sh "<paste the full task request verbatim>" <transcript_file>

  The sub-agent reads the source, does ALL the analysis, and returns the complete
  deliverable as the exec output. It may take 1-3 minutes — that is expected.

STEP 2 — if the exec result is the line
    `Command still running (session <SID>, pid <PID>). Use process ...`
  the sub is finishing in the background. This is NORMAL, not an error. Do EXACTLY:
    - <SID> = the two-word name right after "session" and before the comma
      (e.g. "session crisp-trail, pid 165409" → sessionId is `crisp-trail`).
      It is NOT the pid and NOT "<SID>-<PID>".
    - Call the `process` tool with action `poll` and that sessionId to WAIT.
      Keep polling until you get the sub's finished output.
    - NEVER start a second exec. NEVER re-dispatch. NEVER stop here. One sub only.

STEP 3 — write the sub-agent's output VERBATIM to the deliverable file named in
  the task (use the write/create-file tool). Copy it in full; do not edit or
  re-analyze. **This file write is MANDATORY and is the ONLY thing that scores.
  You are not done until the file is written.**

Do NOT read the transcript yourself. Dispatch → wait/poll → write the file.
"""


SWARM_TEMPLATES: list[SwarmTemplate] = [
    SwarmTemplate(
        template_id="QWEN_SIMPLE",
        description="Qwen-friendly minimal helper prompt — one simple subagent call, no plan block, no offsets, Lead must write final file.",
        expected_features=["optional_subagent", "simple_tool_doc", "force_write"],
        prompt_suffix=_QWEN_SIMPLE_SUFFIX,
    ),
    SwarmTemplate(
        template_id="SINGLE_DIRECT",
        description="Single-agent direct baseline — no helper/sub-agent instructions.",
        expected_features=["single_agent", "no_subagent", "force_write"],
        prompt_suffix="""

Complete this task as a single agent.

Workflow:
1. Read the user task and identify the exact required output filename.
2. Read the meeting transcript yourself.
3. Extract the information required by the task.
4. Write the required output file using the exact filename from the task.
5. Check that the output file exists and is non-empty.

Rules:
- Do not call any helper or sub-agent.
- Do not answer only in chat.
- If some facts remain uncertain, write the best supported answer instead of continuing indefinitely.
- The task is not complete until the required file is written.
""",
    ),
    SwarmTemplate(
        template_id="EVOLVED_V7",
        description="v7 — /no_think + force-first-action (kills advisory no-op) + v4 poll-fallback + mandatory write (kills council dispatch-then-quit).",
        expected_features=["subagent_x1", "relay_only", "no_think", "force_action", "poll_fallback", "force_write"],
        prompt_suffix=_EVOLVED_V7_SUFFIX,
    ),
    SwarmTemplate(
        template_id="EVOLVED_V4",
        description="v4 — sync-exec relay + correct process-poll fallback (kills gov re-dispatch panic).",
        expected_features=["subagent_x1", "relay_only", "poll_fallback", "force_write"],
        prompt_suffix=_EVOLVED_V4_SUFFIX,
    ),
    SwarmTemplate(
        template_id="EVOLVED_V1",
        description="DSv4-Pro evolved v1 — ultra-short 2-line relay skill.",
        expected_features=["subagent_x1", "relay_only", "evolved"],
        prompt_suffix=_EVOLVED_V1_SUFFIX,
    ),
    SwarmTemplate(
        template_id="EVOLVED_V2",
        description="DSv4-Pro evolved v2 — names transcript+output file explicitly, relay.",
        expected_features=["subagent_x1", "relay_only", "evolved", "names_output_file"],
        prompt_suffix=_EVOLVED_V2_SUFFIX,
    ),
    SwarmTemplate(
        template_id="EVOLVED_V3",
        description="v3 — /no_think relay, forces immediate output-file write (kills overthinking-fatal).",
        expected_features=["subagent_x1", "relay_only", "no_think", "force_write"],
        prompt_suffix=_EVOLVED_V3_SUFFIX,
    ),
    SwarmTemplate(
        template_id="RELAY",
        description="Minimal relay — Lead dispatches full task to sub-agent, writes output verbatim.",
        expected_features=["subagent_x1", "relay_only", "no_lead_analysis"],
        prompt_suffix=_RELAY_SUFFIX,
    ),
    SwarmTemplate(
        template_id="SKILL_FULL",
        description="Stage 1 — full 5-step Swarm Meeting Analysis Skill protocol.",
        expected_features=["protocol_decide", "narrow_subtask", "verify", "synthesize"],
        prompt_suffix=_SKILL_FULL_SUFFIX,
    ),
    SwarmTemplate(
        template_id="M1_lone",
        description="No sub-agent — Lead does everything itself (control).",
        expected_features=["no_subagent", "lead_does_all_reads"],
        prompt_suffix=SUBAGENT_TOOL_DOC + """
**Execution style for this task**: solo. Do NOT use the sub-agent. Read the
source yourself and write the deliverable directly.

Before doing anything, output a `<plan>` block:

```
<plan>
strategy: solo-no-subagent
steps:
  - read source file
  - write final deliverable
</plan>
```

Then execute solo. No subagent.sh calls.
""",
    ),
    SwarmTemplate(
        template_id="M2_split",
        description="Pre-decompose into 2 slices, dispatch one sub-agent per slice, synthesize.",
        expected_features=["subagent_x2", "parallel_slice"],
        prompt_suffix=SUBAGENT_TOOL_DOC + """
**Execution style for this task**: parallel-decompose-2.

Before doing anything, output a `<plan>` block:

```
<plan>
strategy: parallel-decompose-2
slices:
  - id: slice_1
    focus: <one specific aspect>
    offset_length: <approximate offset:length>
  - id: slice_2
    focus: <another aspect>
    offset_length: <approximate offset:length>
synthesis: how the final deliverable combines both summaries
</plan>
```

Then dispatch TWO sub-agent calls (one per slice), wait for each summary,
then write the final deliverable using both summaries. Do NOT read the
full source yourself — let the sub-agents do the reading.
""",
    ),
    SwarmTemplate(
        template_id="M3_review",
        description="Sub-agent does primary work, Lead verifies by re-reading source.",
        expected_features=["subagent_x1", "lead_verifies"],
        prompt_suffix=SUBAGENT_TOOL_DOC + """
**Execution style for this task**: sub-extract-then-lead-verify.

Before doing anything, output a `<plan>` block:

```
<plan>
strategy: sub-extract-then-verify
phases:
  - dispatch: sub-agent extracts the bulk of the deliverable
  - verify: Lead spot-checks the summary against the source
  - finalize: Lead writes deliverable with verification fixes
</plan>
```

Then:
1. Dispatch ONE sub-agent with the full task on the whole source.
2. Read the source yourself to spot-check the sub's summary.
3. Write the final deliverable, fixing anything the sub missed.
""",
    ),
    SwarmTemplate(
        template_id="M4_iter",
        description="Iterative sub-agent dispatching — Lead re-dispatches if gaps remain.",
        expected_features=["subagent_loop", "redispatch"],
        prompt_suffix=SUBAGENT_TOOL_DOC + """
**Execution style for this task**: iterative-dispatch.

Before doing anything, output a `<plan>` block:

```
<plan>
strategy: iterative-dispatch
rounds:
  - round_1: dispatch initial sub-agent(s) for broad coverage
  - round_2: inspect summaries, dispatch more sub-agents to fill gaps if needed
  - synthesis: write final deliverable
</plan>
```

Then iterate:
1. Dispatch sub-agent(s) for an initial pass.
2. Read the summaries returned. If a key aspect is missing or unclear,
   dispatch another sub-agent with a refined task description targeting
   exactly that gap.
3. Stop when summaries are sufficient or you've done 4 dispatches.
4. Write the final deliverable.
""",
    ),
]


def get_template(template_id: str) -> SwarmTemplate:
    for t in SWARM_TEMPLATES:
        if t.template_id == template_id:
            return t
    raise KeyError(f"Unknown template id: {template_id}")


def apply_template(task_prompt: str, template_id: str) -> str:
    t = get_template(template_id)
    return task_prompt.rstrip() + t.prompt_suffix


_PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.DOTALL | re.IGNORECASE)


def extract_plan(response_or_transcript: str) -> Optional[str]:
    """Pull a `<plan>...</plan>` block. Skips template-example placeholders."""
    if not response_or_transcript:
        return None
    matches = _PLAN_RE.findall(response_or_transcript)
    if not matches:
        return None
    real = [m for m in matches
            if "<one specific" not in m
            and "<another aspect>" not in m
            and "<approximate offset:length>" not in m
            and "<plan>" not in m]
    chosen = (real[-1] if real else matches[-1]).strip()
    return chosen or None


def extract_plan_from_transcript(transcript: list) -> Optional[str]:
    """Extract plan ONLY from assistant message text."""
    if not transcript:
        return None
    assistant_texts: list[str] = []
    for event in transcript:
        if event.get("type") != "message":
            continue
        msg = event.get("message", {})
        if msg.get("role") != "assistant":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            assistant_texts.append(content)
        elif isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    assistant_texts.append(item.get("text", ""))
    blob = "\n\n".join(assistant_texts)
    return extract_plan(blob)


def all_template_ids() -> list[str]:
    return [t.template_id for t in SWARM_TEMPLATES]


def count_subagent_calls(transcript: list) -> int:
    """Count how many times the Lead invoked `exec subagent.sh`."""
    n = 0
    for event in transcript or []:
        if event.get("type") != "message":
            continue
        msg = event.get("message", {})
        if msg.get("role") != "assistant":
            continue
        for item in msg.get("content", []) or []:
            if not isinstance(item, dict) or item.get("type") != "toolCall":
                continue
            if str(item.get("name", "")).lower() != "exec":
                continue
            cmd = str((item.get("arguments") or {}).get("command", ""))
            if "subagent.sh" in cmd or "run_subagent.py" in cmd:
                n += 1
    return n


if __name__ == "__main__":
    print("Multi-agent templates:")
    for t in SWARM_TEMPLATES:
        print(f"  {t.template_id}: {t.description}")
