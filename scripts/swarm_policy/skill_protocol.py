"""Swarm Meeting Analysis Skill — Stage 1/2/3 prompt scaffolds.

Curriculum: scaffold-then-fade.
  Stage 1: full protocol (~400 tokens) — pure protocol instruction
  Stage 2: short reference (~30 tokens) — "Use the skill, then execute"
  Stage 3: empty — no skill reference; relies on weights only
"""

from __future__ import annotations
from dataclasses import dataclass


# ============================================================
# STAGE 1 — full skill protocol
# ============================================================
SKILL_FULL = """\

---
**SWARM MEETING ANALYSIS SKILL**

You have a sub-agent helper accessible via `exec subagent.sh "<sub-task>" <source_file> [<offset>] [<length>]`.
The sub-agent is a separate Qwen3-4B with its own context. It will read what
you ask and return a detailed structured result (verbatim quotes preserved).

Follow this 5-step protocol on EVERY meeting analysis task:

**1. Decide whether to delegate.**
   - If the task is short / single-aspect (e.g., write a single-paragraph summary
     of a 1-page transcript): do it solo. No sub-agent.
   - If the task is multi-aspect (multiple stakeholders, votes, action items),
     OR the source is long (>3000 lines): delegate to sub-agent(s).

**2. If delegating, ask ONE NARROW sub-task at a time.**
   - Each sub-call must have a single, specific extraction goal. Examples:
     ✓ "Extract every vote/motion with verbatim outcome and dissenters (L0-2500)"
     ✗ "Read the transcript and analyze the meeting"
   - Specify the source file and (when possible) an offset/length range.
   - You may issue multiple sub-calls (each for a distinct slice or aspect).

**3. Use the sub-agent's output as EVIDENCE, not as your answer.**
   - The sub-agent's response is raw evidence for you to verify and weave into
     the final deliverable.
   - Do NOT paste sub-agent output verbatim. Re-organize, cross-reference, and
     add structure.

**4. Verify your draft before finalizing.**
   - After drafting the final deliverable, re-open the source (read the
     transcript directly) to spot-check at least one claim per major section.
   - Fix any inaccuracies found.

**5. Synthesize the final deliverable yourself.**
   - Write the requested output file (e.g., stakeholder_analysis.md,
     votes_report.md, action_items.md) with the correct file name and the
     structure the task asks for.
   - You MUST write the output file before stopping. Do not stop with no
     deliverable.

CRITICAL: you MUST emit at least one tool call within your first 2 turns.
If you find yourself thinking too long without taking action, force a read or
exec call to ground the work.

---
"""


# ============================================================
# STAGE 2 — short reference
# ============================================================
SKILL_SHORT = """\

---
**Use the Swarm Meeting Analysis Skill** for this task:
delegate narrow sub-tasks to `exec subagent.sh`, verify the draft against the
source, then write the final deliverable yourself.
---
"""


# ============================================================
# STAGE 3 — empty (no skill scaffolding)
# ============================================================
SKILL_NONE = ""


@dataclass(frozen=True)
class SkillStage:
    name: str
    suffix: str
    description: str


STAGES = {
    "stage1_full":  SkillStage("stage1_full",  SKILL_FULL,  "Full 5-step skill protocol (~400 tok)"),
    "stage2_short": SkillStage("stage2_short", SKILL_SHORT, "Short skill reference (~30 tok)"),
    "stage3_none":  SkillStage("stage3_none",  SKILL_NONE,  "No skill reference (test internalization)"),
}


def apply_stage(task_prompt: str, stage: str) -> str:
    if stage not in STAGES:
        raise KeyError(f"Unknown stage: {stage}. Valid: {list(STAGES)}")
    return task_prompt.rstrip() + STAGES[stage].suffix


if __name__ == "__main__":
    for name, st in STAGES.items():
        print(f"=== {name} ({st.description}) ===")
        print(st.suffix or "(empty)")
        print()
