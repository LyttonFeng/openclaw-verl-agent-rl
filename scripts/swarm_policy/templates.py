"""Prompt templates that bias the agent toward different swarm/team policies.

Each template wraps the original task prompt with a meta-instruction that asks
the model to explicitly emit a `<plan>...</plan>` block describing its team
policy *before* execution. The plan content is parsed back out for the
swarm-policy judge.

The four templates intentionally span the swarm-decision spectrum so that K
rollouts with K different templates give the GRPO group real variance:

  T1_solo:        no decomposition (control / lower bound)
  T2_decompose2:  two parallel sub-investigations
  T3_verify:      sequential scan → draft → verify → finalize
  T4_evidence:    evidence-first (quote table before prose)

For each template we also encode the "expected swarm features" so the judge
can grade adherence later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class SwarmTemplate:
    template_id: str
    description: str
    expected_features: list[str]  # judge rubric anchors
    prompt_suffix: str             # appended after the task prompt


SWARM_TEMPLATES: list[SwarmTemplate] = [
    SwarmTemplate(
        template_id="T1_solo",
        description="Single-pass baseline. No decomposition.",
        expected_features=["single_pass", "no_subtasks"],
        prompt_suffix="""

---
**Execution style for this task**: single-pass.

Before doing anything, output a `<plan>` block with this minimal structure:

```
<plan>
strategy: single-pass
steps:
  - read transcript fully (one pass)
  - write final deliverable directly
</plan>
```

Then execute it. Do not split into sub-investigations or sub-workers.
""",
    ),
    SwarmTemplate(
        template_id="T2_decompose2",
        description="Two parallel sub-investigations + final synthesis.",
        expected_features=["decompose_2", "parallel_subtasks"],
        prompt_suffix="""

---
**Execution style for this task**: decompose into two parallel sub-investigations.

Before doing anything, output a `<plan>` block with this structure:

```
<plan>
strategy: decompose-2-parallel
subtasks:
  - id: sub_1
    focus: <one specific aspect of the task>
    transcript_range: <offset range or pages>
  - id: sub_2
    focus: <another specific aspect>
    transcript_range: <offset range or pages>
final_synthesis: how the final deliverable combines the two sub-results
</plan>
```

Then run the two sub-investigations sequentially in the same agent. Read each
range, collect compact notes, then write the final deliverable using both sets
of notes.
""",
    ),
    SwarmTemplate(
        template_id="T3_verify",
        description="Sequential scan → draft → verify → finalize.",
        expected_features=["sequential", "explicit_verify_step"],
        prompt_suffix="""

---
**Execution style for this task**: four-phase with explicit verification.

Before doing anything, output a `<plan>` block with this structure:

```
<plan>
strategy: scan-draft-verify-finalize
phases:
  - scan: identify candidate items in transcript (offsets covered)
  - draft: write initial deliverable
  - verify: re-read transcript to check coverage and accuracy of draft
  - finalize: produce final deliverable with verification fixes
</plan>
```

Then execute each phase in order. The `verify` phase MUST re-open the
transcript and cross-check, not just inspect the draft.
""",
    ),
    SwarmTemplate(
        template_id="T4_evidence",
        description="Evidence-first: build quote table before prose.",
        expected_features=["evidence_table_first", "structured_quotes"],
        prompt_suffix="""

---
**Execution style for this task**: evidence-first.

Before doing anything, output a `<plan>` block with this structure:

```
<plan>
strategy: evidence-first
steps:
  - scan transcript and collect a quote/evidence table (offset, text, label)
  - write the evidence table to an intermediate file
  - produce the final deliverable using the evidence table as the source of truth
acceptance_check: every claim in the final deliverable must trace to a row in the evidence table
</plan>
```

Then execute. The evidence table comes before any narrative prose.
""",
    ),
]


def get_template(template_id: str) -> SwarmTemplate:
    for t in SWARM_TEMPLATES:
        if t.template_id == template_id:
            return t
    raise KeyError(f"Unknown template id: {template_id}")


def apply_template(task_prompt: str, template_id: str) -> str:
    """Return the original task prompt with the template's meta-instruction appended."""
    t = get_template(template_id)
    return task_prompt.rstrip() + t.prompt_suffix


_PLAN_RE = re.compile(r"<plan>(.*?)</plan>", re.DOTALL | re.IGNORECASE)


def extract_plan(response_or_transcript: str) -> Optional[str]:
    """Pull a `<plan>...</plan>` block from response or transcript text.

    Returns the inner text (stripped) or None if not found. Uses the LAST match
    so a template-example block in the user prompt is NOT mistaken for the
    agent's actual emitted plan.
    """
    if not response_or_transcript:
        return None
    matches = _PLAN_RE.findall(response_or_transcript)
    if not matches:
        return None
    # Drop matches that look like the template example (contain placeholder text
    # like "<one specific aspect of the task>"). Prefer the last concrete block.
    real = [m for m in matches if "<one specific" not in m and "<another specific" not in m
            and "<offset range or pages>" not in m and "<plan>" not in m]
    chosen = (real[-1] if real else matches[-1]).strip()
    return chosen or None


def extract_plan_from_transcript(transcript: list) -> Optional[str]:
    """Extract plan ONLY from assistant message text content.

    Avoids matching the template example embedded in the user prompt. Returns
    the LAST plan emitted by assistant (in case the agent re-plans mid-run).
    """
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


if __name__ == "__main__":
    print("Templates:")
    for t in SWARM_TEMPLATES:
        print(f"  {t.template_id}: {t.description}")
    print(f"\nExample apply_template:")
    print(apply_template("Do the task.", "T2_decompose2"))
