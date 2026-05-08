"""
Roadmap PRM schema.

A Roadmap is a per-task "map" used by an LLM judge to score whether each agent
turn is making progress toward task completion.

Design principles:
- Roadmap is NOT a step-by-step script. The agent is free to traverse it in
  any order; the roadmap just tells the judge what counts as progress.
- Milestones are checkpoints. Each turn may hit 0..N milestones.
- Pitfalls are explicit anti-patterns that earn -1.
- Signals are concrete features (file names, keywords, output structure) that
  let the judge make a decision without relying on vague intuition.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class Milestone:
    id: str
    desc: str
    signals: list[str] = field(default_factory=list)


@dataclass
class Pitfall:
    id: str
    desc: str
    severity: str = "medium"  # "low" | "medium" | "high"


@dataclass
class Roadmap:
    task_id: str
    goal: str
    inputs: list[str]
    output: str
    milestones: list[Milestone]
    pitfalls: list[Pitfall]
    notes: str = ""

    @classmethod
    def from_yaml(cls, path: str | Path) -> "Roadmap":
        data = yaml.safe_load(Path(path).read_text())
        return cls(
            task_id=data["task_id"],
            goal=data["goal"],
            inputs=list(data.get("inputs", [])),
            output=data["output"],
            milestones=[Milestone(**m) for m in data["milestones"]],
            pitfalls=[Pitfall(**p) for p in data["pitfalls"]],
            notes=data.get("notes", ""),
        )

    def to_judge_text(self) -> str:
        """Render the roadmap as a compact text block for the judge prompt."""
        lines = [
            f"Task: {self.task_id}",
            f"Goal: {self.goal}",
            f"Inputs: {', '.join(self.inputs)}",
            f"Expected output: {self.output}",
            "",
            "Milestones (any of these = +1 progress when first hit or substantially advanced):",
        ]
        for m in self.milestones:
            lines.append(f"  [{m.id}] {m.desc}")
            if m.signals:
                lines.append(f"    Signals: {'; '.join(m.signals)}")
        lines.append("")
        lines.append("Pitfalls (hitting any = -1 regression):")
        for p in self.pitfalls:
            lines.append(f"  [{p.id}] ({p.severity}) {p.desc}")
        if self.notes:
            lines.append("")
            lines.append(f"Notes: {self.notes}")
        return "\n".join(lines)
