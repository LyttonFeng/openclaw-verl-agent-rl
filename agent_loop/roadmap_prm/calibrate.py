"""
Calibration: validate that a roadmap+rubric correctly separates expert
trajectories from failure trajectories.

Goal: tune the roadmap until
  - expert mean turn score >= +0.5 (most turns score +1, few 0, very rare -1)
  - failure mean turn score <= +0.2
  - separation (expert_mean - failure_mean) >= 0.5

Usage (programmatic):
    from agent_loop.roadmap_prm.calibrate import calibrate_roadmap
    report = calibrate_roadmap(
        roadmap_path="agent_loop/roadmap_prm/roadmaps/task_meeting_blog_post.yaml",
        expert_trajectories=[(path, prompt), ...],
        failure_trajectories=[(path, prompt), ...],
    )
"""

from __future__ import annotations

import json
import statistics
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .judge import JudgeResult, judge_trajectory
from .schema import Roadmap
from .trajectory import Turn, parse_openclaw_jsonl


@dataclass
class TrajectoryScore:
    label: str  # "expert" or "failure"
    path: str
    n_turns: int
    scores: list[int]
    milestones: list[str]
    pitfalls: list[str]
    reasons: list[str]
    mean: float

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class CalibrationReport:
    roadmap_path: str
    expert: list[TrajectoryScore] = field(default_factory=list)
    failure: list[TrajectoryScore] = field(default_factory=list)

    def expert_mean(self) -> float:
        if not self.expert:
            return 0.0
        all_scores = [s for ts in self.expert for s in ts.scores]
        return statistics.mean(all_scores) if all_scores else 0.0

    def failure_mean(self) -> float:
        if not self.failure:
            return 0.0
        all_scores = [s for ts in self.failure for s in ts.scores]
        return statistics.mean(all_scores) if all_scores else 0.0

    def separation(self) -> float:
        return self.expert_mean() - self.failure_mean()

    def summary(self) -> str:
        lines = [
            f"Roadmap: {self.roadmap_path}",
            f"Expert trajectories: {len(self.expert)}",
            f"  mean turn score: {self.expert_mean():+.3f}",
        ]
        if self.expert:
            for ts in self.expert:
                lines.append(f"  - {Path(ts.path).name}: n={ts.n_turns} mean={ts.mean:+.2f} scores={ts.scores}")
        lines.append(f"Failure trajectories: {len(self.failure)}")
        lines.append(f"  mean turn score: {self.failure_mean():+.3f}")
        if self.failure:
            for ts in self.failure:
                lines.append(f"  - {Path(ts.path).name}: n={ts.n_turns} mean={ts.mean:+.2f} scores={ts.scores}")
        lines.append(f"Separation: {self.separation():+.3f} (target >= +0.5)")

        # Pitfall and milestone hit rates
        pf_counts: dict[str, int] = {}
        for ts in self.failure:
            for p in ts.pitfalls:
                if p != "none":
                    pf_counts[p] = pf_counts.get(p, 0) + 1
        if pf_counts:
            lines.append("Pitfalls hit on failure trajectories:")
            for p, c in sorted(pf_counts.items(), key=lambda x: -x[1]):
                lines.append(f"  {p}: {c}")
        return "\n".join(lines)

    def to_json(self) -> str:
        return json.dumps({
            "roadmap_path": self.roadmap_path,
            "expert_mean": self.expert_mean(),
            "failure_mean": self.failure_mean(),
            "separation": self.separation(),
            "expert": [ts.to_dict() for ts in self.expert],
            "failure": [ts.to_dict() for ts in self.failure],
        }, indent=2, ensure_ascii=False)


def _score_one_trajectory(
    roadmap: Roadmap,
    path: str | Path,
    prompt: str,
    label: str,
    max_turns: int | None = None,
) -> TrajectoryScore:
    turns = parse_openclaw_jsonl(path)
    if max_turns is not None:
        turns = turns[:max_turns]
    if not turns:
        return TrajectoryScore(label=label, path=str(path), n_turns=0,
                               scores=[], milestones=[], pitfalls=[], reasons=[], mean=0.0)
    results = judge_trajectory(roadmap, prompt, turns)
    scores = [r.score for r in results]
    return TrajectoryScore(
        label=label,
        path=str(path),
        n_turns=len(turns),
        scores=scores,
        milestones=[r.milestone for r in results],
        pitfalls=[r.pitfall for r in results],
        reasons=[r.reason for r in results],
        mean=statistics.mean(scores) if scores else 0.0,
    )


def calibrate_roadmap(
    roadmap_path: str | Path,
    expert_trajectories: list[tuple[str, str]],  # [(jsonl_path, task_prompt), ...]
    failure_trajectories: list[tuple[str, str]],
    max_turns_per_trajectory: int | None = None,
    verbose: bool = True,
) -> CalibrationReport:
    """Run the judge over expert and failure trajectories, report separation."""
    roadmap = Roadmap.from_yaml(roadmap_path)
    report = CalibrationReport(roadmap_path=str(roadmap_path))

    if verbose:
        print(f"=== Calibrating {roadmap.task_id} ===")
        print(f"Expert: {len(expert_trajectories)} trajectories")
        print(f"Failure: {len(failure_trajectories)} trajectories")

    for path, prompt in expert_trajectories:
        if verbose:
            print(f"  Scoring expert: {Path(path).name}")
        ts = _score_one_trajectory(roadmap, path, prompt, "expert",
                                   max_turns=max_turns_per_trajectory)
        report.expert.append(ts)
        if verbose:
            print(f"    n_turns={ts.n_turns} mean={ts.mean:+.2f} scores={ts.scores}")

    for path, prompt in failure_trajectories:
        if verbose:
            print(f"  Scoring failure: {Path(path).name}")
        ts = _score_one_trajectory(roadmap, path, prompt, "failure",
                                   max_turns=max_turns_per_trajectory)
        report.failure.append(ts)
        if verbose:
            print(f"    n_turns={ts.n_turns} mean={ts.mean:+.2f} scores={ts.scores}")

    if verbose:
        print()
        print(report.summary())
    return report
