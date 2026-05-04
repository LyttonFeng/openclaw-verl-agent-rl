from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class VerificationResult:
    score: float
    passed: bool
    breakdown: dict[str, float]
    notes: list[str]


def _lower(value: Any) -> str:
    return str(value or "").lower()


def _email_patterns(email_id: str) -> list[str]:
    suffix = email_id.split("_", 1)[-1]
    return [
        email_id.lower(),
        f"{email_id.lower()}.txt",
        f"email {suffix}",
        f"email-{suffix}",
    ]


def _mentions_email(content: str, email_id: str) -> bool:
    text = _lower(content)
    return any(pattern in text for pattern in _email_patterns(email_id))


def _window_around_email(content: str, email_id: str, radius: int = 700) -> str:
    text = _lower(content)
    positions = [text.find(pattern) for pattern in _email_patterns(email_id)]
    positions = [p for p in positions if p >= 0]
    if not positions:
        return ""
    pos = min(positions)
    return text[max(0, pos - radius) : pos + radius]


def _email_window(content: str, email_ids: list[str], radius: int = 1000) -> str:
    text = _lower(content)
    positions: list[int] = []
    for email_id in email_ids:
        hits = [text.find(pattern) for pattern in _email_patterns(email_id)]
        hits = [p for p in hits if p >= 0]
        if not hits:
            return ""
        positions.append(min(hits))
    return text[max(0, min(positions) - radius) : min(len(text), max(positions) + radius)]


def _priority_near_email(content: str, email_id: str, priority: str) -> bool:
    window = _window_around_email(content, email_id)
    return bool(window and re.search(rf"\b{re.escape(_lower(priority))}\b", window))


def _category_or_action_near_email(content: str, email_id: str) -> bool:
    window = _window_around_email(content, email_id)
    if not window:
        return False
    category_terms = (
        "incident",
        "customer",
        "client",
        "security",
        "release",
        "review",
        "newsletter",
        "spam",
        "vendor",
        "internal",
        "finance",
        "automated",
    )
    action_terms = (
        "action",
        "recommended",
        "respond",
        "reply",
        "review",
        "join",
        "archive",
        "defer",
        "schedule",
        "rotate",
        "escalate",
    )
    return any(term in window for term in category_terms) and any(term in window for term in action_terms)


def _read_report(workspace_path: Path) -> str:
    report_path = workspace_path / "triage_report.md"
    try:
        return report_path.read_text("utf-8", errors="replace")
    except Exception:
        return ""


def _coverage_score(content: str) -> float:
    required = [f"email_{i:02d}" for i in range(1, 14)]
    return sum(1 for email_id in required if _mentions_email(content, email_id)) / len(required)


def _priority_score(content: str, reward_rubric: dict[str, Any]) -> float:
    expected = reward_rubric.get("expected_priorities")
    if not isinstance(expected, dict) or not expected:
        return 1.0
    correct = 0
    for email_id, priority in expected.items():
        if _priority_near_email(content, str(email_id), str(priority)):
            correct += 1
    return correct / len(expected)


def _binding_score(content: str, reward_rubric: dict[str, Any]) -> float:
    expected = reward_rubric.get("expected_bindings")
    if not isinstance(expected, dict) or not expected:
        return 1.0
    correct = 0
    for email_id, spec in expected.items():
        email_id_s = str(email_id)
        if not _mentions_email(content, email_id_s) or not isinstance(spec, dict):
            continue
        clues = spec.get("required_any")
        if not isinstance(clues, list) or not clues:
            correct += 1
            continue
        min_matches = int(spec.get("min_matches", 1) or 1)
        window = _window_around_email(content, email_id_s)
        matches = sum(1 for clue in clues if _lower(clue) in window)
        if matches >= min_matches:
            correct += 1
    return correct / len(expected)


def _incident_score(content: str, reward_rubric: dict[str, Any]) -> float:
    groups = reward_rubric.get("expected_incident_groups")
    if not isinstance(groups, list) or not groups:
        return 1.0
    correct = 0
    for group in groups:
        if not isinstance(group, dict):
            continue
        emails = group.get("emails")
        if not isinstance(emails, list) or not emails:
            continue
        email_ids = [str(email) for email in emails]
        if any(not _mentions_email(content, email_id) for email_id in email_ids):
            continue
        window = _email_window(content, email_ids)
        priority = _lower(group.get("priority"))
        has_priority = bool(priority and re.search(rf"\b{re.escape(priority)}\b", window))
        has_linkage = any(
            term in window
            for term in ("incident", "outage", "alert", "correlated", "linked", "same")
        )
        required_clues = group.get("required_clues")
        clue_hits = 0
        if isinstance(required_clues, list):
            clue_hits = sum(1 for clue in required_clues if _lower(clue) in window)
        has_clues = clue_hits >= 2 if isinstance(required_clues, list) and required_clues else True
        if has_priority and has_linkage and has_clues:
            correct += 1
    return correct / max(1, len(groups))


def _required_fields_score(content: str) -> float:
    required = [f"email_{i:02d}" for i in range(1, 14)]
    return sum(1 for email_id in required if _category_or_action_near_email(content, email_id)) / len(required)


def verify_task16_per_instance(
    workspace_path: str | Path,
    reward_rubric: dict[str, Any],
    threshold: float = 0.72,
    min_coverage: float = 0.90,
    min_priority: float = 0.80,
    min_category: float = 0.75,
    min_required_fields: float = 0.75,
) -> VerificationResult:
    workspace = Path(workspace_path)
    content = _read_report(workspace)
    if not content:
        return VerificationResult(
            score=0.0,
            passed=False,
            breakdown={
                "coverage": 0.0,
                "priority": 0.0,
                "bindings": 0.0,
                "incident": 0.0,
                "required_fields": 0.0,
            },
            notes=["missing triage_report.md"],
        )

    coverage = _coverage_score(content)
    priority = _priority_score(content, reward_rubric)
    bindings = _binding_score(content, reward_rubric)
    incident = _incident_score(content, reward_rubric)
    required_fields = _required_fields_score(content)
    score = (
        0.25 * coverage
        + 0.20 * priority
        + 0.30 * bindings
        + 0.15 * incident
        + 0.10 * required_fields
    )

    notes: list[str] = []
    if coverage < min_coverage:
        notes.append(f"coverage {coverage:.2f} < {min_coverage:.2f}")
    if priority < min_priority:
        notes.append(f"priority {priority:.2f} < {min_priority:.2f}")
    if required_fields < min_required_fields:
        notes.append(f"required_fields {required_fields:.2f} < {min_required_fields:.2f}")
    if bindings < min_category:
        notes.append(f"bindings {bindings:.2f} < {min_category:.2f}")

    passed = bool(score >= threshold and not notes)
    return VerificationResult(
        score=score,
        passed=passed,
        breakdown={
            "coverage": coverage,
            "priority": priority,
            "bindings": bindings,
            "incident": incident,
            "required_fields": required_fields,
        },
        notes=notes,
    )
