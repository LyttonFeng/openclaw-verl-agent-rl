#!/usr/bin/env python3
"""Score a meeting_analysis rollout against consensus gold and trajectory policy."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

from common import normalize_text, read_jsonl

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))


def read_transcript_text(path: Path) -> str:
    if not path:
        return ""
    text = path.read_text(encoding="utf-8", errors="replace")
    return text


def workspace_text(workspace: Path) -> str:
    if not workspace.exists():
        return ""
    chunks = []
    for file in workspace.rglob("*"):
        if file.is_file() and file.suffix.lower() in {".md", ".txt", ".json", ".csv"}:
            if file.name in {"BOOTSTRAP.md", "AGENTS.md", "IDENTITY.md", "HEARTBEAT.md", "USER.md", "TOOLS.md", "SOUL.md"}:
                continue
            try:
                chunks.append(f"\n# {file.name}\n{file.read_text(encoding='utf-8', errors='replace')}")
            except OSError:
                pass
    return "\n".join(chunks)


def claim_hit(claim: str, output: str, *, threshold: float = 0.68) -> bool:
    claim_norm = normalize_text(claim)
    output_norm = normalize_text(output)
    if not claim_norm or not output_norm:
        return False
    words = [w for w in claim_norm.split() if len(w) > 2]
    if not words:
        return False
    unique_words = set(words)
    hits = sum(1 for word in unique_words if word in output_norm)
    numbers = re.findall(r"\b\d+(?:\.\d+)?\b", claim_norm)
    number_ok = all(n in output_norm for n in numbers[:4])
    return hits / max(len(unique_words), 1) >= threshold and number_ok


def trajectory_policy_features(transcript_text: str) -> dict[str, float]:
    text = transcript_text.lower()
    read_calls = len(re.findall(r"\b(read_file|read|grep|search|rg)\b", text))
    write_calls = len(re.findall(r"\b(write_file|write|edit)\b", text))
    verify_terms = len(re.findall(r"\b(verify|check|cross-check|audit|confirm|validate)\b", text))
    return {
        "multi_read": min(1.0, read_calls / 6.0),
        "writes_output": min(1.0, write_calls / 2.0),
        "pre_final_verification": min(1.0, verify_terms / 8.0),
    }


def score(gold: dict[str, Any], transcript_path: Path, workspace_path: Path) -> dict[str, Any]:
    transcript = read_transcript_text(transcript_path)
    output = workspace_text(workspace_path)
    ledger = gold.get("consensus_ledger", [])
    low_agreement = gold.get("low_agreement_items", [])
    hits = []
    misses = []
    for item in ledger:
        if claim_hit(item.get("claim", ""), output):
            hits.append(item)
        else:
            misses.append(item)
    low_hits = [item for item in low_agreement if claim_hit(item.get("claim", ""), output)]
    recall = len(hits) / len(ledger) if ledger else 0.0
    low_agreement_recall = len(low_hits) / len(low_agreement) if low_agreement else 0.0
    policy = trajectory_policy_features(transcript)
    output_quality = min(1.0, len(output) / 3500.0)
    policy["output_quality"] = output_quality
    policy_score = (
        0.30 * policy["multi_read"]
        + 0.25 * policy["writes_output"]
        + 0.25 * policy["pre_final_verification"]
        + 0.20 * output_quality
    )
    final_score = 0.65 * recall + 0.20 * low_agreement_recall + 0.15 * policy_score
    return {
        "score": round(final_score, 4),
        "gold_recall": round(recall, 4),
        "low_agreement_recall": round(low_agreement_recall, 4),
        "policy_score": round(policy_score, 4),
        "policy_features": policy,
        "gold_total": len(ledger),
        "gold_hits": len(hits),
        "low_agreement_total": len(low_agreement),
        "low_agreement_hits": len(low_hits),
        "missed_claims": [m.get("claim", "") for m in misses[:20]],
    }


def find_gold(path: Path, task_id: str) -> dict[str, Any]:
    if path.suffix == ".jsonl":
        for row in read_jsonl(path):
            if row.get("task_id") == task_id:
                return row
        raise RuntimeError(f"Task {task_id} not found in {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Score rollout with consensus gold and trajectory policy.")
    parser.add_argument("--gold", required=True, help="Consensus gold JSON or JSONL.")
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--transcript", required=True)
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--output")
    args = parser.parse_args()

    gold = find_gold(Path(args.gold), args.task_id)
    result = score(gold, Path(args.transcript), Path(args.workspace))
    payload = {"task_id": args.task_id, **result}
    text = json.dumps(payload, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
