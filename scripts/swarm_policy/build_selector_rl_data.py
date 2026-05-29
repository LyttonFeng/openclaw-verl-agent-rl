#!/usr/bin/env python3
"""Build select-only swarm-policy RL data from candidate-merge results.

Each sample asks a lead model to choose the best candidate report from A/B/C.
The reward for a choice is the already-computed terminal score of that
candidate, so selector RL can run without invoking the full benchmark in the
inner training loop.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


CHOICES = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


def read_text(path: Path, max_chars: int) -> str:
    if not path.exists():
        return ""
    text = path.read_text(encoding="utf-8", errors="replace").strip()
    if len(text) > max_chars:
        return text[:max_chars] + "\n\n[TRUNCATED]"
    return text


def read_result(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def candidate_output(result_dir: Path, candidate: dict[str, Any], output_file: str, max_chars: int) -> str:
    workspace = candidate.get("workspace_path")
    paths: list[Path] = []
    if workspace:
        paths.append(Path(workspace) / output_file)
        if not Path(workspace).is_absolute():
            paths.append(result_dir / workspace / output_file)
    paths.append(result_dir / f"candidate_{candidate.get('candidate_id')}" / output_file)
    for path in paths:
        text = read_text(path, max_chars)
        if text:
            return text
    return f"(candidate did not write {output_file})"


def build_prompt(task_id: str, output_file: str, candidates: list[dict[str, str]]) -> str:
    blocks = []
    for item in candidates:
        blocks.append(f"### Candidate {item['label']}\n{item['text']}")
    return (
        "You are training a swarm selector policy.\n\n"
        "Choose the candidate report that is most likely to receive the highest benchmark score.\n"
        "Do not merge or rewrite the report. Only choose one candidate.\n\n"
        f"Task id: {task_id}\n"
        f"Required output file: {output_file}\n\n"
        + "\n\n".join(blocks)
        + "\n\nReturn only JSON in this exact form:\n"
        '{"choice":"A"}'
    )


def build_record(path: Path, *, min_gap: float, max_candidate_chars: int) -> dict[str, Any] | None:
    result = read_result(path)
    result_dir = path.parent
    candidates_raw = result.get("candidates", [])
    scores = [float(x) for x in result.get("candidate_scores", [])]
    if len(candidates_raw) < 2 or len(scores) != len(candidates_raw):
        return None

    sorted_scores = sorted(scores, reverse=True)
    gap = sorted_scores[0] - sorted_scores[1] if len(sorted_scores) > 1 else sorted_scores[0]
    if gap < min_gap:
        return None

    output_file = result.get("output_file") or "output.md"
    task_id = result.get("task_id") or path.parent.name
    candidates = []
    reward_map: dict[str, float] = {}
    for idx, candidate in enumerate(candidates_raw):
        label = CHOICES[idx]
        text = candidate_output(result_dir, candidate, output_file, max_candidate_chars)
        candidates.append({"label": label, "text": text})
        reward_map[label] = scores[idx]

    best_idx = max(range(len(scores)), key=lambda i: scores[i])
    best_choice = CHOICES[best_idx]
    return {
        "sample_id": f"{task_id}::{path.parent.name}",
        "task_id": task_id,
        "source_result": str(path),
        "mode": "select_only",
        "prompt": build_prompt(task_id, output_file, candidates),
        "choices": [c["label"] for c in candidates],
        "candidate_scores": scores,
        "reward_map": reward_map,
        "best_choice": best_choice,
        "best_score": scores[best_idx],
        "score_gap": gap,
        "output_file": output_file,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", help="result.json files or directories containing result.json")
    parser.add_argument("--output", required=True)
    parser.add_argument("--min-gap", type=float, default=0.03)
    parser.add_argument("--max-candidate-chars", type=int, default=12000)
    args = parser.parse_args()

    paths: list[Path] = []
    for item in args.results:
        p = Path(item)
        if p.is_dir():
            paths.extend(sorted(p.rglob("result.json")))
        else:
            paths.append(p)

    rows = []
    skipped = 0
    for path in paths:
        try:
            row = build_record(path, min_gap=args.min_gap, max_candidate_chars=args.max_candidate_chars)
        except Exception as exc:
            print(f"skip {path}: {exc}")
            skipped += 1
            continue
        if row is None:
            skipped += 1
            continue
        rows.append(row)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(rows)} selector samples to {out}; skipped={skipped}")


if __name__ == "__main__":
    main()
