#!/usr/bin/env python3
"""
Select valid GRPO samples from a PRM-scored graded_trajectories file.

A "valid GRPO group" = a (task_id) group whose rollout scores have variance
> threshold. Without variance, advantage = (score - mean)/std collapses to 0
and the group contributes nothing to training. Filtering these out before
the train step keeps the optimiser focused on rollouts where there is
something to learn.

Also produces a diagnostics report extending the legacy
`run_meeting_offline_grpo_loop.print_diagnostics_report` with PRM-specific
sections:
  - per-turn +1 / 0 / -1 distribution
  - milestone hit frequency
  - pitfall hit frequency
  - PRM status distribution (ok / no_roadmap / parse_error / ...)
  - per-task PRM mean turn score
  - terminal vs PRM alignment (Pearson correlation, disagreement count)

Usage:
    python3 rl/train/select_grpo_samples.py \
        --graded-file /workspace/.../graded_trajectories_prm.jsonl \
        --output-dir  /workspace/.../selected \
        --variance-threshold 1e-8

Outputs (in --output-dir):
  - graded_trajectories_prm_valid.jsonl   # filtered (only valid groups)
  - prm_diagnostics_report.md             # human-readable analysis
  - selection_summary.json                # machine-readable counts

Stdout: a one-page summary echo of the report's headline stats.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, pstdev


# ─── Stats helpers ──────────────────────────────────────────────────────────


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    """Pearson r between xs and ys (None if degenerate)."""
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mx) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - my) ** 2 for y in ys))
    if den_x == 0 or den_y == 0:
        return None
    return num / (den_x * den_y)


def _safe_var(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    m = mean(values)
    return sum((v - m) ** 2 for v in values) / len(values)


# ─── Loading ────────────────────────────────────────────────────────────────


def load_records(path: Path) -> list[dict]:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


# ─── Diagnostics blocks ─────────────────────────────────────────────────────


def basic_diagnostics(records: list[dict]) -> dict:
    """Mirror of legacy print_diagnostics_report — basic shape of the dataset."""
    total = len(records)
    fatal = sum(1 for r in records if r.get("diagnostics", {}).get("fatal"))
    nonzero = sum(1 for r in records if r.get("score", 0) > 0)
    scores = [float(r.get("score", 0.0)) for r in records]
    mean_score = mean(scores) if scores else 0.0

    failure_tag_counts: dict[str, int] = defaultdict(int)
    for r in records:
        for tag in r.get("diagnostics", {}).get("failure_tags", []) or []:
            failure_tag_counts[tag] += 1

    think_chars = [r.get("diagnostics", {}).get("thinking_chars", 0) or 0 for r in records]
    mean_think = mean(think_chars) if think_chars else 0

    return {
        "total": total,
        "fatal": fatal,
        "nonzero": nonzero,
        "mean_score": mean_score,
        "score_min": min(scores) if scores else 0,
        "score_max": max(scores) if scores else 0,
        "failure_tags": dict(failure_tag_counts),
        "mean_thinking_chars": mean_think,
    }


def prm_diagnostics(records: list[dict]) -> dict:
    """PRM-specific: turn-level distributions + roadmap hit frequencies."""
    status_counts: dict[str, int] = defaultdict(int)
    milestone_counts: dict[str, int] = defaultdict(int)
    pitfall_counts: dict[str, int] = defaultdict(int)
    n_turns_total = 0
    n_pos_total = 0
    n_neg_total = 0
    n_zero_total = 0

    n_with_prm = 0
    n_without_prm = 0
    per_record_prm_mean: list[float] = []

    for r in records:
        status = r.get("prm_status", "missing")
        status_counts[status] += 1

        scores = r.get("prm_turn_scores", []) or []
        if status == "ok" and scores:
            n_with_prm += 1
            n_turns_total += len(scores)
            n_pos_total += sum(1 for s in scores if s > 0)
            n_neg_total += sum(1 for s in scores if s < 0)
            n_zero_total += sum(1 for s in scores if s == 0)
            per_record_prm_mean.append(mean(scores))
        else:
            n_without_prm += 1

        for ms in r.get("prm_milestones", []) or []:
            if ms and ms != "none":
                milestone_counts[ms] += 1
        for pf in r.get("prm_pitfalls", []) or []:
            if pf and pf != "none":
                pitfall_counts[pf] += 1

    return {
        "status_counts": dict(status_counts),
        "n_with_prm_signal": n_with_prm,
        "n_without_prm_signal": n_without_prm,
        "n_turns_total": n_turns_total,
        "n_pos": n_pos_total,
        "n_neg": n_neg_total,
        "n_zero": n_zero_total,
        "milestone_counts": dict(milestone_counts),
        "pitfall_counts": dict(pitfall_counts),
        "per_record_prm_mean_avg": mean(per_record_prm_mean) if per_record_prm_mean else 0.0,
    }


def terminal_vs_prm_alignment(records: list[dict]) -> dict:
    """Cross-check: is high terminal-score correlated with high PRM mean?"""
    pairs: list[tuple[float, float]] = []
    for r in records:
        if r.get("prm_status") != "ok":
            continue
        scores = r.get("prm_turn_scores", []) or []
        if not scores:
            continue
        terminal = float(r.get("score", 0.0))
        prm_mean = mean(scores)
        pairs.append((terminal, prm_mean))

    if len(pairs) < 2:
        return {"n_pairs": len(pairs), "pearson_r": None,
                "disagreements_high_terminal_low_prm": 0,
                "disagreements_low_terminal_high_prm": 0}

    xs, ys = zip(*pairs)
    r = _pearson(list(xs), list(ys))

    # Flag clear disagreements: terminal in top half but PRM mean ≤ 0, or vice versa
    high_t_low_p = sum(1 for t, p in pairs if t >= 0.5 and p <= 0)
    low_t_high_p = sum(1 for t, p in pairs if t <= 0.2 and p >= 0.5)

    return {
        "n_pairs": len(pairs),
        "pearson_r": r,
        "disagreements_high_terminal_low_prm": high_t_low_p,
        "disagreements_low_terminal_high_prm": low_t_high_p,
    }


# ─── GRPO group filter ──────────────────────────────────────────────────────


def _is_bad_trajectory(record: dict) -> bool:
    """Conservative trajectory-level filter for obviously unusable rollouts."""
    if record.get("timed_out"):
        return True
    if record.get("status") not in (None, "success", "ok"):
        return True

    diagnostics = record.get("diagnostics") or {}
    if diagnostics.get("fatal"):
        return True

    features = record.get("policy_features") or {}
    if float(features.get("writes_output", 1.0) or 0.0) <= 0.0:
        return True
    if features.get("expected_output_file") and not features.get("expected_output_exists", True):
        return True
    if float(features.get("multi_read", 1.0) or 0.0) <= 0.0:
        return True
    if float(features.get("output_quality", 1.0) or 0.0) <= 0.0:
        return True
    if int(features.get("output_chars", 500) or 0) < 100:
        return True
    if int(features.get("tool_success", 1) or 0) <= 0:
        return True
    # Agentic process gate: reject trajectories whose PROCESS is not worth
    # imitating, even if they scrape a non-zero score. These are the behaviors
    # that degraded tech_action_items (read-only / context overflow /
    # compaction-before-write). write_calls<1 catches "never wrote a file".
    if int(features.get("write_calls", 1) or 0) < 1:
        return True
    if features.get("context_overflow"):
        return True
    if features.get("compaction_before_write"):
        return True
    if features.get("read_without_write"):
        return True
    if features.get("bad_access_phrase"):
        return True

    # Only enforce quote validity when the output contains several explicit
    # quote-like spans. Some slim tasks are summaries/lists and do not require
    # verbatim quotations, so absence of quotes is not a hard failure here.
    quote_count = int(features.get("quotes", 0) or 0)
    quote_ratio = features.get("quote_verified_ratio")
    if quote_count >= 2 and quote_ratio is not None and float(quote_ratio) < 0.5:
        return True

    return False


def filter_valid_groups(
    records: list[dict],
    variance_threshold: float,
    drop_perfect_tie_groups: bool = False,
    drop_bad_trajectories: bool = False,
) -> tuple[list[dict], dict]:
    """
    Keep only records whose task_id group has score variance > threshold.

    Returns (kept_records, group_summary_dict).
    """
    dropped_bad = [r for r in records if drop_bad_trajectories and _is_bad_trajectory(r)]
    candidate_records = [r for r in records if r not in dropped_bad]

    by_task: dict[str, list[dict]] = defaultdict(list)
    for r in candidate_records:
        by_task[r.get("task_id", "?")].append(r)

    valid_tasks: list[str] = []
    invalid_tasks: list[tuple[str, float, float]] = []  # (tid, score_var, score_mean)
    per_task_rows: list[dict] = []

    for tid, group in by_task.items():
        scores = [float(r.get("score", 0.0)) for r in group]
        var = _safe_var(scores)
        m = mean(scores) if scores else 0.0

        # PRM aggregate for this task
        prm_means_in_group = []
        for r in group:
            ts = r.get("prm_turn_scores", []) or []
            if r.get("prm_status") == "ok" and ts:
                prm_means_in_group.append(mean(ts))
        prm_mean_in_task = mean(prm_means_in_group) if prm_means_in_group else None

        is_perfect_tie = len(scores) > 1 and all(abs(s - 1.0) < 1e-9 for s in scores)
        is_valid = var > variance_threshold
        if drop_perfect_tie_groups and is_perfect_tie:
            is_valid = False
        per_task_rows.append({
            "task_id": tid,
            "n": len(group),
            "score_mean": m,
            "score_var": var,
            "score_std": math.sqrt(var) if var > 0 else 0.0,
            "score_min": min(scores) if scores else 0,
            "score_max": max(scores) if scores else 0,
            "prm_mean": prm_mean_in_task,
            "valid_grpo": is_valid,
        })
        if is_valid:
            valid_tasks.append(tid)
        else:
            invalid_tasks.append((tid, var, m))

    valid_set = set(valid_tasks)
    kept = [r for r in candidate_records if r.get("task_id") in valid_set]

    summary = {
        "n_tasks_total": len(by_task),
        "n_tasks_valid": len(valid_tasks),
        "n_tasks_invalid": len(invalid_tasks),
        "n_records_total": len(records),
        "n_records_candidate": len(candidate_records),
        "n_records_kept": len(kept),
        "n_records_dropped_bad": len(dropped_bad),
        "drop_perfect_tie_groups": drop_perfect_tie_groups,
        "drop_bad_trajectories": drop_bad_trajectories,
        "variance_threshold": variance_threshold,
        "invalid_tasks": [
            {"task_id": t, "score_var": v, "score_mean": m} for t, v, m in invalid_tasks
        ],
        "per_task_rows": per_task_rows,
    }
    return kept, summary


# ─── β recommendation ───────────────────────────────────────────────────────


def recommend_beta(prm: dict, align: dict, *, alpha: float = 1.0) -> dict:
    """
    Estimate whether PRM signal will help training, and recommend a β range.

    Heuristic logic:
      1. If Pearson r(terminal, prm_mean) ≤ 0 → PRM disagrees with terminal
         judge, do NOT use it (β = 0).
      2. Compute "PRM Health Score" = max(0, r) × (1 - fraction_zero_turns),
         in [0, 1]. Higher = stronger and well-aligned PRM signal.
      3. Recommend β so that per-token PRM term β·{-1, 0, +1} is ~1/5 to ~1/2
         the size of terminal term α·terminal_adv (terminal_adv range ≈ ±0.5
         for score ∈ [0, 1]). That puts β in [0.05, 0.25].
      4. Sparser signal (more zero turns) → can afford larger β to amplify
         the rare hits. Denser signal → smaller β to avoid over-pushing.

    Output dict:
      verdict:         "use" | "use_with_caution" | "skip"
      health_score:    float in [0, 1]
      beta_recommended: float (single suggested value)
      beta_range:      (low, high)
      rationale:       human-readable reason string
    """
    n_turns = prm.get("n_turns_total", 0)
    n_zero = prm.get("n_zero", 0)
    pearson = align.get("pearson_r")

    # Edge: no PRM signal at all
    if n_turns == 0:
        return {
            "verdict": "skip",
            "health_score": 0.0,
            "beta_recommended": 0.0,
            "beta_range": (0.0, 0.0),
            "rationale": "No PRM signal collected (n_turns=0). Run PRM scoring first.",
        }

    frac_zero = n_zero / n_turns
    frac_nonzero = 1.0 - frac_zero

    # Edge: judge has no opinion (almost everything is 0)
    if frac_nonzero < 0.1:
        return {
            "verdict": "skip",
            "health_score": 0.0,
            "beta_recommended": 0.0,
            "beta_range": (0.0, 0.0),
            "rationale": (
                f"PRM judge calls {frac_zero*100:.0f}% of turns 'neutral (0)'. "
                f"Too sparse — set β=0 and revise the roadmap to make milestones/"
                f"pitfalls fire more reliably."
            ),
        }

    # Edge: alignment unknown (too few pairs)
    if pearson is None:
        return {
            "verdict": "use_with_caution",
            "health_score": frac_nonzero * 0.3,  # uncertain
            "beta_recommended": 0.05,
            "beta_range": (0.0, 0.1),
            "rationale": (
                f"Too few paired records to compute Pearson r. PRM signal density "
                f"OK ({frac_nonzero*100:.0f}% nonzero) but alignment unverified — "
                f"start with small β=0.05."
            ),
        }

    # Edge: misaligned with terminal
    if pearson < 0:
        return {
            "verdict": "skip",
            "health_score": 0.0,
            "beta_recommended": 0.0,
            "beta_range": (0.0, 0.0),
            "rationale": (
                f"Pearson r={pearson:+.3f} — PRM judge DISAGREES with terminal "
                f"judge. PRM is noise, not signal. β=0 until roadmap is fixed."
            ),
        }

    # Healthy regime: r > 0
    health = max(0.0, pearson) * frac_nonzero  # in [0, 1]

    # β recommendation: aim for per-token PRM term to be ≈ alpha·0.5 × scale
    # where scale ∈ [0.1, 0.5] depending on health.
    # → β ≈ alpha × 0.5 × scale × (1 / max(turn_score_magnitude=1)) = α × 0.5 × scale
    # Sparse high-quality signal → use upper end of the band.
    if frac_nonzero < 0.3 and pearson > 0.5:
        # Rare but precise hits — amplify
        scale_low, scale_high = 0.2, 0.5
    elif pearson > 0.5:
        # Dense and aligned — moderate
        scale_low, scale_high = 0.1, 0.3
    else:
        # 0 < r ≤ 0.5: weak alignment — keep β small
        scale_low, scale_high = 0.05, 0.15

    beta_low = alpha * 0.5 * scale_low
    beta_high = alpha * 0.5 * scale_high
    beta_rec = (beta_low + beta_high) / 2.0

    # Verdict
    if pearson >= 0.5 and frac_nonzero >= 0.3:
        verdict = "use"
    elif pearson >= 0.3 or frac_nonzero >= 0.2:
        verdict = "use_with_caution"
    else:
        verdict = "use_with_caution"

    rationale = (
        f"Pearson r={pearson:+.3f} (PRM↔terminal alignment) × "
        f"{frac_nonzero*100:.0f}% nonzero turns → health={health:.2f}. "
        f"Recommended β range [{beta_low:.2f}, {beta_high:.2f}], suggested {beta_rec:.2f} "
        f"(α={alpha} fixed). Current default β=0.1 is "
        f"{'inside' if beta_low <= 0.1 <= beta_high else 'OUTSIDE'} this range."
    )

    return {
        "verdict": verdict,
        "health_score": health,
        "beta_recommended": beta_rec,
        "beta_range": (beta_low, beta_high),
        "rationale": rationale,
    }


# ─── Report writers ─────────────────────────────────────────────────────────


def write_markdown_report(
    out_path: Path,
    *,
    graded_file: Path,
    basic: dict,
    prm: dict,
    align: dict,
    selection: dict,
    beta_rec: dict,
) -> None:
    lines = []
    lines.append(f"# PRM Diagnostics Report\n")
    lines.append(f"**Source:** `{graded_file}`\n")
    lines.append(f"**Records:** {basic['total']}  |  **Tasks:** {selection['n_tasks_total']}  "
                 f"|  **Variance threshold:** {selection['variance_threshold']}\n")
    lines.append("\n---\n\n")

    # 1. Overall
    lines.append("## 1. Overall\n\n")
    lines.append(f"- Total trajectories: **{basic['total']}** (fatal: {basic['fatal']}, "
                 f"nonzero score: {basic['nonzero']})\n")
    lines.append(f"- Score: mean=**{basic['mean_score']:.3f}**, "
                 f"range=[{basic['score_min']:.2f}, {basic['score_max']:.2f}]\n")
    lines.append(f"- Mean thinking chars: {basic['mean_thinking_chars']:.0f}\n")
    if basic["failure_tags"]:
        lines.append("\n**Failure tags:**\n\n")
        for tag, c in sorted(basic["failure_tags"].items(), key=lambda x: -x[1]):
            pct = c / basic["total"] * 100 if basic["total"] else 0
            lines.append(f"- `{tag}` — {c} ({pct:.0f}%)\n")
    lines.append("\n")

    # 2. GRPO Signal
    lines.append("## 2. GRPO Signal (group-level score variance)\n\n")
    lines.append(f"- Tasks with variance > {selection['variance_threshold']}: "
                 f"**{selection['n_tasks_valid']} / {selection['n_tasks_total']}**\n")
    lines.append(f"- Records kept: **{selection['n_records_kept']} / "
                 f"{selection['n_records_total']}**\n")
    if selection["invalid_tasks"]:
        lines.append("\n**Filtered out (no variance):**\n\n")
        lines.append("| task_id | score_var | score_mean |\n|---|---|---|\n")
        for t in sorted(selection["invalid_tasks"], key=lambda x: x["score_mean"]):
            lines.append(f"| {t['task_id']} | {t['score_var']:.4f} | {t['score_mean']:.2f} |\n")
    lines.append("\n")

    # 3. PRM Process Signal
    lines.append("## 3. PRM Process Signal\n\n")
    lines.append(f"- Records with PRM signal (status=ok, turns>0): "
                 f"**{prm['n_with_prm_signal']}** / {basic['total']}\n")
    lines.append(f"- Total turns scored: **{prm['n_turns_total']}**  →  "
                 f"+1: {prm['n_pos']} ({prm['n_pos']/max(prm['n_turns_total'],1)*100:.0f}%), "
                 f"0: {prm['n_zero']} ({prm['n_zero']/max(prm['n_turns_total'],1)*100:.0f}%), "
                 f"-1: {prm['n_neg']} ({prm['n_neg']/max(prm['n_turns_total'],1)*100:.0f}%)\n")
    lines.append(f"- Avg PRM mean per record: **{prm['per_record_prm_mean_avg']:+.3f}**\n")
    lines.append("\n**PRM status distribution:**\n\n")
    for st, c in sorted(prm["status_counts"].items(), key=lambda x: -x[1]):
        lines.append(f"- `{st}` — {c}\n")

    if prm["milestone_counts"]:
        lines.append("\n**Milestone hits (top 10):**\n\n")
        for ms, c in sorted(prm["milestone_counts"].items(), key=lambda x: -x[1])[:10]:
            lines.append(f"- `{ms}` — {c}\n")

    if prm["pitfall_counts"]:
        lines.append("\n**Pitfall hits (top 10):**\n\n")
        for pf, c in sorted(prm["pitfall_counts"].items(), key=lambda x: -x[1])[:10]:
            lines.append(f"- `{pf}` — {c}\n")
    lines.append("\n")

    # 4. Terminal × PRM Alignment
    lines.append("## 4. Terminal × PRM Alignment\n\n")
    pearson = align["pearson_r"]
    pearson_str = f"{pearson:+.3f}" if pearson is not None else "N/A"
    lines.append(f"- Paired records: {align['n_pairs']}\n")
    lines.append(f"- Pearson r (terminal_score, prm_mean): **{pearson_str}**\n")
    lines.append(f"  - r > 0.5: PRM aligned with terminal judge ✅\n")
    lines.append(f"  - 0 < r < 0.5: weakly aligned ⚠\n")
    lines.append(f"  - r ≤ 0: misaligned, PRM signal not useful ❌\n")
    lines.append(f"- Disagreements:\n")
    lines.append(f"  - high terminal (≥0.5) but low PRM (≤0): "
                 f"{align['disagreements_high_terminal_low_prm']}\n")
    lines.append(f"  - low terminal (≤0.2) but high PRM (≥0.5): "
                 f"{align['disagreements_low_terminal_high_prm']}\n")
    lines.append("\n")

    # 5. β Recommendation
    lines.append("## 5. β Recommendation (PRM weight)\n\n")
    lines.append(f"- **Verdict:** `{beta_rec['verdict']}`\n")
    lines.append(f"- **PRM Health Score:** {beta_rec['health_score']:.2f}  "
                 f"(0 = useless, 1 = strong & aligned)\n")
    lines.append(f"- **Recommended β:** **{beta_rec['beta_recommended']:.3f}**  "
                 f"(range [{beta_rec['beta_range'][0]:.3f}, "
                 f"{beta_rec['beta_range'][1]:.3f}])\n")
    lines.append(f"- **Rationale:** {beta_rec['rationale']}\n")
    lines.append("\n")

    # 6. Per-task detail
    lines.append("## 6. Per-task Detail\n\n")
    lines.append("| task_id | n | score_mean | score_var | prm_mean | valid |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for row in sorted(selection["per_task_rows"], key=lambda r: (-int(r["valid_grpo"]),
                                                                  r["task_id"])):
        prm_str = f"{row['prm_mean']:+.2f}" if row["prm_mean"] is not None else "—"
        valid_mark = "✅" if row["valid_grpo"] else "❌"
        lines.append(f"| {row['task_id']} | {row['n']} | {row['score_mean']:.2f} | "
                     f"{row['score_var']:.4f} | {prm_str} | {valid_mark} |\n")

    out_path.write_text("".join(lines))


def print_stdout_summary(*, basic: dict, prm: dict, align: dict, selection: dict,
                          beta_rec: dict) -> None:
    print("=" * 70)
    print("  PRM ROUND DIAGNOSTICS — HEADLINE")
    print("=" * 70)
    print(f"  Records:           {basic['total']} (fatal {basic['fatal']}, "
          f"nonzero {basic['nonzero']})")
    print(f"  Score:             mean={basic['mean_score']:.3f} "
          f"[min {basic['score_min']:.2f}, max {basic['score_max']:.2f}]")
    print(f"  Tasks valid GRPO:  {selection['n_tasks_valid']} / "
          f"{selection['n_tasks_total']}  (kept {selection['n_records_kept']} records)")
    print(f"  PRM signal:        {prm['n_with_prm_signal']} records, "
          f"{prm['n_turns_total']} turns total")
    if prm["n_turns_total"] > 0:
        print(f"  PRM turns:         +1: {prm['n_pos']}  0: {prm['n_zero']}  "
              f"-1: {prm['n_neg']}  (avg per-record mean {prm['per_record_prm_mean_avg']:+.3f})")
    pr = align["pearson_r"]
    pr_s = f"{pr:+.3f}" if pr is not None else "N/A"
    print(f"  Terminal×PRM r:    {pr_s}  ({align['n_pairs']} pairs)")
    if prm["pitfall_counts"]:
        top_pf = sorted(prm["pitfall_counts"].items(), key=lambda x: -x[1])[:3]
        print(f"  Top pitfalls:      {', '.join(f'{k}({v})' for k, v in top_pf)}")
    if prm["milestone_counts"]:
        top_ms = sorted(prm["milestone_counts"].items(), key=lambda x: -x[1])[:3]
        print(f"  Top milestones:    {', '.join(f'{k}({v})' for k, v in top_ms)}")
    print(f"  β verdict:         {beta_rec['verdict']}  "
          f"(health {beta_rec['health_score']:.2f})")
    print(f"  β recommended:     {beta_rec['beta_recommended']:.3f}  "
          f"(range [{beta_rec['beta_range'][0]:.3f}, "
          f"{beta_rec['beta_range'][1]:.3f}])")
    print(f"  Reason:            {beta_rec['rationale']}")
    print("=" * 70)


# ─── Main ───────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--graded-file", required=True,
                    help="Path to graded_trajectories_prm.jsonl")
    ap.add_argument("--output-dir", required=True,
                    help="Where to write the filtered jsonl + report")
    ap.add_argument("--variance-threshold", type=float, default=1e-8,
                    help="Group score variance must exceed this to be 'valid GRPO'")
    ap.add_argument("--alpha", type=float, default=1.0,
                    help="Terminal advantage coefficient (only used to scale β recommendation)")
    ap.add_argument("--drop-perfect-tie-groups", action="store_true",
                    help="Drop groups where every rollout scored exactly 1.0")
    ap.add_argument("--drop-bad-trajectories", action="store_true",
                    help="Drop fatal/timed-out/no-output/no-tool-trace trajectories before group filtering")
    args = ap.parse_args()

    in_path = Path(args.graded_file)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading {in_path} ...")
    records = load_records(in_path)
    print(f"  → {len(records)} records")

    basic = basic_diagnostics(records)
    prm = prm_diagnostics(records)
    align = terminal_vs_prm_alignment(records)
    kept, selection = filter_valid_groups(
        records,
        args.variance_threshold,
        drop_perfect_tie_groups=args.drop_perfect_tie_groups,
        drop_bad_trajectories=args.drop_bad_trajectories,
    )
    beta_rec = recommend_beta(prm, align, alpha=args.alpha)

    # Write filtered jsonl
    valid_path = out_dir / "graded_trajectories_prm_valid.jsonl"
    with open(valid_path, "w") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Filtered jsonl → {valid_path}  ({len(kept)} records)")

    # Write markdown report
    md_path = out_dir / "prm_diagnostics_report.md"
    write_markdown_report(md_path,
                          graded_file=in_path,
                          basic=basic, prm=prm, align=align, selection=selection,
                          beta_rec=beta_rec)
    print(f"Report → {md_path}")

    # Write JSON summary (machine-readable)
    summary_path = out_dir / "selection_summary.json"
    summary = {
        "input": str(in_path),
        "variance_threshold": args.variance_threshold,
        "basic": basic,
        "prm": prm,
        "alignment": align,
        "selection": {
            k: v for k, v in selection.items() if k != "per_task_rows"
        },
        "beta_recommendation": beta_rec,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary JSON → {summary_path}")

    print()
    print_stdout_summary(basic=basic, prm=prm, align=align, selection=selection,
                          beta_rec=beta_rec)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
