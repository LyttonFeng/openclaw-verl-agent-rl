#!/usr/bin/env python3
"""Standalone smoke test for task16 data and event reward logic.

This does not start veRL, vLLM, or OpenClaw. It verifies the pieces that should
work on any engineer machine before launching RL:

- canonical task16 markdown exists
- generated parquet files have the expected 91/11 rows plus synthetic split
- task16 event reward gives positive signal for a plausible successful trace
"""

from __future__ import annotations

import argparse
import tempfile
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from rewards.task16_event_reward import compute_episode_rewards  # noqa: E402
from rl.agent_loop.per_instance_verifier import verify_task16_per_instance  # noqa: E402


def _assert_task_data(data_dir: Path) -> None:
    train_path = data_dir / "train.parquet"
    val_path = data_dir / "val.parquet"
    if not train_path.is_file() or not val_path.is_file():
        raise AssertionError(
            f"Missing parquet files under {data_dir}. Run scripts/build_task16_prompts.py first."
        )

    train = pd.read_parquet(train_path)
    val = pd.read_parquet(val_path)
    if len(train) != 91 or len(val) != 11:
        raise AssertionError(f"Expected train=91 val=11 rows, got train={len(train)} val={len(val)}")

    for name, frame in (("train", train), ("val", val)):
        task_ids = {
            item.get("task_id")
            for item in frame["extra_info"]
            if isinstance(item, dict)
        }
        if task_ids != {"task_16_email_triage"}:
            raise AssertionError(f"{name} has unexpected task ids: {sorted(task_ids)}")

    synth_count = sum(
        1
        for item in train["extra_info"]
        if isinstance(item, dict) and item.get("synthetic_instance") is True
    )
    if synth_count != 20:
        raise AssertionError(f"Expected 20 synthetic train rows, got {synth_count}")

    stage2_path = data_dir / "train_stage2_balanced.parquet"
    if not stage2_path.is_file():
        raise AssertionError(f"Missing stage2 parquet: {stage2_path}")
    stage2 = pd.read_parquet(stage2_path)
    stage2_synth_count = sum(
        1
        for item in stage2["extra_info"]
        if isinstance(item, dict) and item.get("synthetic_instance") is True
    )
    if len(stage2) != 32 or stage2_synth_count != 20:
        raise AssertionError(
            f"Expected stage2=32 rows with 20 synthetic, got rows={len(stage2)} synthetic={stage2_synth_count}"
        )

    val_synth_path = data_dir / "val_synth5.parquet"
    if not val_synth_path.is_file():
        raise AssertionError(f"Missing synthetic val parquet: {val_synth_path}")
    val_synth = pd.read_parquet(val_synth_path)
    val_synth_count = sum(
        1
        for item in val_synth["extra_info"]
        if isinstance(item, dict) and item.get("synthetic_instance") is True
    )
    if len(val_synth) != 5 or val_synth_count != 5:
        raise AssertionError(
            f"Expected val_synth5=5 synthetic rows, got rows={len(val_synth)} synthetic={val_synth_count}"
        )


def _assert_reward_smoke() -> None:
    report = """## Incident Groups

### Production database incident
- Covered emails: email_01, email_13
- Priority: P0
- Category: production incident
- Rationale: customer-facing 500 errors and correlated latency alert.
- Recommended action: join war room and coordinate SRE/backend response.

## Standalone Items

- email_05: Priority P1, Category customer, Recommended action: schedule call and send SOC 2.
- email_08: Priority P1, Category security, Recommended action: rotate credentials by deadline.
- email_02: Priority P3, Category marketing, Recommended action: review by Wednesday.
- email_03: Priority P4, Category dependency update, Recommended action: monitor passing CI.
- email_04: Priority P3, Category HR, Recommended action: complete benefits enrollment.
- email_06: Priority P4, Category notification, Recommended action: ignore or review later.
- email_07: Priority P2, Category management, Recommended action: finish self-assessment.
- email_09: Priority P4, Category newsletter, Recommended action: no action.
- email_10: Priority P2, Category code review, Recommended action: review auth refactor.
- email_11: Priority P4, Category vendor promo, Recommended action: ignore.
- email_12: Priority P3, Category finance, Recommended action: review budget notes.
"""
    trajectory = [
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "read",
                        "arguments": '{"path":"inbox/email_01.txt"}',
                    }
                },
                {
                    "function": {
                        "name": "read",
                        "arguments": '{"path":"inbox/email_13.txt"}',
                    }
                },
            ],
        },
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "function": {
                        "name": "write",
                        "arguments": '{"path":"triage_report.md","content":' + repr(report) + "}",
                    }
                }
            ],
        },
    ]
    rewards = compute_episode_rewards(
        trajectory,
        terminal_success=True,
        task_id="task_16_email_triage",
        extra_info={},
    )
    if len(rewards) != 2:
        raise AssertionError(f"Expected 2 turn rewards, got {len(rewards)}: {rewards}")
    if sum(rewards) <= 0:
        raise AssertionError(f"Expected positive task16 reward, got {rewards}")


def _assert_per_instance_verifier_smoke(tmp_dir: Path) -> None:
    report = """## Incident Groups

### AsterBank checkout incident
- Covered emails: email_01, email_13
- Priority: P0
- Category: production incident
- Rationale: AsterBank checkout outage on postgres primary plus Sentinel p95 latency alert.
- Recommended action: join incident response and fix checkout.

## Standalone Items

- email_02: Priority P3, Category marketing, Recommended action: review later.
- email_03: Priority P4, Category automated dependency update, Recommended action: defer.
- email_04: Priority P3, Category internal, Recommended action: complete enrollment later.
- email_05: Priority P1, Category customer, Recommended action: respond to NimbusMart about 1.8M renewal.
- email_06: Priority P4, Category calendar, Recommended action: no urgent action.
- email_07: Priority P3, Category internal, Recommended action: submit self assessment.
- email_08: Priority P1, Category security, Recommended action: complete Okta token rotation by May 3 17:00 UTC.
- email_09: Priority P4, Category newsletter, Recommended action: archive.
- email_10: Priority P2, Category release review, Recommended action: review billing export freeze.
- email_11: Priority P4, Category vendor promotional, Recommended action: archive.
- email_12: Priority P3, Category finance, Recommended action: review budget notes later.
"""
    workspace = tmp_dir / "task16_synth_smoke"
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "triage_report.md").write_text(report, encoding="utf-8")
    rubric = {
        "expected_incident_groups": [
            {
                "emails": ["email_01", "email_13"],
                "priority": "p0",
                "required_clues": ["asterbank", "checkout", "postgres primary", "p95 latency"],
            }
        ],
        "expected_priorities": {
            "email_01": "p0",
            "email_13": "p0",
            "email_05": "p1",
            "email_08": "p1",
            "email_10": "p2",
            "email_11": "p4",
        },
        "expected_bindings": {
            "email_01": {"required_any": ["asterbank", "checkout", "postgres primary", "outage"], "min_matches": 2},
            "email_05": {"required_any": ["nimbusmart", "1.8m renewal", "customer"], "min_matches": 2},
            "email_08": {"required_any": ["okta token rotation", "may 3 17:00 utc", "security"], "min_matches": 2},
            "email_10": {"required_any": ["billing export freeze", "release", "review"], "min_matches": 2},
            "email_11": {"required_any": ["promotional", "vendor"], "min_matches": 2},
            "email_13": {"required_any": ["sentinel", "p95 latency", "checkout", "alert"], "min_matches": 2},
        },
    }
    result = verify_task16_per_instance(workspace, rubric)
    if not result.passed:
        raise AssertionError(f"Expected per-instance verifier pass, got {result}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data/task16_prompts")
    args = parser.parse_args()

    task_path = REPO_ROOT / "pinchbench_tasks/task_16_email_triage.md"
    if not task_path.is_file():
        raise AssertionError(f"Missing canonical task file: {task_path}")

    _assert_task_data(args.data_dir)
    _assert_reward_smoke()
    with tempfile.TemporaryDirectory(prefix="task16_verifier_") as td:
        _assert_per_instance_verifier_smoke(Path(td))
    print("OK: task16 data and reward smoke test passed")


if __name__ == "__main__":
    main()
