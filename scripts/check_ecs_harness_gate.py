#!/usr/bin/env python3
"""Gate checks for ECS OpenClaw task16 RL training.

This script is intentionally stricter than check_env.py. It verifies that the
ECS harness is configured so terminal reward is trustworthy before launching RL.
It can also validate a diagnostics JSONL produced by a smoke or validation run.
"""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]


def fail(message: str) -> None:
    raise SystemExit(f"FAIL: {message}")


def ok(message: str) -> None:
    print(f"OK: {message}")


def _source(path: Path) -> str:
    if not path.is_file():
        fail(f"missing file: {path}")
    return path.read_text(encoding="utf-8")


def _parse_python(path: Path) -> ast.Module:
    try:
        return ast.parse(_source(path), filename=str(path))
    except SyntaxError as exc:
        fail(f"{path} is not valid Python: {exc}")


def _has_success_execution_result(tree: ast.Module) -> bool:
    for node in ast.walk(tree):
        if not isinstance(node, ast.Dict):
            continue
        pairs = list(zip(node.keys, node.values))
        for key, value in pairs:
            if not (
                isinstance(key, ast.Constant)
                and key.value == "status"
                and isinstance(value, ast.Constant)
            ):
                continue
            if value.value == "success":
                return True
    return False


def check_static_agent_loop() -> None:
    path = REPO_ROOT / "agent_loop/openclaw_agent_loop.py"
    text = _source(path)
    tree = _parse_python(path)

    if not _has_success_execution_result(tree):
        fail("agent loop does not set execution_result status to success")
    if '"status": "completed"' in text or "'status': 'completed'" in text:
        fail("agent loop still contains execution_result status=completed")
    ok("execution_result.status is success")

    required_snippets = [
        "OPENCLAW_HOME",
        "skipBootstrap",
        "'skills': []",
        "self._last_grading_status[\"sync_status\"] = \"ok\"",
        "judge_status[\"grade_call\"] = \"ok\"",
        "+data.apply_chat_template_kwargs.enable_thinking",
    ]
    missing = [s for s in required_snippets if s not in text and s not in _source(REPO_ROOT / "rl/train/run_reinforce_lora.sh")]
    if missing:
        fail(f"missing expected harness/training snippets: {missing}")
    ok("fresh OpenClaw home, skill disabling, sync status, judge status, and non-thinking config are present")

    rsync_fetch = re.search(
        r"ECS workspace fetch for grading|rsync workspace failed for",
        text,
    )
    if not rsync_fetch:
        fail("agent loop does not appear to fetch ECS workspace for grading")
    if "-p {self.oc_config.ssh_port}" not in text and "-p\", str(self.oc_config.ssh_port)" not in text:
        fail("ECS workspace rsync does not include OPENCLAW_PORT")
    ok("ECS workspace rsync includes OPENCLAW_PORT")


def check_env(strict_env: bool) -> None:
    persistent = os.environ.get("PINCHBENCH_ECS_PERSISTENT_AGENT", "").strip().lower()
    if persistent in {"1", "true", "yes", "on"}:
        fail("PINCHBENCH_ECS_PERSISTENT_AGENT is set; formal RL training must not reuse persistent agents")
    ok("persistent ECS agent is disabled")

    if os.environ.get("PINCHBENCH_DISABLE_DEFAULT_SKILLS", "").strip().lower() not in {"1", "true", "yes", "on"}:
        if strict_env:
            fail("PINCHBENCH_DISABLE_DEFAULT_SKILLS must be set for formal task16 RL")
        print("WARN: PINCHBENCH_DISABLE_DEFAULT_SKILLS is not set; set it before formal RL")
    else:
        ok("global/default skills disabled")

    for name in ("OPENCLAW_HOST", "OPENCLAW_USER", "OPENCLAW_PORT", "OPENCLAW_SSH_KEY"):
        value = os.environ.get(name, "").strip()
        if not value and strict_env:
            fail(f"{name} is required")
        if value:
            ok(f"{name} set")


def check_remote_openclaw() -> None:
    host = os.environ.get("OPENCLAW_HOST", "").strip()
    if not host:
        print("WARN: OPENCLAW_HOST not set; skipping remote OpenClaw check")
        return
    user = os.environ.get("OPENCLAW_USER", "root")
    port = os.environ.get("OPENCLAW_PORT", "22")
    key = os.environ.get("OPENCLAW_SSH_KEY", str(Path.home() / ".ssh/id_ed25519"))
    cmd = [
        "ssh",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        "-i",
        key,
        "-p",
        port,
        f"{user}@{host}",
        "command -v openclaw >/dev/null && openclaw --version",
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
    if result.returncode != 0:
        fail("remote OpenClaw check failed: " + (result.stderr.strip() or result.stdout.strip()))
    ok("remote OpenClaw reachable: " + result.stdout.strip())


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            fail(f"invalid JSONL at {path}:{lineno}: {exc}")
        if isinstance(obj, dict):
            rows.append(obj)
    return rows


def _judge_status(row: dict[str, Any]) -> dict[str, Any]:
    evidence = row.get("evidence")
    grading_status = evidence.get("grading_status") if isinstance(evidence, dict) else {}
    if not isinstance(grading_status, dict):
        return {}
    judge = grading_status.get("judge_status")
    return judge if isinstance(judge, dict) else {}


def check_diagnostics(path: Path, min_rows: int) -> None:
    if not path.is_file():
        fail(f"diagnostics JSONL not found: {path}")
    rows = _load_jsonl(path)
    if len(rows) < min_rows:
        fail(f"diagnostics has {len(rows)} rows, expected at least {min_rows}")
    ok(f"diagnostics rows >= {min_rows}: {len(rows)}")

    failures: list[str] = []
    report_count = 0
    for idx, row in enumerate(rows, start=1):
        evidence = row.get("evidence") if isinstance(row.get("evidence"), dict) else {}
        grading_status = evidence.get("grading_status") if isinstance(evidence.get("grading_status"), dict) else {}
        judge = _judge_status(row)
        if grading_status.get("sync_status") not in {"ok", "local"}:
            failures.append(f"row {idx}: sync_status={grading_status.get('sync_status')!r}")
        if judge.get("backend") != "api":
            failures.append(f"row {idx}: judge backend is not api: {judge.get('backend')!r}")
        if judge.get("preflight") != "ok":
            failures.append(f"row {idx}: judge preflight={judge.get('preflight')!r}")
        if judge.get("grade_call") != "ok":
            failures.append(f"row {idx}: judge grade_call={judge.get('grade_call')!r}")
        notes = str(grading_status.get("notes") or "").lower()
        if "skipped" in notes and ("judge" in notes or "execution failed" in notes or "completed" in notes):
            failures.append(f"row {idx}: suspicious skipped-grading notes={notes[:160]!r}")
        if evidence.get("bad_read_paths"):
            failures.append(f"row {idx}: bad_read_paths={evidence.get('bad_read_paths')!r}")
        if evidence.get("bad_report_paths"):
            failures.append(f"row {idx}: bad_report_paths={evidence.get('bad_report_paths')!r}")
        if evidence.get("workspace_seeded_files", 13) < 13:
            failures.append(f"row {idx}: workspace_seeded_files={evidence.get('workspace_seeded_files')!r}")
        if evidence.get("report_exists_local"):
            report_count += 1

    if failures:
        preview = "\n".join(failures[:20])
        fail(f"diagnostics gate failed:\n{preview}")
    ok("diagnostics judge/sync/path checks passed")
    print(f"INFO: report_exists rows: {report_count}/{len(rows)}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", type=Path, help="Rollout diagnostics JSONL from a smoke/eval run")
    parser.add_argument("--min-diagnostic-rows", type=int, default=5)
    parser.add_argument("--strict-env", action="store_true", help="Fail when formal training env vars are missing")
    parser.add_argument("--skip-remote", action="store_true", help="Skip SSH OpenClaw check")
    args = parser.parse_args()

    check_static_agent_loop()
    check_env(strict_env=args.strict_env)
    if not args.skip_remote:
        check_remote_openclaw()
    if args.diagnostics:
        check_diagnostics(args.diagnostics, args.min_diagnostic_rows)
    else:
        print("WARN: no diagnostics JSONL provided; run a task16 smoke/val and pass --diagnostics before RL")

    print("OK: ECS harness gate checks completed")


if __name__ == "__main__":
    main()
