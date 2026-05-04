#!/usr/bin/env python3
"""Check generated task16 parquet files."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    args = parser.parse_args()

    train_path = args.data_dir / "train.parquet"
    train_small_path = args.data_dir / "train_small.parquet"
    train_tiny_path = args.data_dir / "train_tiny.parquet"
    train_synth_path = args.data_dir / "train_synth20.parquet"
    train_stage2_path = args.data_dir / "train_stage2_balanced.parquet"
    val_synth_path = args.data_dir / "val_synth5.parquet"
    val_path = args.data_dir / "val.parquet"
    if not train_path.is_file():
        raise SystemExit(f"Missing {train_path}")
    if not train_small_path.is_file():
        raise SystemExit(f"Missing {train_small_path}")
    if not train_tiny_path.is_file():
        raise SystemExit(f"Missing {train_tiny_path}")
    if not train_synth_path.is_file():
        raise SystemExit(f"Missing {train_synth_path}")
    if not train_stage2_path.is_file():
        raise SystemExit(f"Missing {train_stage2_path}")
    if not val_synth_path.is_file():
        raise SystemExit(f"Missing {val_synth_path}")
    if not val_path.is_file():
        raise SystemExit(f"Missing {val_path}")

    train = pd.read_parquet(train_path)
    train_small = pd.read_parquet(train_small_path)
    train_tiny = pd.read_parquet(train_tiny_path)
    train_synth = pd.read_parquet(train_synth_path)
    train_stage2 = pd.read_parquet(train_stage2_path)
    val_synth = pd.read_parquet(val_synth_path)
    val = pd.read_parquet(val_path)
    expected = {
        "train": 91,
        "train_small": 32,
        "train_tiny": 16,
        "train_synth20": 20,
        "train_stage2_balanced": 32,
        "val_synth5": 5,
        "val": 11,
    }
    actual = {
        "train": len(train),
        "train_small": len(train_small),
        "train_tiny": len(train_tiny),
        "train_synth20": len(train_synth),
        "train_stage2_balanced": len(train_stage2),
        "val_synth5": len(val_synth),
        "val": len(val),
    }
    if actual != expected:
        raise SystemExit(f"Unexpected row counts: {actual}, expected {expected}")

    for name, frame in (
        ("train", train),
        ("train_small", train_small),
        ("train_tiny", train_tiny),
        ("train_synth20", train_synth),
        ("train_stage2_balanced", train_stage2),
        ("val_synth5", val_synth),
        ("val", val),
    ):
        task_ids = {
            row.get("task_id")
            for row in frame["extra_info"]
            if isinstance(row, dict)
        }
        if task_ids != {"task_16_email_triage"}:
            raise SystemExit(f"{name} has non-task16 ids: {sorted(task_ids)}")

    synth_flags = [
        row.get("synthetic_instance")
        for row in train_synth["extra_info"]
        if isinstance(row, dict)
    ]
    if synth_flags != [True] * 20:
        raise SystemExit("train_synth20 rows must all be marked synthetic_instance=True")

    stage2_synth_count = sum(
        1
        for row in train_stage2["extra_info"]
        if isinstance(row, dict) and row.get("synthetic_instance") is True
    )
    if stage2_synth_count != 20:
        raise SystemExit(
            f"train_stage2_balanced must include 20 synthetic rows, got {stage2_synth_count}"
        )

    val_synth_count = sum(
        1
        for row in val_synth["extra_info"]
        if isinstance(row, dict) and row.get("synthetic_instance") is True
    )
    if val_synth_count != 5:
        raise SystemExit(f"val_synth5 must include 5 synthetic rows, got {val_synth_count}")

    print(
        "OK: "
        f"train.parquet={actual['train']} rows, "
        f"train_small.parquet={actual['train_small']} rows, "
        f"train_tiny.parquet={actual['train_tiny']} rows, "
        f"train_synth20.parquet={actual['train_synth20']} rows, "
        f"train_stage2_balanced.parquet={actual['train_stage2_balanced']} rows, "
        f"val_synth5.parquet={actual['val_synth5']} rows, "
        f"val.parquet={actual['val']} rows"
    )


if __name__ == "__main__":
    main()
