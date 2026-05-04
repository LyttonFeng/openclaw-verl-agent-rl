#!/usr/bin/env python3
"""Patch installed veRL REINFORCE++ source to bypass masked_whiten.

Ray TaskRunner workers import veRL in fresh Python processes, so patching the
driver registry is not sufficient. This script edits the installed
``verl/trainer/ppo/core_algos.py`` source before Ray workers start.
"""

from __future__ import annotations

from pathlib import Path


OLD = """        advantages = verl_F.masked_whiten(returns, response_mask)
        advantages = advantages * response_mask
"""

NEW = """        # PinchBench patch: rewards are already task-normalized and broadcast
        # to assistant turn spans. Do not whiten again across all response
        # tokens; sparse/near-constant rewards make masked_whiten explode.
        advantages = returns * response_mask
"""


def main() -> None:
    import verl.trainer.ppo.core_algos as core_algos

    path = Path(core_algos.__file__).resolve()
    text = path.read_text("utf-8")

    if NEW in text:
        print(f"[patch_verl_core_algos_no_whiten] already patched: {path}")
        return

    if OLD not in text:
        raise SystemExit(
            "[patch_verl_core_algos_no_whiten] target snippet not found in "
            f"{path}; inspect veRL version before training"
        )

    backup = path.with_suffix(path.suffix + ".pinchbench.bak")
    if not backup.exists():
        backup.write_text(text, "utf-8")

    path.write_text(text.replace(OLD, NEW, 1), "utf-8")
    print(f"[patch_verl_core_algos_no_whiten] patched: {path}")


if __name__ == "__main__":
    main()
