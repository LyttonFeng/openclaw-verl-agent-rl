#!/usr/bin/env python3
"""Patch installed Ray to make node startup wait configurable.

Ray 2.55 hardcodes:

    raylet_start_wait_time_s = 30

inside `ray/_private/node.py`, which is too short for our container. This patch
switches it to:

    raylet_start_wait_time_s = int(os.environ.get("RAY_raylet_start_wait_time_s", "180"))
"""

from __future__ import annotations

import sys
from pathlib import Path

NEEDLE = """            raylet_start_wait_time_s = 30
"""

REPLACEMENT = """            raylet_start_wait_time_s = int(
                os.environ.get("RAY_raylet_start_wait_time_s", "180")
            )
"""


def ray_private_file(name: str) -> Path:
    import ray._private

    return Path(ray._private.__file__).resolve().parent / name


def main() -> int:
    target = ray_private_file("node.py")
    backup = target.with_suffix(target.suffix + ".pinchbench.bak")
    text = target.read_text()
    if REPLACEMENT in text:
        print(f"[patch_ray_node_start_wait] already patched: {target}")
        return 0
    if NEEDLE not in text:
        print(f"[patch_ray_node_start_wait] needle not found: {target}", file=sys.stderr)
        return 1
    if not backup.exists():
        backup.write_text(text)
    target.write_text(text.replace(NEEDLE, REPLACEMENT, 1))
    print(f"[patch_ray_node_start_wait] patched: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
