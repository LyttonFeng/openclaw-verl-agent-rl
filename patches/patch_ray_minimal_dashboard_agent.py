#!/usr/bin/env python3
"""Force Ray dashboard agent to start in minimal mode.

Ray 2.55 on our A100 pod can fail during raylet startup with:

  Timed out waiting for file .../metrics_agent_port_<node_id>

The dashboard agent writes this file very early in `--minimal` mode, but much
later in the full mode after loading many modules. This patch appends
`--minimal` to the generated dashboard agent command in the installed Ray
services module so raylet can observe the port file before its internal timeout.
"""

from __future__ import annotations

import sys
from pathlib import Path

NEEDLE = """    if ray._private.utils.get_dashboard_dependency_error() is not None:
        # If dependencies are not installed, it is the minimally packaged
        # ray. We should restrict the features within dashboard agent
        # that requires additional dependencies to be downloaded.
        dashboard_agent_command.append("--minimal")
"""

REPLACEMENT = """    if ray._private.utils.get_dashboard_dependency_error() is not None:
        # If dependencies are not installed, it is the minimally packaged
        # ray. We should restrict the features within dashboard agent
        # that requires additional dependencies to be downloaded.
        dashboard_agent_command.append("--minimal")
    elif "--minimal" not in dashboard_agent_command:
        dashboard_agent_command.append("--minimal")
"""


def ray_private_file(name: str) -> Path:
    import ray._private

    return Path(ray._private.__file__).resolve().parent / name


def main() -> int:
    target = ray_private_file("services.py")
    backup = target.with_suffix(target.suffix + ".pinchbench.bak")
    text = target.read_text()
    if REPLACEMENT in text:
        print(f"[patch_ray_minimal_dashboard_agent] already patched: {target}")
        return 0
    if NEEDLE not in text:
        print(f"[patch_ray_minimal_dashboard_agent] needle not found: {target}", file=sys.stderr)
        return 1
    if not backup.exists():
        backup.write_text(text)
    target.write_text(text.replace(NEEDLE, REPLACEMENT, 1))
    print(f"[patch_ray_minimal_dashboard_agent] patched: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
