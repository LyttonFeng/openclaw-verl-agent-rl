#!/usr/bin/env python3
"""Wrapper: import RLOnlineRail patch, then exec jiuwenclaw.app.

PYTHONSTARTUP only fires for interactive python; this wrapper makes the
monkey-patch run before `python -m jiuwenclaw.app`. Same CLI semantics —
forwards argv (e.g. `--dotenv FILE`) to jiuwenclaw.app.
"""

from __future__ import annotations

import runpy
import sys

# Apply the patch (no-op if USE_RL_ONLINE_RAIL is not set)
import inject_rl_online_rail  # noqa: F401  (side-effect import)

# Now run jiuwenclaw.app exactly as `python -m jiuwenclaw.app`
# argv[0] is the wrapper path; jiuwenclaw.app expects its own argv shape.
# We keep sys.argv as-is; argparse in jiuwenclaw consumes from argv[1:].
runpy.run_module("jiuwenclaw.app", run_name="__main__")
