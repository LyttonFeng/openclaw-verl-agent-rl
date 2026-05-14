"""Inject RLOnlineRail into every jiuwenclaw DeepAgent at startup.

Loaded via PYTHONSTARTUP or `python -c "import inject_rl_online_rail"`
before `python -m jiuwenclaw.app ...`. Gated by USE_RL_ONLINE_RAIL=1 env.

Why this exists: colleague wrote RLOnlineRail + build_rl_online_rail_from_env
but nothing in jiuwenclaw production code path actually calls them — only
tests. We monkey-patch DeepAgent.configure so every agent built by jiuwenclaw
stack gets the rail without touching jiuwenclaw source.
"""

from __future__ import annotations

import logging
import os
import sys

logger = logging.getLogger("inject_rl_rail")
if not logger.handlers:
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(logging.Formatter("%(asctime)s [inject_rl_rail] %(levelname)s %(message)s"))
    logger.addHandler(h)
logger.setLevel(logging.INFO)


def _enabled() -> bool:
    return os.getenv("USE_RL_ONLINE_RAIL", "").strip().lower() in ("1", "true", "yes", "on")


def install() -> bool:
    if not _enabled():
        logger.info("USE_RL_ONLINE_RAIL not truthy, skipping injection")
        return False
    try:
        from openjiuwen.agent_evolving.agent_rl.online.rail import (
            build_rl_online_rail_from_env,
        )
        from openjiuwen.harness.deep_agent import DeepAgent
    except Exception as exc:
        logger.warning("import failed (%s), cannot inject rail", exc)
        return False

    if getattr(DeepAgent, "_rl_online_rail_patched", False):
        logger.info("already patched, skipping")
        return True

    original_configure = DeepAgent.configure

    def patched_configure(self, config, *args, **kwargs):
        result = original_configure(self, config, *args, **kwargs)
        try:
            rail = build_rl_online_rail_from_env()
            if rail is not None:
                self.add_rail(rail)
                logger.info(
                    "injected RLOnlineRail into DeepAgent id=%s gateway=%s",
                    getattr(getattr(self, "card", None), "id", "?"),
                    os.getenv("TRAJECTORY_GATEWAY_URL", "(default)"),
                )
        except Exception as exc:
            logger.warning("inject failed: %r", exc)
        return result

    DeepAgent.configure = patched_configure
    DeepAgent._rl_online_rail_patched = True
    logger.info("DeepAgent.configure patched, RLOnlineRail will auto-inject on agent creation")
    return True


install()
