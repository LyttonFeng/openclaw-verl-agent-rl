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


def _say(msg: str) -> None:
    """Bypass logging (jiuwenclaw reconfigures it and may drop our handler).
    Write straight to stderr fd so output survives all log re-setup."""
    try:
        sys.stderr.write(f"[inject_rl_rail] {msg}\n")
        sys.stderr.flush()
    except Exception:
        pass


def _do_patch() -> bool:
    """Perform the actual class patch. Called once jiuwenclaw module is loadable."""
    # Ensure jiuwenclaw repo is on sys.path (we may have been launched outside its cwd)
    repo = os.environ.get("JIUWENCLAW_REPO", "/root/jiuwen_work/jiuwenclaw")
    if repo and repo not in sys.path and os.path.isdir(repo):
        sys.path.insert(0, repo)
    try:
        from openjiuwen.agent_evolving.agent_rl.online.rail import (
            build_rl_online_rail_from_env,
        )
        from jiuwenclaw.server.runtime.agent_adapter.interface_deep import (
            JiuWenClawDeepAdapter,
        )
    except Exception as exc:
        _say(f"import failed ({exc!r})")
        return False

    if getattr(JiuWenClawDeepAdapter, "_rl_online_rail_patched", False):
        _say("already patched")
        return True

    original_build = JiuWenClawDeepAdapter._build_agent_rails

    def patched_build_agent_rails(self, *args, **kwargs):
        rails_list = original_build(self, *args, **kwargs)
        try:
            rl_rail = build_rl_online_rail_from_env()
            if rl_rail is not None:
                rails_list.append(rl_rail)
                self._rl_online_rail = rl_rail
                _say(
                    f"appended RLOnlineRail to rails (count={len(rails_list)}, "
                    f"gateway={os.getenv('TRAJECTORY_GATEWAY_URL', '(default)')})"
                )
        except Exception as exc:
            _say(f"RLOnlineRail append failed: {exc!r}")
        return rails_list

    JiuWenClawDeepAdapter._build_agent_rails = patched_build_agent_rails
    JiuWenClawDeepAdapter._rl_online_rail_patched = True
    _say("JiuWenClawDeepAdapter._build_agent_rails patched")
    return True


def install() -> bool:
    """Install patch via sys.meta_path post-import hook.

    Jiuwenclaw's own bootstrap imports interface_deep at some point; we wrap
    a Finder that detects when that import completes and immediately calls
    _do_patch. This avoids the racing/threading mess of a polling daemon.
    """
    if not _enabled():
        _say("USE_RL_ONLINE_RAIL not truthy, skipping injection")
        return False

    target_mod = "jiuwenclaw.server.runtime.agent_adapter.interface_deep"

    if target_mod in sys.modules:
        # Already imported — patch now
        if _do_patch():
            _say("patched immediately (module already imported)")
        return True

    import importlib.abc
    import importlib.util

    class _PostImportFinder(importlib.abc.MetaPathFinder):
        _fired = False

        def find_spec(self, fullname, path=None, target=None):
            if fullname == target_mod and not _PostImportFinder._fired:
                # Find the real spec via the rest of meta_path, wrap its loader
                # to call _do_patch after exec_module completes.
                for finder in sys.meta_path:
                    if finder is self:
                        continue
                    if not hasattr(finder, "find_spec"):
                        continue
                    spec = finder.find_spec(fullname, path, target)
                    if spec is None:
                        continue
                    original_loader = spec.loader
                    if original_loader is None:
                        continue

                    class _WrappedLoader(importlib.abc.Loader):
                        def create_module(self, spec):
                            return original_loader.create_module(spec)
                        def exec_module(self, module):
                            original_loader.exec_module(module)
                            try:
                                _do_patch()
                                _PostImportFinder._fired = True
                            except Exception as exc:
                                _say(f"post-import patch failed: {exc!r}")

                    spec.loader = _WrappedLoader()
                    _say(f"wrapped loader for {fullname}")
                    return spec
            return None

    sys.meta_path.insert(0, _PostImportFinder())
    _say(f"installed meta_path hook for {target_mod}")
    return True


install()
