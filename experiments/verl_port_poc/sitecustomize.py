# sitecustomize.py — auto-imported by Python at startup if found on sys.path.
# We rely on PYTHONPATH including this dir so every Python process (jiuwenclaw.app,
# app_agentserver, app_gateway subprocesses) picks it up.
#
# This is the right level because jiuwenclaw.app spawns app_agentserver via
# subprocess.Popen — wrapping just the main process misses where the
# DeepAdapter actually lives.
try:
    import inject_rl_online_rail  # noqa: F401  (side-effect: installs the patch)
except Exception:
    pass
