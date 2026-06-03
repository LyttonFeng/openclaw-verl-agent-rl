"""Diagnostics module — analyze rollout / bench trajectories.

Public API:
    diagnose(...)  →  DiagnosticsResult
    register_plugin(...), resolve_plugin(...)
    render_markdown_report(...), dump_json(...)

Importing this package auto-registers all known task-family plugins.
"""

from .core import DiagnosticsResult, diagnose
from .protocol import TaskPlugin, register_plugin, resolve_plugin
from .reporters import dump_json, render_markdown_report

# Auto-register plugins
from . import plugins  # noqa: F401

__all__ = [
    "DiagnosticsResult",
    "diagnose",
    "TaskPlugin",
    "register_plugin",
    "resolve_plugin",
    "render_markdown_report",
    "dump_json",
]
