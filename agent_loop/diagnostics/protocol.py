"""TaskPlugin protocol + plugin registry.

Each task family (meeting_analysis, task16_ecs, ...) registers a TaskPlugin
that supplies family-specific knowledge consumed by the diagnostics core:

    - which output file each task is expected to write
    - which input filenames count as "the transcript / source"
    - per-task minimum output length (optional)
    - escape-hatch custom checker (optional)

The diagnostics core does NOT re-invent grading. It consumes the existing
automated grading breakdown (from result.json or compute_score) and the PRM
turn scores. Plugins therefore deliberately do not carry keyword lists.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional


# Per-trajectory custom checker signature: (parsed_trajectory, workspace_path) -> extra failure tags
CustomChecker = Callable[[list, Optional[str]], list[str]]


@dataclass
class TaskPlugin:
    family_id: str
    expected_output_file: dict[str, str] = field(default_factory=dict)
    expected_input_files: set[str] = field(default_factory=set)
    min_output_chars: dict[str, int] = field(default_factory=dict)
    task_id_prefix_match: tuple[str, ...] = ()
    custom_checks: Optional[CustomChecker] = None


PLUGINS: dict[str, TaskPlugin] = {}


def register_plugin(plugin: TaskPlugin) -> None:
    if plugin.family_id in PLUGINS:
        raise ValueError(f"plugin already registered: {plugin.family_id}")
    PLUGINS[plugin.family_id] = plugin


def resolve_plugin(task_id: str) -> Optional[TaskPlugin]:
    for plugin in PLUGINS.values():
        if plugin.task_id_prefix_match and task_id.startswith(plugin.task_id_prefix_match):
            return plugin
        if task_id in plugin.expected_output_file:
            return plugin
    return None
