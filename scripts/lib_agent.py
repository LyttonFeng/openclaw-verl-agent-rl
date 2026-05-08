"""
OpenClaw agent execution helpers for PinchBench.
"""

from __future__ import annotations

import json
import logging
import os
import platform
import signal
import shlex
import stat
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib import error, request

from lib_tasks import Task


logger = logging.getLogger(__name__)

USE_SHELL = platform.system() == "Windows"


def _openclaw_catalog_model_id(model_id: str) -> str:
    """Map HuggingFace-style ids (e.g. Qwen/Qwen3-4B) to a single catalog name for OpenClaw.

    OpenClaw treats ``/`` in the model field as ``provider/model``. A defaultModel of
    ``Qwen/Qwen3-4B`` becomes ``qwen/Qwen3-4B`` and does not match the ``custom`` provider.
    vLLM ``--served-model-name`` also registers the basename (Qwen3-4B).
    """
    s = model_id.strip()
    if "/" in s:
        return s.rsplit("/", 1)[-1]
    return s


def _openclaw_provider_model_pair(model_id: str, *, base_url: str | None = None) -> tuple[str, str]:
    """Resolve an OpenClaw provider/model pair for bench agent config.

    Local vLLM/custom endpoints must stay under ``custom``. Cloud Qwen models
    that already exist in the copied OpenClaw catalog must stay under ``qwen``;
    forcing them to ``custom`` produces "Unknown model: custom/qwen3.6-plus".
    """
    mid = _openclaw_catalog_model_id(model_id)
    if base_url:
        return "custom", mid
    if "/" in model_id:
        provider, model = model_id.split("/", 1)
        return provider, model
    if mid in {"qwen-plus", "qwen3.6-plus"}:
        return "dashscope", mid
    if mid.startswith("qwen"):
        return "qwen", mid
    return "custom", mid


def _patch_openclaw_agent_disable_model_fallbacks(agent_id: str, model_id: str) -> None:
    """OpenClaw merges global `agents.defaults.model` fallbacks when the agent's model is a plain string.

    On fallback to a second provider, `resolveFallbackRetryPrompt` replaces the real user message with
    \"Continue where you left off...\" (see openclaw dist/agent-command attempt-execution), which
    breaks PinchBench. Forcing `fallbacks: []` on this agent disables that chain.
    """
    path = _get_openclaw_home() / "openclaw.json"
    if not path.is_file():
        logger.warning("Cannot patch OpenClaw config (missing %s); model fallbacks may still apply.", path)
        return
    try:
        raw = path.read_text("utf-8-sig")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Cannot read OpenClaw config %s: %s", path, exc)
        return
    agents = data.get("agents")
    if not isinstance(agents, dict):
        return
    lst = agents.get("list")
    if not isinstance(lst, list):
        return
    normalized = agent_id.replace(":", "-").lower()
    updated = False
    for entry in lst:
        if not isinstance(entry, dict):
            continue
        eid = entry.get("id")
        if not isinstance(eid, str):
            continue
        if eid != agent_id and eid.lower() != normalized:
            continue
        provider, mid = _openclaw_provider_model_pair(
            model_id,
            base_url=os.environ.get("PINCHBENCH_MODEL_BASE_URL"),
        )
        entry["model"] = {"primary": f"{provider}/{mid}", "fallbacks": []}
        updated = True
        break
    if not updated:
        logger.warning(
            "Bench agent %s not found in openclaw.json agents.list; could not disable model fallbacks.",
            agent_id,
        )
        return
    tmp = path.with_suffix(".json.tmp")
    try:
        tmp.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", "utf-8")
        os.replace(tmp, path)
        logger.info(
            "Patched OpenClaw config: agent %s model fallbacks disabled (PinchBench, primary=%s/%s).",
            agent_id,
            provider,
            mid,
        )
    except OSError as exc:
        logger.warning("Failed to write OpenClaw config %s: %s", path, exc)
        try:
            tmp.unlink(missing_ok=True)
        except OSError:
            pass


class ModelValidationError(Exception):
    """Raised when a model ID is invalid or inaccessible."""

    pass


MAX_OPENCLAW_MESSAGE_CHARS = int(os.environ.get("PINCHBENCH_MAX_MSG_CHARS", "8000"))
JUDGE_MAX_MSG_CHARS = int(os.environ.get("PINCHBENCH_JUDGE_MAX_MSG_CHARS", "3000"))
OPENCLAW_CONTEXT_WINDOW = int(os.environ.get("PINCHBENCH_OPENCLAW_CONTEXT_WINDOW", "32768"))
OPENCLAW_MAX_TOKENS = int(os.environ.get("PINCHBENCH_OPENCLAW_MAX_TOKENS", "8192"))
OPENCLAW_MODEL_TEMPERATURE = os.environ.get("PINCHBENCH_MODEL_TEMPERATURE")
OPENCLAW_MODEL_TOP_P = os.environ.get("PINCHBENCH_MODEL_TOP_P")
OPENCLAW_MODEL_TOP_K = os.environ.get("PINCHBENCH_MODEL_TOP_K")
PINCHBENCH_TASK_PROMPT_HINT = os.environ.get("PINCHBENCH_TASK_PROMPT_HINT", "")
PINCHBENCH_TASK_PROMPT_HINT_TASKS = {
    s.strip() for s in os.environ.get("PINCHBENCH_TASK_PROMPT_HINT_TASKS", "").split(",") if s.strip()
}
PINCHBENCH_SKIP_SKILLS = {
    s.strip()
    for s in os.environ.get("PINCHBENCH_SKIP_SKILLS", "").split(",")
    if s.strip()
}


def _coerce_subprocess_output(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _run_with_process_group(
    args: List[str],
    *,
    timeout: float,
    cwd: Optional[str] = None,
    shell: bool = False,
) -> subprocess.CompletedProcess[str]:
    """Run a command with a hard timeout that tears down the whole process group.

    OpenClaw is a Node-based CLI and may spawn children.  `subprocess.run(...,
    timeout=...)` kills the immediate child but can leave descendants holding
    session/workspace locks.  PinchBench semantics are unchanged here: a timed
    out task is still a timeout, but the runner can continue to the next run.
    """
    popen_kwargs: Dict[str, Any] = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "text": True,
        "cwd": cwd,
        "shell": shell,
    }
    if not USE_SHELL:
        popen_kwargs["start_new_session"] = True

    proc = subprocess.Popen(args, **popen_kwargs)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        return subprocess.CompletedProcess(args, proc.returncode, stdout, stderr)
    except subprocess.TimeoutExpired as exc:
        if not USE_SHELL:
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                stdout, stderr = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                stdout, stderr = proc.communicate()
        else:
            proc.kill()
            stdout, stderr = proc.communicate()
        exc.stdout = stdout
        exc.stderr = stderr
        raise exc


def _compose_task_prompt(task: Any) -> str:
    prompt = task.prompt
    if not PINCHBENCH_TASK_PROMPT_HINT:
        return prompt
    task_id = getattr(task, "id", "") or ""
    if PINCHBENCH_TASK_PROMPT_HINT_TASKS and task_id not in PINCHBENCH_TASK_PROMPT_HINT_TASKS:
        return prompt
    return f"{prompt.rstrip()}\n\n## Additional Execution Hint\n{PINCHBENCH_TASK_PROMPT_HINT.strip()}\n"


def slugify_model(model_id: str) -> str:
    return model_id.replace("/", "-").replace(".", "-").lower()


def validate_openrouter_model(model_id: str, timeout_seconds: float = 10.0) -> bool:
    """
    Validate that a model ID exists on OpenRouter.

    Args:
        model_id: Model ID (with or without openrouter/ prefix)
        timeout_seconds: HTTP request timeout

    Returns:
        True if model is valid and accessible

    Raises:
        ModelValidationError: If model doesn't exist or validation fails
    """
    # Strip openrouter/ prefix if present
    bare_model_id = model_id
    if bare_model_id.startswith("openrouter/"):
        bare_model_id = bare_model_id[len("openrouter/") :]

    # Skip validation for non-OpenRouter models
    if "/" not in bare_model_id:
        logger.info("Skipping model validation for non-OpenRouter model: %s", model_id)
        return True

    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        logger.warning("OPENROUTER_API_KEY not set, skipping model validation")
        return True

    logger.info("🔍 Validating model: %s", bare_model_id)

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://pinchbench.com",
        "X-Title": "PinchBench",
    }

    # First, try the specific model endpoint (fast path for valid models)
    encoded_model_id = bare_model_id.replace("/", "%2F")
    specific_endpoint = f"https://openrouter.ai/api/v1/models/{encoded_model_id}"
    req = request.Request(specific_endpoint, headers=headers, method="GET")
    try:
        with request.urlopen(req, timeout=timeout_seconds) as resp:
            # Model exists - validation passed
            logger.info("✅ Model validated: %s", bare_model_id)
            return True
    except error.HTTPError as exc:
        if exc.code == 404:
            # Model not found - fall through to fetch full catalog for suggestions
            pass
        else:
            logger.warning("OpenRouter API error during validation: %s", exc)
            return True
    except error.URLError as exc:
        logger.warning("Network error during model validation: %s", exc)
        return True

    # Model not found - fetch full catalog for "did you mean" suggestions
    catalog_endpoint = "https://openrouter.ai/api/v1/models"
    req = request.Request(catalog_endpoint, headers=headers, method="GET")
    try:
        with request.urlopen(req, timeout=timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        logger.warning("OpenRouter API error fetching model catalog: %s", exc)
        raise ModelValidationError(f"Model '{bare_model_id}' not found on OpenRouter.")
    except error.URLError as exc:
        logger.warning("Network error fetching model catalog: %s", exc)
        raise ModelValidationError(f"Model '{bare_model_id}' not found on OpenRouter.")
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse OpenRouter response: %s", exc)
        raise ModelValidationError(f"Model '{bare_model_id}' not found on OpenRouter.")

    models = data.get("data", [])
    model_ids = {
        mid
        for m in models
        if isinstance(m, dict)
        for mid in [m.get("id")]
        if isinstance(mid, str) and mid
    }

    # Some OpenRouter model detail lookups intermittently return 404 for valid
    # IDs. Treat an exact catalog hit as authoritative to avoid false negatives.
    if bare_model_id in model_ids:
        logger.info("✅ Model validated via catalog fallback: %s", bare_model_id)
        return True

    # Check for close matches (typos)
    close_matches = []
    bare_lower = bare_model_id.lower()
    for mid in model_ids:
        mid_lower = mid.lower()
        if mid_lower == bare_lower:
            continue
        if bare_lower in mid_lower or mid_lower in bare_lower:
            close_matches.append(mid)

    error_msg = f"Model '{bare_model_id}' not found on OpenRouter."
    if close_matches:
        close_matches_str = ", ".join(sorted(close_matches)[:5])
        error_msg += f" Did you mean: {close_matches_str}?"
    else:
        # Try to suggest based on provider
        provider = bare_model_id.split("/")[0] if "/" in bare_model_id else None
        if provider:
            provider_models = [m for m in model_ids if m.startswith(f"{provider}/")]
            if provider_models:
                error_msg += (
                    f" Available {provider} models: {', '.join(sorted(provider_models)[:5])}"
                )

    raise ModelValidationError(error_msg)


def _get_agent_workspace(agent_id: str) -> Path | None:
    """Get the workspace path for an agent from OpenClaw config."""
    try:
        list_result = subprocess.run(
            ["openclaw", "agents", "list"],
            capture_output=True,
            text=True,
            check=False,
            shell=USE_SHELL,
        )
        if list_result.returncode != 0:
            return None

        # Parse the agent list output to find workspace
        # OpenClaw normalizes colons to dashes and lowercases agent names
        normalized_id = agent_id.replace(":", "-").lower()
        lines = list_result.stdout.split("\n")
        found_agent = False
        for line in lines:
            stripped = line.strip()
            if stripped.startswith(f"- {agent_id}") or stripped.startswith(f"- {normalized_id}"):
                found_agent = True
            elif found_agent and "Workspace:" in line:
                workspace_str = line.split("Workspace:")[1].strip()
                # Expand ~ if present
                if workspace_str.startswith("~/"):
                    workspace_str = str(Path.home() / workspace_str[2:])
                return Path(workspace_str)
            elif found_agent and line.strip().startswith("-"):
                # Found next agent, stop looking
                break
        return None
    except Exception as exc:
        logger.warning("Failed to get agent workspace: %s", exc)
        return None


def ensure_agent_exists(
    agent_id: str,
    model_id: str,
    workspace_dir: Path,
    *,
    base_url: str | None = None,
    api_key: str | None = None,
) -> bool:
    """Ensure the OpenClaw agent exists with the correct workspace.

    If the agent already exists but points to a different workspace, it is
    deleted and recreated so that the new workspace takes effect.

    When *base_url* is provided, a custom OpenAI-compatible provider is
    configured in the agent's ``models.json`` instead of relying on
    OpenRouter.  *api_key* defaults to ``${OPENAI_API_KEY}`` (resolved by
    OpenClaw at runtime) if not given.

    Returns True if the agent was (re)created.
    """
    def _configure_bench_agent_files() -> None:
        # Configure models.json for the bench agent. In remote RL runs this
        # also works when the pod has no local openclaw CLI installed; the
        # generated files are later rsynced to the ECS OpenClaw home.
        bench_agent_dir = _get_agent_store_dir(agent_id) / "agent"
        bench_agent_dir.mkdir(parents=True, exist_ok=True)
        bench_models = bench_agent_dir / "models.json"
        main_models = _get_openclaw_home() / "agents" / "main" / "agent" / "models.json"

        if base_url:
            os.environ["PINCHBENCH_MODEL_BASE_URL"] = base_url
            key_ref = api_key if api_key else "dummy"
            mid = _openclaw_catalog_model_id(model_id)
            reasoning_enabled = os.environ.get("OPENCLAW_MODEL_REASONING", "").lower() in {
                "1",
                "true",
                "yes",
                "on",
            }
            custom_provider: dict[str, Any] = {
                "baseUrl": base_url,
                # pi-coding-agent (OpenClaw's embedded registry) requires a non-empty apiKey whenever
                # custom models are defined; omitting it invalidates models.json and yields Unknown model.
                # vLLM accepts Bearer dummy; OpenClaw resolves this literal key for HTTP (see model-auth.ts).
                "apiKey": key_ref,
                "api": "openai-completions",
                "models": [
                    {
                        "id": mid,
                        "name": mid,
                        "reasoning": reasoning_enabled,
                        "input": ["text"],
                        "contextWindow": OPENCLAW_CONTEXT_WINDOW,
                        "maxTokens": OPENCLAW_MAX_TOKENS,
                        **(
                            {"temperature": float(OPENCLAW_MODEL_TEMPERATURE)}
                            if OPENCLAW_MODEL_TEMPERATURE is not None
                            else {}
                        ),
                        **(
                            {"topP": float(OPENCLAW_MODEL_TOP_P)}
                            if OPENCLAW_MODEL_TOP_P is not None
                            else {}
                        ),
                        **(
                            {"topK": int(OPENCLAW_MODEL_TOP_K)}
                            if OPENCLAW_MODEL_TOP_K is not None
                            else {}
                        ),
                    }
                ],
            }
            data: dict[str, Any] = {
                "providers": {"custom": custom_provider},
                "defaultProvider": "custom",
                "defaultModel": mid,
            }
            bench_models.write_text(json.dumps(data, indent=2, ensure_ascii=False), "utf-8")
            _repair_bench_models_json_defaults(agent_id, model_id)
            logger.info(
                "Configured custom provider (%s) with model %s for agent %s",
                base_url,
                model_id,
                agent_id,
            )
        elif main_models.exists():
            os.environ.pop("PINCHBENCH_MODEL_BASE_URL", None)
            # Standard OpenRouter flow — copy main's models.json and set defaults
            import shutil as _shutil

            _shutil.copy2(main_models, bench_models)
            provider_name, model_name = _openclaw_provider_model_pair(model_id)
            try:
                raw = bench_models.read_text("utf-8-sig")
                data = json.loads(raw)
                providers = data.get("providers")
                if isinstance(providers, dict) and provider_name in providers:
                    data["defaultProvider"] = provider_name
                    data["defaultModel"] = model_name
                    bench_models.write_text(
                        json.dumps(data, indent=2, ensure_ascii=False), "utf-8"
                    )
                    logger.info(
                        "Set bench agent default model to %s / %s", provider_name, model_name
                    )
            except Exception as exc:
                logger.warning("Failed to set default model in bench models.json: %s", exc)
            logger.info("Copied main agent models.json to bench agent %s", agent_id)

        # Delete sessions.json so OpenClaw picks up the new defaultProvider/defaultModel
        # instead of reusing a cached session entry that still points to an old model.
        bench_sessions_dir = _get_agent_store_dir(agent_id) / "sessions"
        sessions_store = bench_sessions_dir / "sessions.json"
        if sessions_store.exists():
            try:
                sessions_store.unlink()
                logger.info("Deleted stale sessions.json for bench agent %s", agent_id)
            except OSError as exc:
                logger.warning("Failed to delete sessions.json: %s", exc)

        _patch_openclaw_agent_disable_model_fallbacks(agent_id, model_id)
        if base_url:
            _ensure_bench_auth_custom_profile(agent_id)

    workspace_dir.mkdir(parents=True, exist_ok=True)

    if _remote_openclaw_enabled() and base_url:
        logger.info(
            "Remote OpenClaw runtime enabled; preparing bench agent files for ECS sync"
        )
        _configure_bench_agent_files()
        return True

    try:
        list_result = subprocess.run(
            ["openclaw", "agents", "list"],
            capture_output=True,
            text=True,
            check=False,
            shell=USE_SHELL,
            timeout=30,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Timed out listing local OpenClaw agents")
        return False
    except FileNotFoundError:
        if _remote_openclaw_enabled() and base_url:
            logger.warning(
                "openclaw CLI not found locally; creating bench agent files for remote ECS sync"
            )
            _configure_bench_agent_files()
            return True
        logger.error("openclaw CLI not found while listing agents")
        return False

    if list_result.returncode == 0:
        # Check for exact agent ID match — avoid substring false positives
        # (e.g. "bench-foo-4" matching "bench-foo-4-5" in the output).
        # Output format is "- <agent_id>" or "- <agent_id> (default)" per line.
        # OpenClaw normalizes colons to dashes in directory/display names, so
        # also check the normalized form.
        existing_agents = set()
        for line in list_result.stdout.splitlines():
            line = line.strip()
            if line.startswith("- "):
                # Extract agent name: "- bench-foo-4-5" or "- main (default)"
                name_part = line[2:].split()[0] if line[2:].strip() else ""
                if name_part:
                    existing_agents.add(name_part.lower())
        normalized_id = agent_id.replace(":", "-").lower()
        if agent_id.lower() in existing_agents or normalized_id in existing_agents:
            # Agent exists — check if workspace matches
            current_workspace = _get_agent_workspace(agent_id)
            if (
                current_workspace is not None
                and current_workspace.resolve() == workspace_dir.resolve()
            ):
                logger.info("Agent %s already exists with correct workspace", agent_id)
                return False
            # Workspace is stale or unknown — delete and recreate
            delete_name = normalized_id if normalized_id in existing_agents else agent_id
            logger.info(
                "Agent %s exists with stale workspace (%s != %s), recreating",
                agent_id,
                current_workspace,
                workspace_dir,
            )
            subprocess.run(
                ["openclaw", "agents", "delete", delete_name, "--force"],
                capture_output=True,
                text=True,
                check=False,
                shell=USE_SHELL,
                timeout=30,
            )

    logger.info("Creating OpenClaw agent %s", agent_id)
    try:
        create_result = subprocess.run(
            [
                "openclaw",
                "agents",
                "add",
                agent_id,
                "--model",
                model_id,
                "--workspace",
                str(workspace_dir),
                "--non-interactive",
            ],
            capture_output=True,
            text=True,
            check=False,
            shell=USE_SHELL,
            timeout=60,
        )
    except subprocess.TimeoutExpired:
        logger.warning("Timed out creating local OpenClaw agent %s", agent_id)
        return False
    except FileNotFoundError:
        if _remote_openclaw_enabled() and base_url:
            logger.warning(
                "openclaw CLI not found locally while creating agent; using remote ECS sync files"
            )
            _configure_bench_agent_files()
            return True
        logger.error("openclaw CLI not found while creating agent")
        return False

    if create_result.returncode != 0:
        logger.warning(
            "Agent creation returned %s: %s", create_result.returncode, create_result.stderr
        )

    _configure_bench_agent_files()
    return True


def _ensure_bench_auth_custom_profile(agent_id: str) -> None:
    """OpenClaw resolves API keys for the ``custom`` provider via auth-profiles.json under the agent dir.

    models.json ``apiKey`` alone is not enough for the embedded agent runtime; without a profile, requests fail with
    ``No API key found for provider \"custom\"`` even when vLLM accepts ``Bearer dummy``.
    """
    bench_models = _get_agent_store_dir(agent_id) / "agent" / "models.json"
    if not bench_models.is_file():
        return
    try:
        data = json.loads(bench_models.read_text("utf-8-sig"))
    except (OSError, json.JSONDecodeError):
        return
    providers = data.get("providers")
    if not isinstance(providers, dict):
        return
    custom = providers.get("custom")
    if not isinstance(custom, dict):
        return
    key = str(custom.get("apiKey", "")).strip() or "dummy"
    auth_path = _get_agent_store_dir(agent_id) / "agent" / "auth-profiles.json"
    try:
        if auth_path.is_file():
            store = json.loads(auth_path.read_text("utf-8-sig"))
        else:
            store = {"version": 1, "profiles": {}}
    except (OSError, json.JSONDecodeError):
        store = {"version": 1, "profiles": {}}
    profiles = store.setdefault("profiles", {})
    profile_id = "pinchbench-custom"
    want = {"type": "api_key", "provider": "custom", "key": key}
    if profiles.get(profile_id) == want:
        return
    profiles[profile_id] = want
    order = store.setdefault("order", {})
    order["custom"] = [profile_id]
    try:
        auth_path.write_text(json.dumps(store, indent=2, ensure_ascii=False) + "\n", "utf-8")
        logger.info(
            "Wrote auth-profiles.json for agent %s (custom / %s)",
            agent_id,
            profile_id,
        )
    except OSError as exc:
        logger.warning("Failed to write auth-profiles.json for %s: %s", agent_id, exc)


def _repair_bench_models_json_defaults(agent_id: str, model_id: str) -> None:
    """OpenClaw may merge global providers into the bench agent's models.json and drop defaultProvider/defaultModel.

    Without those keys, the runtime fails model resolution, so no transcript jsonl is written.
    Catalog ids must use :func:`_openclaw_catalog_model_id` (no ``/``) so ``custom`` is not confused with ``qwen``.
    """
    bench_models = _get_agent_store_dir(agent_id) / "agent" / "models.json"
    if not bench_models.is_file():
        return
    try:
        raw = bench_models.read_text("utf-8-sig")
        data = json.loads(raw)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read models.json for repair (%s): %s", bench_models, exc)
        return
    providers = data.get("providers")
    if not isinstance(providers, dict):
        return
    custom = providers.get("custom")
    if not isinstance(custom, dict) or not str(custom.get("baseUrl", "")).strip():
        return
    mid = _openclaw_catalog_model_id(model_id)
    changed = False
    if not str(custom.get("apiKey", "")).strip():
        custom["apiKey"] = "dummy"
        changed = True
    if data.get("defaultProvider") != "custom":
        data["defaultProvider"] = "custom"
        changed = True
    if data.get("defaultModel") != mid:
        data["defaultModel"] = mid
        changed = True
    models_list = custom.get("models")
    if isinstance(models_list, list) and models_list and isinstance(models_list[0], dict):
        m0 = models_list[0]
        if m0.get("id") != mid or m0.get("name") != mid:
            m0["id"] = mid
            m0["name"] = mid
            changed = True
    if changed:
        try:
            bench_models.write_text(json.dumps(data, indent=2, ensure_ascii=False), "utf-8")
            logger.info(
                "Repaired models.json defaultProvider/defaultModel for agent %s → custom / %s",
                agent_id,
                mid,
            )
        except OSError as exc:
            logger.warning("Failed to write repaired models.json: %s", exc)
    _ensure_bench_auth_custom_profile(agent_id)


def cleanup_agent_sessions(agent_id: str) -> None:
    """Remove stored session transcripts for an agent to avoid unbounded growth."""
    agent_dir = _get_agent_store_dir(agent_id)
    sessions_dir = agent_dir / "sessions"
    if not sessions_dir.exists():
        return
    removed = 0
    for pattern in ("*.jsonl", "*.jsonl.lock", "*.ndjson"):
        for path in sessions_dir.rglob(pattern):
            try:
                path.unlink()
                removed += 1
            except OSError as exc:
                logger.warning("Failed to remove session file %s: %s", path, exc)
    sessions_store = sessions_dir / "sessions.json"
    if sessions_store.exists():
        try:
            sessions_store.unlink()
        except OSError as exc:
            logger.warning("Failed to remove session store %s: %s", sessions_store, exc)
    if removed:
        logger.info("Removed %s old OpenClaw session transcripts for %s", removed, agent_id)


def _pinchbench_agent_workspace_for_run(run_id: str) -> Path | None:
    """Resolve ``/tmp/pinchbench/<run>/agent_workspace`` from benchmark ``run_id``.

    ``execute_openclaw_task`` receives ``run_id`` like ``\"0029-1\"``; :func:`ensure_agent_exists`
    pins the OpenClaw agent to ``/tmp/pinchbench/0029/agent_workspace``. Grading must use that same
    path. Parsing ``openclaw agents list`` can return a *stale* workspace from an older run, so
    automated graders then see no files and score 0 despite a valid transcript.
    """
    if not run_id:
        return None
    head = str(run_id).split("-", 1)[0].strip()
    if not head.isdigit():
        return None
    run_root = Path(os.environ.get("PINCHBENCH_RUN_ROOT", "/tmp/pinchbench"))
    return run_root / head / "agent_workspace"


def _benchmark_workspace_clean_mode(task: Task) -> bool:
    raw = os.environ.get("PINCHBENCH_CLEAN_BENCHMARK_WORKSPACE", "1").strip().lower()
    enabled = raw in {"1", "true", "yes", "on"}
    return enabled and str(getattr(task, "task_id", "")).startswith("task_")


def _workspace_contamination_paths() -> list[str]:
    return [
        "SOUL.md",
        "BOOTSTRAP.md",
        "USER.md",
        "IDENTITY.md",
        "HEARTBEAT.md",
        "TOOLS.md",
        "AGENTS.md",
        "MEMORY.md",
        "REPORT.md",
        "INBOX.md",
        "emails.md",
        "emails.txt",
        "inbox_emails.txt",
        "event_map.md",
        "report_schema.md",
        "triax_report.md",
        "tripe_report.md",
        "memory",
    ]


def _scrub_benchmark_workspace(workspace: Path) -> None:
    import shutil

    for rel in _workspace_contamination_paths():
        target = workspace / rel
        try:
            if target.is_dir():
                shutil.rmtree(target, ignore_errors=True)
            elif target.exists():
                target.unlink()
        except OSError as exc:
            logger.warning("Failed to scrub workspace contamination %s: %s", target, exc)


def prepare_task_workspace(skill_dir: Path, run_id: str, task: Task, agent_id: str) -> Path:
    """
    Prepare workspace for a task by copying fixtures.
    Uses the agent's configured workspace to ensure files are in the right place.
    """
    import shutil

    # Prefer PinchBench run layout (must match ensure_agent_exists / openclaw agent workspace).
    workspace = _pinchbench_agent_workspace_for_run(run_id)
    if workspace is None:
        workspace = _get_agent_workspace(agent_id)
    if workspace is None:
        # Fallback to task-specific workspace if agent workspace not found
        logger.warning("Could not find agent workspace, using fallback")
        rid = str(run_id).split("-", 1)[0] if run_id else ""
        run_root = Path(os.environ.get("PINCHBENCH_RUN_ROOT", "/tmp/pinchbench"))
        workspace = run_root / (rid or run_id) / task.task_id

    bootstrap_files = ["SOUL.md", "BOOTSTRAP.md", "USER.md", "IDENTITY.md", "HEARTBEAT.md", "TOOLS.md"]
    clean_mode = _benchmark_workspace_clean_mode(task)

    def _remove_readonly(func, path, _):
        try:
            os.chmod(path, stat.S_IWRITE)
            func(path)
        except OSError:
            pass

    saved_bootstrap: dict[str, bytes] = {}
    if workspace.exists():
        if not clean_mode:
            for fname in bootstrap_files:
                fpath = workspace / fname
                if fpath.exists():
                    saved_bootstrap[fname] = fpath.read_bytes()
        shutil.rmtree(workspace, onerror=_remove_readonly)
    workspace.mkdir(parents=True, exist_ok=True)

    if not clean_mode:
        for fname, content in saved_bootstrap.items():
            (workspace / fname).write_bytes(content)
    else:
        logger.info("Preparing clean benchmark workspace for %s (no bootstrap restore)", task.task_id)

    for file_spec in task.workspace_files:
        if "content" in file_spec:
            dest = workspace / file_spec["path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(file_spec["content"])
            continue

        source = skill_dir / "assets" / file_spec["source"]
        dest = workspace / file_spec["dest"]
        dest.parent.mkdir(parents=True, exist_ok=True)
        try:
            dest.write_bytes(source.read_bytes())
        except FileNotFoundError:
            logger.error("Workspace file not found: %s", source)
            raise

    # Copy skills from main workspace to benchmark workspace
    # This enables benchmark agents to use installed skills like nano-pdf
    main_skills_dir = Path.home() / ".openclaw" / "workspace" / "skills"
    if main_skills_dir.exists():
        dest_skills_dir = workspace / "skills"
        dest_skills_dir.mkdir(parents=True, exist_ok=True)
        for skill_dir_src in main_skills_dir.iterdir():
            if skill_dir_src.is_dir():
                if skill_dir_src.name in PINCHBENCH_SKIP_SKILLS:
                    logger.info("Skipped benchmark skill copy: %s", skill_dir_src.name)
                    continue
                dest_skill_dir = dest_skills_dir / skill_dir_src.name
                # Copy skill directory
                import shutil

                if dest_skill_dir.exists():
                    shutil.rmtree(dest_skill_dir, onerror=_remove_readonly)
                shutil.copytree(skill_dir_src, dest_skill_dir)
                logger.info("Copied skill to benchmark workspace: %s", skill_dir_src.name)

    if clean_mode:
        _scrub_benchmark_workspace(workspace)

    return workspace


def _get_agent_store_dir(agent_id: str) -> Path:
    base_dir = _get_openclaw_home() / "agents"
    # OpenClaw normalizes agent IDs to lowercase and replaces colons with dashes
    normalized_id = agent_id.replace(":", "-").lower()
    direct_dir = base_dir / agent_id
    if direct_dir.exists():
        return direct_dir
    normalized_dir = base_dir / normalized_id
    if normalized_dir.exists():
        return normalized_dir
    return direct_dir


def _get_openclaw_home() -> Path:
    pinch_home = os.environ.get("PINCHBENCH_OPENCLAW_HOME")
    if pinch_home:
        return Path(pinch_home).expanduser()
    openclaw_home = os.environ.get("OPENCLAW_HOME")
    if openclaw_home:
        # OpenClaw CLI treats OPENCLAW_HOME as the parent directory and stores
        # config under "$OPENCLAW_HOME/.openclaw".
        return Path(openclaw_home).expanduser() / ".openclaw"
    return Path.home() / ".openclaw"


def _remote_openclaw_enabled() -> bool:
    if os.environ.get("PINCHBENCH_FORCE_LOCAL_OPENCLAW", "").strip().lower() in {"1", "true", "yes"}:
        return False
    host = (os.environ.get("OPENCLAW_HOST") or os.environ.get("ECS_HOST") or "").strip()
    return bool(host and host not in {"localhost", "127.0.0.1"})


def _remote_openclaw_ssh_parts() -> tuple[str, str, str, str]:
    host = (os.environ.get("OPENCLAW_HOST") or os.environ.get("ECS_HOST") or "").strip()
    user = os.environ.get("OPENCLAW_USER", "root")
    port = os.environ.get("OPENCLAW_PORT", "22")
    ssh_key = os.environ.get("OPENCLAW_SSH_KEY", str(Path.home() / ".ssh" / "id_ed25519"))
    return host, user, port, ssh_key


def _remote_openclaw_activate_prefix() -> str:
    return os.environ.get("OPENCLAW_REMOTE_ACTIVATE_CMD", "").strip()


def _remote_openclaw_home_parent() -> str:
    return os.environ.get(
        "PINCHBENCH_REMOTE_OPENCLAW_HOME",
        "/tmp/pinchbench/openclaw_home/pinchbench_benchmark",
    ).strip()


def _remote_openclaw_shell_prefix() -> str:
    home_parent = _remote_openclaw_home_parent()
    if not home_parent:
        return ""
    patch_code = (
        "import json, os; "
        "p=os.path.join(os.environ['OPENCLAW_HOME'],'.openclaw','openclaw.json'); "
        "d={}; "
        "\ntry:\n d=json.load(open(p,encoding='utf-8'))\nexcept Exception:\n d={}\n"
        "a=d.setdefault('agents',{}); "
        "a.setdefault('list',[]); "
        "defaults=a.setdefault('defaults',{}); "
        "defaults['skipBootstrap']=True; "
        "defaults['skills']=[]; "
        "open(p,'w',encoding='utf-8').write(json.dumps(d,indent=2,ensure_ascii=False)+'\\n')"
    )
    return " && ".join(
        [
            f"export OPENCLAW_HOME={shlex.quote(home_parent)}",
            "mkdir -p \"$OPENCLAW_HOME/.openclaw\"",
            f"python3 -c {shlex.quote(patch_code)}",
        ]
    )


def _remote_openclaw_agents_dir_expr() -> str:
    home_parent = _remote_openclaw_home_parent()
    if not home_parent:
        return "$HOME/.openclaw/agents"
    return f"{shlex.quote(home_parent)}/.openclaw/agents"


def _sync_remote_openclaw_agent(agent_id: str) -> None:
    host, user, port, ssh_key = _remote_openclaw_ssh_parts()
    if not host:
        return
    local_home = _get_openclaw_home()
    local_agent_dir = local_home / "agents" / agent_id
    local_agent_dir.parent.mkdir(parents=True, exist_ok=True)
    remote_agents_dir = _remote_openclaw_agents_dir_expr()
    try:
        result = subprocess.run(
            [
                "rsync",
                "-az",
                "--delete",
                "--no-owner",
                "--no-group",
                "--timeout=60",
                "-e",
                f"ssh -o StrictHostKeyChecking=no -i {ssh_key} -p {port}",
                f"{user}@{host}:{remote_agents_dir}/{agent_id}/",
                f"{local_agent_dir}/",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=120,
        )
    except subprocess.TimeoutExpired as exc:
        logger.warning("Timed out syncing OpenClaw agent from ECS: %s", exc)
        return
    if result.returncode != 0:
        logger.warning("Failed to sync OpenClaw agent from ECS: %s %s", result.stderr, result.stdout)
        return
    # Some OpenClaw builds store sessions under the normalized agent directory.
    normalized_id = agent_id.replace(":", "-").lower()
    if normalized_id != agent_id and not (local_agent_dir / "sessions").exists():
        normalized_dir = local_home / "agents" / normalized_id
        normalized_dir.parent.mkdir(parents=True, exist_ok=True)
        try:
            normalized_result = subprocess.run(
                [
                    "rsync",
                    "-az",
                    "--delete",
                    "--no-owner",
                    "--no-group",
                    "--timeout=60",
                    "-e",
                    f"ssh -o StrictHostKeyChecking=no -i {ssh_key} -p {port}",
                    f"{user}@{host}:{remote_agents_dir}/{normalized_id}/",
                    f"{normalized_dir}/",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=120,
            )
        except subprocess.TimeoutExpired as exc:
            logger.debug("Timed out syncing normalized OpenClaw agent from ECS: %s", exc)
            return
        if normalized_result.returncode != 0:
            logger.debug(
                "No normalized OpenClaw agent dir synced for %s: %s %s",
                normalized_id,
                normalized_result.stderr,
                normalized_result.stdout,
            )


def _sync_local_openclaw_agent(agent_id: str, model_id: str, workspace: Path) -> None:
    """Create/update the remote benchmark agent and push local model auth files."""
    host, user, port, ssh_key = _remote_openclaw_ssh_parts()
    if not host:
        return
    local_home = _get_openclaw_home()
    local_agent_dir = local_home / "agents" / agent_id
    if not local_agent_dir.exists():
        logger.warning("Local OpenClaw agent dir missing, cannot sync to ECS: %s", local_agent_dir)
        return
    remote_agents_dir = _remote_openclaw_agents_dir_expr()
    remote_setup = _remote_openclaw_shell_prefix()
    remote_parts = []
    if remote_setup:
        remote_parts.append(remote_setup)
    remote_parts.extend(
        [
            f"mkdir -p {shlex.quote(str(workspace))}",
            f"openclaw agents delete {shlex.quote(agent_id)} --force >/dev/null 2>&1 || true",
            (
                f"openclaw agents add {shlex.quote(agent_id)} "
                f"--model custom/{shlex.quote(_openclaw_catalog_model_id(model_id))} "
                f"--workspace {shlex.quote(str(workspace))} --non-interactive >/dev/null 2>&1 || true"
            ),
            f"mkdir -p {remote_agents_dir}/{shlex.quote(agent_id)}/agent",
        ]
    )
    remote_command = " && ".join(remote_parts)
    create_result = subprocess.run(
        [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-i", ssh_key,
            "-p", str(port),
            f"{user}@{host}",
            remote_command,
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if create_result.returncode != 0:
        logger.warning(
            "Failed to create remote OpenClaw agent %s: %s %s",
            agent_id,
            create_result.stderr,
            create_result.stdout,
        )
    result = subprocess.run(
        [
            "rsync",
            "-az",
            "--delete",
            "-e",
            f"ssh -o StrictHostKeyChecking=no -i {ssh_key} -p {port}",
            f"{local_agent_dir}/agent/",
            f"{user}@{host}:{remote_agents_dir}/{agent_id}/agent/",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        logger.warning("Failed to sync OpenClaw agent to ECS: %s %s", result.stderr, result.stdout)


def _cleanup_remote_agent_sessions(agent_id: str) -> None:
    """Remove ECS OpenClaw session state for one benchmark agent before a fresh episode."""
    host, user, port, ssh_key = _remote_openclaw_ssh_parts()
    if not host:
        return
    remote_agents_dir = _remote_openclaw_agents_dir_expr()
    normalized_id = agent_id.replace(":", "-").lower()
    targets = {agent_id, normalized_id}
    remote_parts = []
    remote_setup = _remote_openclaw_shell_prefix()
    if remote_setup:
        remote_parts.append(remote_setup)
    for target in sorted(targets):
        remote_parts.append(f"rm -rf {remote_agents_dir}/{shlex.quote(target)}/sessions")
    remote_command = " && ".join(remote_parts)
    try:
        result = subprocess.run(
            [
                "ssh",
                "-o", "StrictHostKeyChecking=no",
                "-o", "ConnectTimeout=10",
                "-i", ssh_key,
                "-p", str(port),
                f"{user}@{host}",
                remote_command,
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=60,
        )
    except subprocess.TimeoutExpired as exc:
        logger.warning("Timed out cleaning remote OpenClaw sessions for %s: %s", agent_id, exc)
        return
    if result.returncode != 0:
        logger.warning(
            "Failed to clean remote OpenClaw sessions for %s: %s %s",
            agent_id,
            result.stderr,
            result.stdout,
        )


def _sync_workspace_to_remote(workspace: Path) -> None:
    """Mirror the prepared local benchmark workspace to the same absolute path on ECS."""
    host, user, port, ssh_key = _remote_openclaw_ssh_parts()
    if not host:
        return
    workspace.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ssh",
            "-o", "StrictHostKeyChecking=no",
            "-i", ssh_key,
            "-p", str(port),
            f"{user}@{host}",
            f"mkdir -p {str(workspace.parent)}",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    result = subprocess.run(
        [
            "rsync",
            "-az",
            "--delete",
            "-e",
            f"ssh -o StrictHostKeyChecking=no -i {ssh_key} -p {port}",
            f"{workspace}/",
            f"{user}@{host}:{workspace}/",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        logger.warning("Failed to sync workspace to ECS: %s %s", result.stderr, result.stdout)


def _sync_workspace_from_remote(workspace: Path) -> None:
    """Fetch ECS workspace outputs back before transcript/grading reads local files."""
    host, user, port, ssh_key = _remote_openclaw_ssh_parts()
    if not host:
        return
    workspace.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        [
            "rsync",
            "-az",
            "--delete",
            "-e",
            f"ssh -o StrictHostKeyChecking=no -i {ssh_key} -p {port}",
            f"{user}@{host}:{workspace}/",
            f"{workspace}/",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        logger.warning("Failed to sync workspace from ECS: %s %s", result.stderr, result.stdout)


def _resolve_session_id_from_store(agent_id: str) -> str | None:
    agent_dir = _get_agent_store_dir(agent_id)
    sessions_store = agent_dir / "sessions" / "sessions.json"
    if not sessions_store.exists():
        return None
    try:
        sessions_payload = json.loads(sessions_store.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.warning("Failed to parse sessions store: %s", exc)
        return None
    if not isinstance(sessions_payload, dict):
        return None

    normalized_id = agent_id.replace(":", "-").lower()
    preferred_keys = [
        f"agent:{agent_id}:main",
        f"agent:{agent_id}:default",
        f"agent:{normalized_id}:main",
        f"agent:{normalized_id}:default",
    ]
    for key in preferred_keys:
        entry = sessions_payload.get(key)
        if isinstance(entry, dict) and entry.get("sessionId"):
            return entry["sessionId"]

    newest_entry = None
    newest_timestamp = -1
    for entry in sessions_payload.values():
        if not isinstance(entry, dict):
            continue
        if "sessionId" not in entry:
            continue
        updated_at = entry.get("updatedAt")
        if isinstance(updated_at, (int, float)) and updated_at > newest_timestamp:
            newest_timestamp = updated_at
            newest_entry = entry
    if newest_entry:
        return newest_entry.get("sessionId")
    return None


def _find_transcript_path_from_sessions_store(agent_id: str) -> Optional[Path]:
    """Best-effort transcript path resolution from sessions.json payload values."""
    agent_dir = _get_agent_store_dir(agent_id)
    sessions_store = agent_dir / "sessions" / "sessions.json"
    if not sessions_store.exists():
        return None
    try:
        payload = json.loads(sessions_store.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None

    def _iter_strings(node: Any):
        if isinstance(node, str):
            yield node
        elif isinstance(node, dict):
            for value in node.values():
                yield from _iter_strings(value)
        elif isinstance(node, list):
            for value in node:
                yield from _iter_strings(value)

    suffixes = (".jsonl", ".ndjson")
    session_root = agent_dir / "sessions"
    for value in _iter_strings(payload):
        if not value.endswith(suffixes):
            continue
        candidate = Path(value)
        if not candidate.is_absolute():
            candidate = session_root / value
        if candidate.exists():
            return candidate
    return None


def _find_recent_session_path(agent_dir: Path, started_at: float) -> Path | None:
    sessions_dir = agent_dir / "sessions"
    if not sessions_dir.exists():
        return None
    candidates = list(sessions_dir.rglob("*.jsonl")) + list(sessions_dir.rglob("*.ndjson"))
    if not candidates:
        return None
    tolerance_seconds = 5.0
    recent_candidates = [
        path for path in candidates if path.stat().st_mtime >= (started_at - tolerance_seconds)
    ]
    if not recent_candidates:
        return None
    return max(recent_candidates, key=lambda path: path.stat().st_mtime)


def _is_fresh_transcript(path: Path, started_at: float) -> bool:
    try:
        return path.stat().st_mtime >= (started_at - 5.0)
    except OSError:
        return False


def _load_transcript(
    agent_id: str, session_id: str, started_at: float
) -> tuple[List[Dict[str, Any]], Optional[Path]]:
    agent_dir = _get_agent_store_dir(agent_id)
    transcript_path = None

    # OpenClaw ignores the --session-id we pass and generates its own UUID-based
    # session ID internally.  We need to discover the actual transcript path.
    #
    # Strategy (with retries to handle write-delay):
    #   1. Resolve the real session ID from sessions.json
    #   2. Glob for any .jsonl in the sessions dir (most-recently-modified)
    #   3. Try our passed-in session ID as a last resort
    for attempt in range(15):
        # 1. Try our passed-in session ID first. Current OpenClaw builds often
        # honor --session-id for the transcript filename, and this avoids
        # accidentally selecting a freshened stale file via glob fallback.
        for direct_path in (
            agent_dir / "sessions" / f"{session_id}.jsonl",
            agent_dir / "sessions" / f"{session_id}.ndjson",
        ):
            if direct_path.exists() and _is_fresh_transcript(direct_path, started_at):
                transcript_path = direct_path
                logger.info(
                    "Found transcript via passed session ID: %s (attempt %s)",
                    direct_path.name,
                    attempt + 1,
                )
                break
        if transcript_path is not None:
            break

        # 2. Try sessions.json — some OpenClaw versions write the real UUID here
        resolved_session_id = _resolve_session_id_from_store(agent_id)
        if resolved_session_id:
            session_dir = agent_dir / "sessions"
            for candidate in (
                session_dir / f"{resolved_session_id}.jsonl",
                session_dir / f"{resolved_session_id}.ndjson",
                session_dir / resolved_session_id / "transcript.jsonl",
                session_dir / resolved_session_id / "events.jsonl",
            ):
                if candidate.exists() and _is_fresh_transcript(candidate, started_at):
                    transcript_path = candidate
                    logger.info(
                        "Found transcript via sessions.json: %s (attempt %s)",
                        candidate.name,
                        attempt + 1,
                    )
                    break
            if transcript_path is not None:
                break

        # 2b. Parse transcript-like paths from sessions.json values
        candidate_from_store = _find_transcript_path_from_sessions_store(agent_id)
        if candidate_from_store is not None and _is_fresh_transcript(candidate_from_store, started_at):
            transcript_path = candidate_from_store
            logger.info(
                "Found transcript via sessions.json path: %s (attempt %s)",
                candidate_from_store,
                attempt + 1,
            )
            break

        # 3. Glob fallback — pick the most recently modified .jsonl
        recent_path = _find_recent_session_path(agent_dir, started_at)
        if recent_path is not None:
            transcript_path = recent_path
            logger.info(
                "Found transcript via glob fallback: %s (attempt %s)",
                recent_path.name,
                attempt + 1,
            )
            break

        if attempt < 14:
            time.sleep(1.0)

    if transcript_path is None:
        sessions_dir = agent_dir / "sessions"
        if sessions_dir.exists():
            all_files = list(sessions_dir.iterdir())
            logger.warning(
                "Transcript not found for agent %s. Sessions dir contents: %s",
                agent_id,
                [f.name for f in all_files],
            )
            sessions_store = sessions_dir / "sessions.json"
            if sessions_store.exists():
                try:
                    payload_preview = sessions_store.read_text(encoding="utf-8")[:1200]
                    logger.warning("sessions.json preview: %s", payload_preview)
                except OSError as exc:
                    logger.warning("Could not read sessions.json preview: %s", exc)
        else:
            logger.warning(
                "Transcript not found — sessions dir does not exist: %s",
                sessions_dir,
            )
        return [], None

    transcript: List[Dict[str, Any]] = []
    for line in transcript_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            transcript.append(json.loads(line))
        except json.JSONDecodeError as exc:
            logger.warning("Failed to parse transcript line: %s", exc)
            transcript.append({"raw": line, "parse_error": str(exc)})
    return transcript, transcript_path


def _extract_usage_from_transcript(transcript: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Sum token usage and cost from all assistant messages in transcript."""
    totals = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
        "total_tokens": 0,
        "cost_usd": 0.0,
        "request_count": 0,
    }

    for entry in transcript:
        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {})
        if msg.get("role") != "assistant":
            continue
        totals["request_count"] += 1
        usage = msg.get("usage", {})
        totals["input_tokens"] += usage.get("input", 0)
        totals["output_tokens"] += usage.get("output", 0)
        totals["cache_read_tokens"] += usage.get("cacheRead", 0)
        totals["cache_write_tokens"] += usage.get("cacheWrite", 0)
        totals["total_tokens"] += usage.get("totalTokens", 0)
        cost = usage.get("cost", {})
        totals["cost_usd"] += cost.get("total", 0.0)

    return totals


def _has_successful_assistant_turn(transcript: List[Dict[str, Any]]) -> bool:
    """Return true only when the model produced a non-error assistant turn."""
    for entry in transcript:
        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {})
        if msg.get("role") != "assistant":
            continue
        if msg.get("stopReason") == "error" or msg.get("errorMessage"):
            continue
        if msg.get("tool_calls") or msg.get("toolCalls"):
            return True
        content = msg.get("content", [])
        if isinstance(content, str) and content.strip():
            return True
        if isinstance(content, list):
            for item in content:
                if not isinstance(item, dict):
                    continue
                if item.get("type") in {"toolCall", "tool_use", "function_call"}:
                    return True
                if item.get("type") == "text" and str(item.get("text", "")).strip():
                    return True
    return False


def execute_openclaw_task(
    *,
    task: Task,
    agent_id: str,
    model_id: str,
    run_id: str,
    timeout_multiplier: float,
    skill_dir: Path,
    output_dir: Optional[Path] = None,
    verbose: bool = False,
) -> Dict[str, Any]:
    logger.info("🤖 Agent [%s] starting task: %s", agent_id, task.task_id)
    logger.info("   Task: %s", task.name)
    logger.info("   Category: %s", task.category)
    if verbose:
        logger.info(
            "   Prompt: %s", task.prompt[:500] + "..." if len(task.prompt) > 500 else task.prompt
        )

    # Clean up previous session transcripts so we can reliably find this task's
    # transcript (OpenClaw uses its own UUID-based naming, not our session ID).
    cleanup_agent_sessions(agent_id)
    _repair_bench_models_json_defaults(agent_id, model_id)

    start_time = time.time()
    workspace = prepare_task_workspace(skill_dir, run_id, task, agent_id)
    session_id = f"{task.task_id}_{int(time.time() * 1000)}"
    timeout_seconds = task.timeout_seconds * timeout_multiplier
    stdout = ""
    stderr = ""
    exit_code = -1
    timed_out = False

    remote_enabled = _remote_openclaw_enabled()
    logger.info(
        "OpenClaw execution mode: %s (workspace=%s)",
        "remote" if remote_enabled else "local",
        workspace,
    )
    if remote_enabled:
        _sync_local_openclaw_agent(agent_id, model_id, workspace)
        _cleanup_remote_agent_sessions(agent_id)
        _sync_workspace_to_remote(workspace)
    sessions = task.frontmatter.get("sessions", [])
    if sessions:
        # Multi-session task: send each prompt in sequence
        logger.info("📋 Multi-session task with %d sessions", len(sessions))
        for i, session_entry in enumerate(sessions, 1):
            # Extract prompt text from session entry (handle both string and dict formats)
            if isinstance(session_entry, str):
                session_prompt = session_entry
            elif isinstance(session_entry, dict):
                session_prompt = session_entry.get("prompt") or session_entry.get("message", "")
            else:
                logger.warning("⚠️ Skipping invalid session entry: %s", session_entry)
                continue

            logger.info("   Session %d/%d", i, len(sessions))
            elapsed = time.time() - start_time
            remaining = timeout_seconds - elapsed
            if remaining <= 0:
                timed_out = True
                break
            try:
                if remote_enabled:
                    host, user, port, ssh_key = _remote_openclaw_ssh_parts()
                    remote_prefix = _remote_openclaw_activate_prefix()
                    escaped_prompt = session_prompt.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
                    remote_cmd_parts = []
                    remote_setup = _remote_openclaw_shell_prefix()
                    if remote_setup:
                        remote_cmd_parts.append(remote_setup)
                    remote_cmd_parts.append(f"cd {shlex.quote(str(workspace))}")
                    if remote_prefix:
                        remote_cmd_parts.append(remote_prefix)
                    remote_cmd_parts.extend([
                        f"openclaw agent --agent {agent_id} --session-id {session_id} --message \"{escaped_prompt}\" --local",
                    ])
                    remote_command = " && ".join(remote_cmd_parts)
                    result = _run_with_process_group(
                        [
                            "ssh",
                            "-o", "StrictHostKeyChecking=no",
                            "-o", "ConnectTimeout=10",
                            "-i", ssh_key,
                            "-p", str(port),
                            f"{user}@{host}",
                            remote_command,
                        ],
                        timeout=remaining,
                    )
                else:
                    result = _run_with_process_group(
                        [
                            "openclaw",
                            "agent",
                            "--agent",
                            agent_id,
                            "--session-id",
                            session_id,
                            "--message",
                            session_prompt,
                            "--local",
                        ],
                        cwd=str(workspace),
                        timeout=remaining,
                        shell=USE_SHELL,
                    )
                stdout += result.stdout
                stderr += result.stderr
                exit_code = result.returncode
                if result.returncode not in (0, -1):
                    break
            except subprocess.TimeoutExpired as exc:
                timed_out = True
                stdout += _coerce_subprocess_output(exc.stdout)
                stderr += _coerce_subprocess_output(exc.stderr)
                break
            except FileNotFoundError as exc:
                stderr = f"openclaw command not found: {exc}"
                break
    else:
        # Single-session task: send task.prompt once
        try:
            effective_prompt = _compose_task_prompt(task)
            if remote_enabled:
                host, user, port, ssh_key = _remote_openclaw_ssh_parts()
                remote_prefix = _remote_openclaw_activate_prefix()
                escaped_prompt = effective_prompt.replace("\\", "\\\\").replace('"', '\\"').replace("$", "\\$").replace("`", "\\`")
                remote_cmd_parts = []
                remote_setup = _remote_openclaw_shell_prefix()
                if remote_setup:
                    remote_cmd_parts.append(remote_setup)
                remote_cmd_parts.append(f"cd {shlex.quote(str(workspace))}")
                if remote_prefix:
                    remote_cmd_parts.append(remote_prefix)
                remote_cmd_parts.extend([
                    f"openclaw agent --agent {agent_id} --session-id {session_id} --message \"{escaped_prompt}\" --local",
                ])
                remote_command = " && ".join(remote_cmd_parts)
                result = _run_with_process_group(
                    [
                        "ssh",
                        "-o", "StrictHostKeyChecking=no",
                        "-o", "ConnectTimeout=10",
                        "-i", ssh_key,
                        "-p", str(port),
                        f"{user}@{host}",
                        remote_command,
                    ],
                    timeout=timeout_seconds,
                )
            else:
                result = _run_with_process_group(
                    [
                        "openclaw",
                        "agent",
                        "--agent",
                        agent_id,
                        "--session-id",
                        session_id,
                        "--message",
                        effective_prompt,
                        "--local",
                    ],
                    cwd=str(workspace),
                    timeout=timeout_seconds,
                    shell=USE_SHELL,
                )
            stdout = result.stdout
            stderr = result.stderr
            exit_code = result.returncode
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout = _coerce_subprocess_output(exc.stdout)
            stderr = _coerce_subprocess_output(exc.stderr)
        except FileNotFoundError as exc:
            stderr = f"openclaw command not found: {exc}"

    if remote_enabled:
        _sync_remote_openclaw_agent(agent_id)
        _sync_workspace_from_remote(workspace)
    transcript, transcript_path = _load_transcript(agent_id, session_id, start_time)
    if not transcript and exit_code == 0 and stdout.strip():
        logger.warning(
            "Transcript missing for %s; using stdout fallback transcript for grading.",
            task.task_id,
        )
        prompt_text = task.prompt
        if not sessions:
            prompt_text = _compose_task_prompt(task)
        transcript = [
            {
                "type": "message",
                "message": {
                    "role": "user",
                    "content": [{"type": "text", "text": prompt_text}],
                },
            },
            {
                "type": "message",
                "message": {
                    "role": "assistant",
                    "content": [{"type": "text", "text": stdout.strip()}],
                    "stopReason": "stop",
                },
            },
        ]
    usage = _extract_usage_from_transcript(transcript)
    execution_time = time.time() - start_time

    # Archive the raw transcript JSONL before cleanup_agent_sessions deletes it
    if transcript_path and output_dir:
        import shutil as _shutil
        output_dir.mkdir(parents=True, exist_ok=True)
        archive_dest = output_dir / f"{task.task_id}__{session_id}.jsonl"
        latest_dest = output_dir / f"{task.task_id}.jsonl"
        try:
            _shutil.copy2(transcript_path, archive_dest)
            logger.info("Archived transcript to %s", archive_dest)
            _shutil.copy2(transcript_path, latest_dest)
        except OSError as exc:
            logger.warning("Failed to archive transcript: %s", exc)

    status = "success"
    if timed_out:
        status = "timeout"
    if not transcript:
        status = "error"
    elif not _has_successful_assistant_turn(transcript):
        status = "error"
        if not stderr:
            stderr = "OpenClaw produced no successful assistant turn"
    if exit_code not in (0, -1) and not timed_out:
        status = "error"
    if stderr and "openclaw command not found" in str(stderr):
        status = "error"

    # Verbose logging for debugging
    if verbose:
        logger.info("   [VERBOSE] Exit code: %s", exit_code)
        logger.info("   [VERBOSE] Execution time: %.2fs", execution_time)
        logger.info("   [VERBOSE] Workspace: %s", workspace)
        if stdout:
            logger.info("   [VERBOSE] Stdout (first 1000 chars):\n%s", stdout[:1000])
        if stderr:
            logger.info("   [VERBOSE] Stderr:\n%s", stderr[:1000])
        logger.info("   [VERBOSE] Transcript entries: %d", len(transcript))

        # Show agent responses from transcript
        for entry in transcript:
            if entry.get("type") == "message":
                msg = entry.get("message", {})
                role = msg.get("role", "unknown")
                content = msg.get("content", "")
                if role == "assistant":
                    # Truncate long responses
                    preview = content[:500] + "..." if len(content) > 500 else content
                    logger.info("   [VERBOSE] Agent response: %s", preview)
                elif role == "user":
                    preview = content[:200] + "..." if len(content) > 200 else content
                    logger.info("   [VERBOSE] User message: %s", preview)

        # Show workspace files after task
        if workspace.exists():
            logger.info("   [VERBOSE] Workspace files after task:")
            for f in sorted(workspace.rglob("*")):
                if f.is_file():
                    try:
                        size = f.stat().st_size
                        logger.info("      %s (%d bytes)", f.relative_to(workspace), size)
                    except OSError:
                        logger.info("      %s", f.relative_to(workspace))

    return {
        "agent_id": agent_id,
        "task_id": task.task_id,
        "status": status,
        "transcript": transcript,
        "usage": usage,
        "workspace": str(workspace),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "execution_time": execution_time,
        "stdout": stdout,
        "stderr": stderr,
    }


def run_openclaw_prompt(
    *,
    agent_id: str,
    prompt: str,
    workspace: Path,
    timeout_seconds: float,
) -> Dict[str, Any]:
    """Run a single OpenClaw prompt for helper agents like the judge."""
    cleanup_agent_sessions(agent_id)

    agent_workspace = _get_agent_workspace(agent_id)
    if agent_workspace and agent_workspace.exists():
        for bootstrap_file in ["BOOTSTRAP.md", "SOUL.md", "USER.md", "IDENTITY.md", "HEARTBEAT.md"]:
            bp = agent_workspace / bootstrap_file
            if bp.exists():
                try:
                    bp.unlink()
                    logger.debug("Removed bootstrap file from judge workspace: %s", bootstrap_file)
                except OSError as exc:
                    logger.warning("Failed to remove bootstrap file %s: %s", bootstrap_file, exc)

    start_time = time.time()
    workspace.mkdir(parents=True, exist_ok=True)
    session_id = f"judge_{int(time.time() * 1000)}"
    stdout = ""
    stderr = ""
    exit_code = -1
    timed_out = False

    chunks = [
        prompt[i : i + JUDGE_MAX_MSG_CHARS]
        for i in range(0, max(1, len(prompt)), JUDGE_MAX_MSG_CHARS)
    ]
    if len(chunks) > 1:
        total_chunks = len(chunks)
        chunks = [
            (
                f"You are receiving a long prompt in {total_chunks} parts.\n"
                f"Ignore and do not respond until the final part.\n\n"
                f"Part 1/{total_chunks}:\n{chunks[0]}"
            )
        ] + [
            (
                f"Part {i + 2}/{total_chunks}:\n{chunks[i + 1]}"
                if i + 2 < total_chunks
                else (
                    f"Part {i + 2}/{total_chunks} (final):\n{chunks[i + 1]}\n"
                    "All parts received. Proceed with final judgment now."
                )
            )
            for i in range(0, total_chunks - 1)
        ]
    for chunk in chunks:
        elapsed = time.time() - start_time
        remaining = timeout_seconds - elapsed
        if remaining <= 0:
            timed_out = True
            break
        try:
            openclaw_path = os.environ.get("OPENCLAW_PATH", "openclaw")
            # On Windows, cmd.exe splits command-line arguments at literal newlines,
            # causing the message to be truncated after the first line.
            # Escape newlines to literal \n sequences so the full prompt is received.
            send_chunk = (
                chunk.replace("\r\n", "\\n").replace("\n", "\\n").replace("\r", "\\n")
                if USE_SHELL
                else chunk
            )
            if _remote_openclaw_enabled():
                host, user, port, ssh_key = _remote_openclaw_ssh_parts()
                remote_prefix = _remote_openclaw_activate_prefix()
                escaped_chunk = send_chunk.replace('"', '\\"')
                remote_cmd_parts = [f"cd {workspace}"]
                if remote_prefix:
                    remote_cmd_parts.append(remote_prefix)
                remote_cmd_parts.extend([
                    f"{openclaw_path} agent --agent {agent_id} --session-id {session_id} --message \"{escaped_chunk}\" --local",
                ])
                remote_command = " && ".join(remote_cmd_parts)
                result = subprocess.run(
                    [
                        "ssh",
                        "-o", "StrictHostKeyChecking=no",
                        "-o", "ConnectTimeout=10",
                        "-i", ssh_key,
                        "-p", str(port),
                        f"{user}@{host}",
                        remote_command,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=remaining,
                    check=False,
                )
            else:
                result = subprocess.run(
                    [
                        openclaw_path,
                        "agent",
                        "--agent",
                        agent_id,
                        "--session-id",
                        session_id,
                        "--message",
                        send_chunk,
                        "--local",
                    ],
                    capture_output=True,
                    text=True,
                    cwd=str(workspace),
                    timeout=remaining,
                    check=False,
                    shell=USE_SHELL,
                )
            stdout += result.stdout
            stderr += result.stderr
            exit_code = result.returncode
            if result.returncode not in (0, -1) and not timed_out:
                break
        except subprocess.TimeoutExpired as exc:
            timed_out = True
            stdout += _coerce_subprocess_output(exc.stdout)
            stderr += _coerce_subprocess_output(exc.stderr)
            break
        except FileNotFoundError as exc:
            stderr += f"openclaw command not found: {exc}"
            break

    if _remote_openclaw_enabled():
        _sync_remote_openclaw_agent(agent_id)
    transcript, _ = _load_transcript(agent_id, session_id, start_time)
    execution_time = time.time() - start_time

    status = "success"
    if timed_out:
        status = "timeout"
    if not transcript:
        status = "error"
    if exit_code not in (0, -1) and not timed_out:
        status = "error"
    if stderr and "openclaw command not found" in str(stderr):
        status = "error"

    return {
        "agent_id": agent_id,
        "status": status,
        "transcript": transcript,
        "workspace": str(workspace),
        "exit_code": exit_code,
        "timed_out": timed_out,
        "execution_time": execution_time,
        "stdout": stdout,
        "stderr": stderr,
    }


_JUDGE_SYSTEM_MSG = (
    "You are a strict grading function. "
    "Respond with ONLY a JSON object, no prose, no markdown fences, no extra text."
)


def call_judge_api(
    *,
    prompt: str,
    model: str,
    timeout_seconds: float = 120.0,
    base_url: Optional[str] = None,
    api_key: Optional[str] = None,
    response_json: bool = False,
) -> Dict[str, Any]:
    """Call a judge model directly via API, bypassing OpenClaw.

    Dispatches based on model prefix:
      - openrouter/* -> OpenRouter chat completions API
      - anthropic/*  -> Anthropic Messages API
      - openai/*     -> OpenAI chat completions API
      - claude       -> headless Claude CLI (claude -p)
      - (custom base_url) -> any OpenAI-compatible endpoint

    Returns {"status": str, "text": str, "error"?: str}.
    """
    # Custom endpoint takes priority over prefix-based dispatch
    if base_url:
        resolved_key = (
            api_key
            or os.environ.get("DEEPSEEK_API_KEY")
            or os.environ.get("PINCHBENCH_GRADE_JUDGE_API_KEY")
            or os.environ.get("JUDGE_API_KEY")
            or os.environ.get("OPENAI_API_KEY", "")
        )
        if not resolved_key:
            return {"status": "error", "text": "", "error": "No API key provided for custom judge endpoint"}
        # Strip provider prefixes — custom endpoints (e.g. DeepSeek) only accept
        # bare model names like ``deepseek-chat``. The CLI/user-facing form
        # (``openai/deepseek-chat``, ``openrouter/...``) is a routing hint, not
        # a model id, so we must remove it before sending to the upstream API.
        bare_model = model
        for _prefix in ("openai/", "openrouter/", "anthropic/", "deepseek/"):
            if bare_model.startswith(_prefix):
                bare_model = bare_model[len(_prefix):]
                break
        endpoint = base_url.rstrip("/") + "/chat/completions"
        return _judge_via_openai_compat(
            prompt, bare_model, endpoint, resolved_key, timeout_seconds, response_json=response_json
        )
    if model == "claude" or model.startswith("claude:"):
        return _judge_via_claude_cli(prompt, model, timeout_seconds)
    if model.startswith("anthropic/"):
        return _judge_via_anthropic(prompt, model, timeout_seconds)
    if model.startswith("openai/"):
        return _judge_via_openai(prompt, model, timeout_seconds)
    # Default: OpenRouter (handles openrouter/ prefix and bare provider/model)
    return _judge_via_openrouter(prompt, model, timeout_seconds)


def _judge_via_openai_compat(
    prompt: str,
    api_model: str,
    endpoint: str,
    api_key: str,
    timeout_seconds: float,
    extra_headers: Optional[Dict[str, str]] = None,
    *,
    response_json: bool = False,
) -> Dict[str, Any]:
    """Shared implementation for OpenAI-compatible chat completions APIs."""
    body: Dict[str, Any] = {
        "model": api_model,
        "messages": [
            {"role": "system", "content": _JUDGE_SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0.0,
        "max_tokens": 2048,
    }
    if response_json:
        body["response_format"] = {"type": "json_object"}
    payload = json.dumps(body).encode("utf-8")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    if extra_headers:
        headers.update(extra_headers)

    req = request.Request(endpoint, data=payload, headers=headers, method="POST")
    try:
        with request.urlopen(req, timeout=timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        logger.error("Judge API error (%s): %s", exc.code, body)
        return {"status": "error", "text": "", "error": f"HTTP {exc.code}: {body}"}
    except error.URLError as exc:
        logger.error("Judge network error: %s", exc)
        return {"status": "error", "text": "", "error": str(exc)}
    except TimeoutError:
        return {"status": "timeout", "text": "", "error": "Request timed out"}

    choices = data.get("choices", [])
    if not choices:
        return {"status": "error", "text": "", "error": "No choices in response"}
    text = choices[0].get("message", {}).get("content", "")
    return {"status": "success", "text": text}


def _judge_via_openrouter(prompt: str, model: str, timeout_seconds: float) -> Dict[str, Any]:
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        return {"status": "error", "text": "", "error": "OPENROUTER_API_KEY not set"}
    bare_model = model.removeprefix("openrouter/")
    return _judge_via_openai_compat(
        prompt, bare_model,
        "https://openrouter.ai/api/v1/chat/completions",
        api_key, timeout_seconds,
        extra_headers={"HTTP-Referer": "https://pinchbench.com", "X-Title": "PinchBench-Judge"},
    )


def _judge_via_openai(prompt: str, model: str, timeout_seconds: float) -> Dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return {"status": "error", "text": "", "error": "OPENAI_API_KEY not set"}
    bare_model = model.removeprefix("openai/")
    return _judge_via_openai_compat(
        prompt, bare_model,
        "https://api.openai.com/v1/chat/completions",
        api_key, timeout_seconds,
    )


def _judge_via_anthropic(prompt: str, model: str, timeout_seconds: float) -> Dict[str, Any]:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {"status": "error", "text": "", "error": "ANTHROPIC_API_KEY not set"}
    bare_model = model.removeprefix("anthropic/")
    payload = json.dumps({
        "model": bare_model,
        "max_tokens": 2048,
        "temperature": 0.0,
        "system": _JUDGE_SYSTEM_MSG,
        "messages": [{"role": "user", "content": prompt}],
    }).encode("utf-8")
    headers = {
        "x-api-key": api_key,
        "Content-Type": "application/json",
        "anthropic-version": "2023-06-01",
    }
    req = request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload, headers=headers, method="POST",
    )
    try:
        with request.urlopen(req, timeout=timeout_seconds) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except error.HTTPError as exc:
        body = ""
        try:
            body = exc.read().decode("utf-8", errors="replace")[:500]
        except Exception:
            pass
        logger.error("Anthropic judge API error (%s): %s", exc.code, body)
        return {"status": "error", "text": "", "error": f"HTTP {exc.code}: {body}"}
    except error.URLError as exc:
        logger.error("Anthropic judge network error: %s", exc)
        return {"status": "error", "text": "", "error": str(exc)}
    except TimeoutError:
        return {"status": "timeout", "text": "", "error": "Request timed out"}

    content = data.get("content", [])
    text = "".join(block.get("text", "") for block in content if block.get("type") == "text")
    return {"status": "success", "text": text}


def _judge_via_claude_cli(prompt: str, model: str, timeout_seconds: float) -> Dict[str, Any]:
    """Use headless Claude CLI (claude -p) as judge."""
    cmd: List[str] = ["claude", "-p"]
    # Support "claude:model-name" to pass --model
    if ":" in model:
        _, cli_model = model.split(":", 1)
        cmd.extend(["--model", cli_model])
    try:
        result = subprocess.run(
            cmd,
            input=f"{_JUDGE_SYSTEM_MSG}\n\n{prompt}",
            capture_output=True,
            text=True,
            timeout=timeout_seconds,
            check=False,
        )
    except FileNotFoundError:
        return {"status": "error", "text": "", "error": "claude CLI not found"}
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "text": "", "error": "claude -p timed out"}
    if result.returncode != 0:
        return {"status": "error", "text": "", "error": f"claude exit {result.returncode}: {result.stderr[:300]}"}
    return {"status": "success", "text": result.stdout}
