"""
Monkey-patch veRL RayPPOTrainer._validate to track best/latest checkpoints.

After each validation, compares val-core/*/reward/mean@* (higher = better).

We intentionally keep this patch conservative:
- Do not delete checkpoint directories here.
- Only update best_ckpt_state.json and latest_checkpointed_iteration.txt.

The older aggressive pruning logic was too easy to misalign with veRL's own
checkpoint layout and could leave only partial artifacts behind. For now we keep
full checkpoint directories so best/latest can be inspected and reused safely.

best_ckpt_state.json includes: best_val, best_step, latest_step.
latest_checkpointed_iteration.txt is set to **latest_step**.

Requires PINCHBENCH_BEST_CKPT=1 and PYTHONPATH including repo root (for sitecustomize).
Save should run on the same steps as validation (e.g. save_freq == test_freq) so each
val has a fresh checkpoint directory to judge.
"""

from __future__ import annotations

import json
import time
from pathlib import Path


def _extract_val_reward(val_metrics: dict) -> float | None:
    for k, v in val_metrics.items():
        if not k.startswith("val-core/"):
            continue
        if "/reward/" not in k:
            continue
        if "mean" not in k:
            continue
        try:
            return float(v)
        except (TypeError, ValueError):
            continue
    return None


def _track_best_and_latest(trainer, val_metrics: dict) -> None:
    default_dir = trainer.config.trainer.default_local_dir
    if not default_dir:
        return
    root = Path(default_dir).resolve()
    state_path = root / "best_ckpt_state.json"

    val_score = _extract_val_reward(val_metrics)
    if val_score is None:
        print("[pinchbench_best_ckpt] no val-core reward metric; skip checkpoint prune")
        return

    step = int(trainer.global_steps)
    cur_dir = root / f"global_step_{step}"
    if not cur_dir.is_dir():
        # save_freq != test_freq: no new checkpoint this step
        return

    best_val = float("-inf")
    best_step: int | None = None
    if state_path.is_file():
        try:
            st = json.loads(state_path.read_text(encoding="utf-8"))
            best_val = float(st.get("best_val", "-inf"))
            bs = st.get("best_step")
            if bs is not None:
                best_step = int(bs)
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass

    tracker = root / "latest_checkpointed_iteration.txt"

    def _write_state(bv: float, bs: int, ls: int) -> None:
        state_path.write_text(
            json.dumps(
                {"best_val": bv, "best_step": bs, "latest_step": ls},
                indent=2,
            ),
            encoding="utf-8",
        )
        if tracker.is_file():
            tracker.write_text(str(ls), encoding="utf-8")

    if val_score > best_val:
        best_val = val_score
        best_step = step
        latest_step = step
        _write_state(best_val, best_step, latest_step)
        print(
            f"[pinchbench_best_ckpt] new best val={best_val:.4f} step={step}; "
            f"kept checkpoint dirs intact"
        )
    else:
        if best_step is None:
            best_step = step
        latest_step = step
        _write_state(best_val, best_step, latest_step)
        print(
            f"[pinchbench_best_ckpt] val={val_score:.4f} <= best={best_val:.4f}; "
            f"updated best/latest tracking only"
        )


def apply_patch() -> None:
    from verl.trainer.ppo import ray_trainer as rt

    if getattr(rt.RayPPOTrainer, "_pinchbench_best_ckpt_patched", False):
        return

    _orig = rt.RayPPOTrainer._validate

    def _validate(self, merged: bool = False):
        step = int(getattr(self, "global_steps", -1))
        t0 = time.monotonic()
        print(f"[pinchbench_validate] start step={step} merged={merged}", flush=True)
        try:
            out = _orig(self, merged=merged)
        except Exception as e:
            dt = time.monotonic() - t0
            print(
                f"[pinchbench_validate] failed step={step} elapsed_s={dt:.1f}: {type(e).__name__}: {e}",
                flush=True,
            )
            raise
        dt = time.monotonic() - t0
        keys = list(out.keys()) if isinstance(out, dict) else []
        print(
            f"[pinchbench_validate] end step={step} elapsed_s={dt:.1f} keys={keys[:8]}",
            flush=True,
        )
        if not merged:
            try:
                _track_best_and_latest(self, out)
            except Exception as e:
                print(f"[pinchbench_best_ckpt] tracking failed: {e}")
        return out

    rt.RayPPOTrainer._validate = _validate  # type: ignore[method-assign]
    rt.RayPPOTrainer._pinchbench_best_ckpt_patched = True
    print("[pinchbench_best_ckpt] RayPPOTrainer._validate patched (track best + latest only)")
