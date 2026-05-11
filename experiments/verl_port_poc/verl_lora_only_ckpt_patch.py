"""LoRA-only checkpoint patch for veRL 0.8 (engine refactor).

Background: veRL 0.7 had ActorRolloutRefWorker.save_checkpoint in
verl/workers/fsdp_workers.py. The pinchbench-skill patch
(rl/verl_lora_only_ckpt_patch.py on the main branch) targets that path.

veRL 0.8.0.dev0 moved the FSDP save into
verl/workers/engine/fsdp/transformer_impl.py::FSDPEngine.save_checkpoint.
This file is the 0.8-compatible re-write — same idea (skip 8 GB FSDP
shards, write only adapter_model.safetensors + adapter_config.json), new
target class.

Motivation: MooseFS on RunPod truncates 8 GB .pt files (EOCD missing /
invalid header), corrupting every full FSDP checkpoint. LoRA-only saves
are 130 MB which write atomically and survive the FS.

All output goes to stderr — the bash launch scripts run `python -c "..."`
device-detection subprocesses and capture stdout, so any stdout pollution
breaks DEVICE=$(...) parsing.

Enable with PINCHBENCH_LORA_ONLY_CKPT=1.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import asdict
from pathlib import Path


def _enabled() -> bool:
    return os.environ.get("PINCHBENCH_LORA_ONLY_CKPT", "").strip().lower() in {"1", "true", "yes", "on"}


def _log(msg: str) -> None:
    print(f"[lora_only_ckpt] {msg}", file=sys.stderr)


def apply_patch() -> None:
    if not _enabled():
        return
    try:
        from verl.workers.engine.fsdp.transformer_impl import FSDPEngine
        from verl.utils.fsdp_utils import (
            collect_lora_params,
            load_fsdp_model_to_gpu,
            offload_fsdp_model_to_cpu,
        )
        import torch
        import torch.distributed as dist
        from safetensors.torch import save_file
    except Exception as e:
        _log(f"import skipped: {e}")
        return

    if getattr(FSDPEngine, "_pinchbench_lora_only_patched", False):
        return

    orig = FSDPEngine.save_checkpoint

    def save_checkpoint(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None, **kwargs):
        if not getattr(self, "_is_lora", False):
            return orig(self, local_path, hdfs_path=hdfs_path, global_step=global_step,
                        max_ckpt_to_keep=max_ckpt_to_keep, **kwargs)

        if self._is_offload_param:
            load_fsdp_model_to_gpu(self.module)

        peft_model = getattr(self.module, "_fsdp_wrapped_module", self.module)
        peft_config = {}
        if dist.get_rank() == 0 and hasattr(peft_model, "peft_config"):
            cfg = peft_model.peft_config.get("default", None)
            if cfg is not None:
                peft_config = asdict(cfg)
                if hasattr(peft_config.get("task_type"), "value"):
                    peft_config["task_type"] = peft_config["task_type"].value
                if hasattr(peft_config.get("peft_type"), "value"):
                    peft_config["peft_type"] = peft_config["peft_type"].value
                if isinstance(peft_config.get("target_modules"), set):
                    peft_config["target_modules"] = list(peft_config["target_modules"])

        lora_params = collect_lora_params(
            module=self.module,
            layered_summon=False,
            base_sync_done=True,
        )

        if dist.get_rank() == 0:
            out = Path(local_path) / "lora_adapter"
            out.mkdir(parents=True, exist_ok=True)
            if not lora_params:
                raise RuntimeError("collected zero LoRA tensors; refusing to write empty adapter")
            cpu = {k: v.detach().contiguous().cpu() for k, v in lora_params.items()}
            save_file(cpu, str(out / "adapter_model.safetensors"))
            with (out / "adapter_config.json").open("w", encoding="utf-8") as f:
                json.dump(peft_config, f, ensure_ascii=False, indent=2)
            _log(f"step={global_step} wrote {len(lora_params)} LoRA tensors -> {out}")

        dist.barrier()
        if self._is_offload_param:
            offload_fsdp_model_to_cpu(self.module)

        if dist.get_rank() == 0 and max_ckpt_to_keep:
            try:
                keep = int(max_ckpt_to_keep)
            except (TypeError, ValueError):
                keep = 0
            if keep > 0:
                cur = Path(local_path).resolve()
                root = cur.parent.parent if cur.name == "actor" else cur.parent
                steps = sorted(
                    [p for p in root.iterdir() if p.is_dir() and p.name.startswith("global_step_")],
                    key=lambda p: int(p.name.removeprefix("global_step_")) if p.name.removeprefix("global_step_").isdigit() else -1,
                )
                stale = steps[:-keep]
                for path in stale:
                    if path.resolve() != cur.parent.resolve():
                        import shutil
                        shutil.rmtree(path, ignore_errors=True)
                        _log(f"pruned: {path}")

    FSDPEngine.save_checkpoint = save_checkpoint
    FSDPEngine._pinchbench_lora_only_patched = True
    _log("FSDPEngine.save_checkpoint patched (LoRA only)")
