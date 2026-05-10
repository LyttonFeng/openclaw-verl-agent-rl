"""Monkey-patch veRL actor checkpointing to save only LoRA adapters.

veRL saves full FSDP actor shards before writing ``actor/lora_adapter``. On
small RunPod disks or flaky network volumes, that can leave a 15GB partial
checkpoint without the LoRA files needed for inference. For LoRA-only
experiments, the adapter is the artifact we need, so this patch skips full actor
state/optimizer saves and writes only ``adapter_model.safetensors`` plus
``adapter_config.json``.

Enable with ``PINCHBENCH_LORA_ONLY_CKPT=1``.
"""

from __future__ import annotations

import shutil
from pathlib import Path


def _prune_old_global_steps(actor_local_path: str, max_ckpt_to_keep) -> None:
    try:
        keep = int(max_ckpt_to_keep)
    except (TypeError, ValueError):
        return
    if keep <= 0:
        return

    actor_dir = Path(actor_local_path).resolve()
    cur_dir = actor_dir.parent
    root = cur_dir.parent
    if not cur_dir.name.startswith("global_step_") or not root.is_dir():
        return

    def _step(path: Path) -> int:
        try:
            return int(path.name.removeprefix("global_step_"))
        except ValueError:
            return -1

    step_dirs = sorted(
        [p for p in root.iterdir() if p.is_dir() and p.name.startswith("global_step_")],
        key=_step,
    )
    stale = [p for p in step_dirs if p != cur_dir][:-max(keep - 1, 0)] if keep > 1 else [p for p in step_dirs if p != cur_dir]
    for path in stale:
        shutil.rmtree(path, ignore_errors=True)
        print(f"[pinchbench_lora_only_ckpt] pruned old checkpoint: {path}")


def apply_patch() -> None:
    from verl.single_controller.base.decorator import MAGIC_ATTR
    from verl.workers import fsdp_workers as fw
    from verl.single_controller.ray.base import RayWorkerGroup

    if not hasattr(RayWorkerGroup, "save_checkpoint"):
        def wg_save_checkpoint(self, *args, **kwargs):
            return self.execute_all_sync("save_checkpoint", *args, **kwargs)

        RayWorkerGroup.save_checkpoint = wg_save_checkpoint
        print("[pinchbench_lora_only_ckpt] RayWorkerGroup.save_checkpoint patched for veRL compatibility")

    cls = fw.ActorRolloutRefWorker
    if getattr(cls, "_pinchbench_lora_only_ckpt_patched", False):
        return

    orig = cls.save_checkpoint

    def save_lora_only(self, local_path, hdfs_path=None, global_step=0, max_ckpt_to_keep=None):
        # Non-LoRA actors should retain veRL's native full checkpoint behavior.
        peft_model = getattr(self, "actor_module", getattr(self, "actor_module_fsdp", None))
        if not (getattr(self, "_is_actor", False) and getattr(self, "_is_lora", False) and hasattr(peft_model, "peft_config")):
            return orig(self, local_path, hdfs_path=hdfs_path, global_step=global_step, max_ckpt_to_keep=max_ckpt_to_keep)

        if local_path is None:
            return

        if getattr(self, "_is_offload_param", False):
            fw.load_fsdp_model_to_gpu(self.actor_module_fsdp)

        local_path = Path(local_path)
        lora_save_path = local_path / "lora_adapter"
        peft_config = {}

        if fw.dist.get_rank() == 0:
            lora_save_path.mkdir(parents=True, exist_ok=True)
            peft_config = fw.asdict(peft_model.peft_config.get("default", {}))
            peft_config["task_type"] = peft_config["task_type"].value
            peft_config["peft_type"] = peft_config["peft_type"].value
            peft_config["target_modules"] = list(peft_config["target_modules"])

        try:
            if fw.fsdp_version(self.actor_module_fsdp) > 0:
                self.actor_module_fsdp = self.actor_module_fsdp.to(fw.get_device_name())
                rollout_cfg = getattr(getattr(self, "config", None), "rollout", {})
                rollout_layered_summon = str(rollout_cfg.get("layered_summon", False)).lower()
                lora_params = fw.collect_lora_params(
                    self.actor_module_fsdp,
                    # Save-time LoRA collection is more reliable via full-params summon.
                    # rollout.layered_summon is a transport/runtime optimization and can
                    # return zero tensors here under FSDP offload.
                    layered_summon=False,
                    base_sync_done=True,
                )
                if fw.dist.get_rank() == 0:
                    if not lora_params:
                        raise RuntimeError(
                            "LoRA-only checkpoint collected zero LoRA tensors "
                            f"(rollout.layered_summon={rollout_layered_summon}); "
                            "refusing to write an empty adapter_model.safetensors"
                        )
                    fw.save_file(lora_params, str(lora_save_path / "adapter_model.safetensors"))
                    with (lora_save_path / "adapter_config.json").open("w", encoding="utf-8") as f:
                        fw.json.dump(peft_config, f, ensure_ascii=False, indent=4)
                    print(f"[pinchbench_lora_only_ckpt] saved {len(lora_params)} LoRA tensors")
        except Exception as e:
            print(f"[pinchbench_lora_only_ckpt] Save LoRA adapter error: {e}")
            raise

        fw.dist.barrier()
        if fw.dist.get_rank() == 0:
            _prune_old_global_steps(str(local_path), max_ckpt_to_keep)
            print(f"[pinchbench_lora_only_ckpt] saved LoRA adapter only: {lora_save_path}")

        if getattr(self, "_is_offload_param", False):
            fw.offload_fsdp_model_to_cpu(self.actor_module_fsdp)

    if hasattr(orig, MAGIC_ATTR):
        setattr(save_lora_only, MAGIC_ATTR, getattr(orig, MAGIC_ATTR))

    cls.save_checkpoint = save_lora_only
    cls._pinchbench_lora_only_ckpt_patched = True
    print("[pinchbench_lora_only_ckpt] ActorRolloutRefWorker.save_checkpoint patched (LoRA only)")
