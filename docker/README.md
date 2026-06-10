# Docker image — naive_meeting_analysis

Reproducible CUDA + torch + vLLM + OpenClaw environment, pinned to the live pod
(probed 2026-06-10). Weights are **not** baked in — mount them at runtime.

## Pinned stack

| Layer | Version |
|-------|---------|
| OS | Ubuntu 22.04 |
| CUDA base | `nvidia/cuda:13.0.1-cudnn-devel-ubuntu22.04` |
| Python | 3.10 |
| torch | 2.11.0+cu130 (from `download.pytorch.org/whl/cu130`) |
| vLLM | 0.22.0 |
| transformers | 5.9.0 · flashinfer 0.6.11 · xformers 0.0.32 |
| OpenClaw | 2026.4.5 (node 22, npm global) |

Full lock: [`requirements.lock.txt`](requirements.lock.txt) (pod `pip freeze`,
with the torch stack and any private accel wheels commented out).

## Why weights are not in the image

`qwen3-4b` is 7.6 GB. Baking it would bloat the image and slow every pull.
Mount it instead, or hf-download into a persistent volume on first boot:

```bash
huggingface-cli download Qwen/Qwen3-4B --local-dir /data/qwen_models/qwen3-4b
```

## Build & push

```bash
# build on a linux/amd64 host (CUDA not needed to build, only to run)
REGISTRY=ghcr.io/lyttonfeng ./build_and_push.sh

# GHCR login first:
echo "$GHCR_PAT" | docker login ghcr.io -u <user> --password-stdin
```

## Run

```bash
# serve the base model on :8021
QWEN_MODELS=/data/qwen_models ./run.sh

# serve with a trained LoRA adapter
QWEN_MODELS=/data/qwen_models \
LORA_ADAPTER=/workspace/repo/results/train/<run>/checkpoint/lora_adapter \
./run.sh

# interactive shell for training / rollouts
./run.sh bash
```

`run.sh` mounts:
- `qwen_models` → `/workspace/qwen_models` (weights)
- this repo → `/workspace/repo` (code + checkpoints)

## Notes / gotchas

- **`devel` base keeps `nvcc`** for flashinfer JIT. If your GPU never JIT-compiles,
  switch the `FROM` to `...-runtime-...` to shrink the image.
- **Private accel wheels** (`tokenspeed-*`, `torch_c_dlpack_ext`) are commented out
  in the lock — they are not on public PyPI. The stack runs without them; install
  only if you have the private index. Expect a small throughput delta.
- **Driver vs CUDA**: the host needs an NVIDIA driver ≥ the one that supports
  CUDA 13.0 (pod ran 580.126.16). The container ships its own CUDA userspace.
- **`--shm-size`**: vLLM/ray need shared memory; `run.sh` sets 16g.
- **Reproduce a training run**: `./run.sh bash`, then run the scripts under
  `train/` (see `train/README.md` on the `naive_meeting_analysis` branch).
