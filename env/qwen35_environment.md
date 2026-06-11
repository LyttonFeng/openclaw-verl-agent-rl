# Qwen3.5-4B environment (naive_ppo_qwen35)

Environment + serving contract for the Qwen3.5-4B non-think pipeline on this pod.
Differs from the Qwen3-4B setup in `training_environment.md` / `benchmark_environment.md`.

## Package versions (pod `/root/openclaw-venv`, 2026-06-11)

```
vllm==0.22.0
transformers==5.9.0
peft==0.19.1
torch==2.11.0+cu130
huggingface_hub==1.17.0
OpenClaw 2026.4.5 (3e72c03)
```

These supersede the old "vLLM 0.10.2 cannot load qwen3_5" constraint: vLLM 0.22
registers `Qwen3_5ForConditionalGeneration`, and `AutoModelForCausalLM`
resolves the checkpoint to the text-only `Qwen3_5ForCausalLM` (vision tower
skipped), so the existing PyTorch/PEFT LoRA training path works unchanged.

## Model

- HF repo `Qwen/Qwen3.5-4B` — `model_type=qwen3_5`, multimodal
  `Qwen3_5ForConditionalGeneration`, Gated DeltaNet hybrid (3 linear : 1 full
  attention, `full_attention_interval=4`), text `max_position_embeddings=262144`
  (256K native, extensible ~1M), `rope_scaling=None`.
- **Weights live on LOCAL disk `/tmp/qwen3.5-4b` ONLY.** Never download/store on
  MFS `/workspace`: it silently corrupts large safetensors (Errno5 short writes)
  and loads ~20x slower (21 min vs ~2 s). A corrupted /workspace copy garbled
  exact token sequences (emitted `meeting/cript.md` for `meeting-transcript.md`)
  and scored 0 on Val3; a clean local `hf download` (hash-verified) fixed it.

## Serving contract (see scripts/start_qwen35_vllm.sh)

```
vllm serve /tmp/qwen3.5-4b --served-model-name qwen35-4b \
  --max-model-len 262144 --gpu-memory-utilization 0.90 --dtype bfloat16 \
  --trust-remote-code --enable-auto-tool-choice --tool-call-parser qwen3_coder
```

- `--tool-call-parser qwen3_coder` — native XML tool calls
  (`<tool_call><function=NAME><parameter=P>...`). NOT hermes; no OpenClaw hermes
  patch.
- **No `--reasoning-parser`** — non-think workflow. With it, the answer is
  misrouted into `message.reasoning` and `content` is empty.
- **No rope flag** — 256K is native.
- Chat template patched to default NON-THINK (scripts/patch_qwen35_template_nothink.py):
  stock template defaults to thinking-ON when `enable_thinking` is omitted, and
  OpenClaw only sends `chat_template_kwargs.enable_thinking` when
  `model.reasoning=true`.

## Benchmark wiring (see scripts/run_val3_bench_qwen35.sh)

- `MODEL=qwen35-4b BASE_URL=http://127.0.0.1:8023/v1`
- `OPENCLAW_MODEL_REASONING=0` (non-think), `PINCHBENCH_MODEL_TEMPERATURE=0`
- Grading judge needs `DEEPSEEK_API_KEY` (source `~/.pinchbench_env`).

## Reference baseline (2026-06-11, fresh local weights, NO training)

| task | Qwen3-4B base | Qwen3-4B C26 (trained) | Qwen3.5-4B non-think |
|---|---|---|---|
| advisory_stakeholders | 0.414 | 0.414 | **0.952** |
| gov_speaker_summary | 0.442 | 0.471 | 0.586 |
| tech_action_items | 0.559 | 0.630 | 0.717 |
| **OVERALL MEETING** | ~47-50% | ~50.5% | **75.2%** |

## Ops note

RunPod kills session processes when the launching ssh disconnects (`setsid`/
`tmux` do not survive). Keep the vLLM server alive by holding one ssh
connection in the foreground (`ServerAliveInterval`), or launch from a
persistent supervisor.
