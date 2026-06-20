# Reproduction artifacts — committee-reward RL + mem0 (qwen3.5-4b)

Preserved before the RunPod pod was recycled. Everything needed to rebuild the env and re-run the
committee judge, the transformers (PEFT) backend, and the RL round recipe.

## Layout
```
repro/
├── env/
│   ├── pip_freeze.txt            # exact venv (226 pkgs) — pip install -r
│   └── pinchbench_env.template   # API keys/hosts SCRUBBED — fill in + `source` it
├── judge/
│   ├── committee_judge.py        # EVAL: pairwise committee (ds-chat / qwen3-max / MiniMax-M3),
│   │                             #   order-consistency + deliberation + sign-test. Reproducible test script.
│   └── lib.py                    # judge plumbing: load_key, parse_trajectory, family_chat (temp=0.0)
├── scripts/
│   ├── tf_agentic/
│   │   ├── tf_shim_batched.py    # TRANSFORMERS backend (PEFT LoRA), batched/parallel (SHIM_MAX_BATCH)
│   │   ├── ruler_reward.py       # TRAIN reward: RULER-style relative committee (listwise)
│   │   ├── inject_committee_reward.py  # final GRPO score = AUTO_W*automated + (1-AUTO_W)*committee
│   │   ├── select_active_tasks.py / rollout_healthcheck.py / update_task_health.py
│   │   └── retrain_committee.sh  # one LoRA update (INIT_LORA= cold | =path continue)
│   ├── start_qwen35_vllm.sh      # vLLM serving (BASE only; non-think template, qwen3_coder tools, native 256K)
│   └── download_qwen35_4b.sh     # fetch weights to LOCAL /tmp (NEVER MFS — silent corruption)
├── recipe/
│   ├── run_base_round.sh         # cold-start committee-reward RL round from base (AUTO_W=0 committee-only)
│   ├── recover_gated_r2.sh       # on-policy continue round (drop dead groups)
│   ├── run_he_round.sh           # H-E variant (key-blend reward)
│   ├── captree_gated_3round.sh / captree_gated_r2r3.sh  # 3-round chain (this session)
│   ├── eval_r1_mem.sh            # eval an adapter + mem: rollout then committee pairwise
│   └── *.py                      # mem build/extract + pairwise wrappers (val3_grounded_mem.py, etc.)
```

## Rebuild the env
```bash
python -m venv openclaw-venv && source openclaw-venv/bin/activate
pip install -r repro/env/pip_freeze.txt          # vLLM 0.22, transformers 5.9, peft, mem0ai, faiss, fastembed
cp repro/env/pinchbench_env.template ~/.pinchbench_env   # then fill DEEPSEEK/DASHSCOPE/MINIMAX keys; chmod 600
source ~/.pinchbench_env
bash repro/scripts/download_qwen35_4b.sh         # weights -> /tmp/qwen3.5-4b (LOCAL disk only)
```

## Key facts / gotchas (learned the hard way)
- **Backend**: serve+train qwen3.5 adapters via `tf_shim_batched.py` (transformers+PEFT). vLLM 0.22 online-LoRA
  is a NO-OP on qwen3_5; vLLM is fine for BASE serving only. Don't mix backends within one comparison.
- **Weights on LOCAL disk** (`/tmp`), never the MFS `/workspace` net disk (silent short-write corruption + 20× slower load).
- **Eval temp 0.3, rollout temp 0.7**; `ROLLOUT_TIMEOUT_MULT>=6` for long transcripts or they time out on the slow shim.
- **Healthcheck before training**: `rollout_healthcheck.py` drops all-dead/timeout groups (e.g. NASA-ledger, advisory-at-r2).
- **mem0 red line**: store only generalizable how-to hints, never answers/entities; `infer=False`; human-review.

## Reproduce the committee eval (base vs adapter, or base vs base+mem)
```bash
# 1) generate deliverables for each side (rollout via tf_shim, see recipe/eval_r1_mem.sh)
# 2) committee pairwise:
JUDGE_LIB_DIR=repro/judge EVAL_TRANSCRIPTS_DIR=<dir with base/ and lora/ subdirs> \
  python repro/judge/committee_judge.py        # prints per-task tally + sign-test
```

See `../findings.md` for the full result narrative and `../to_human/experiment_summary_20260620.html` for the summary.
