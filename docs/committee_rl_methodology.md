# Committee-Reward RL for qwen3.5-4b (RULER + rubrics + reference, hybrid with automated)

Meeting-analysis agentic RL on qwen3.5-4b. This doc captures the method, the harness-check
discipline, the reward design, the train/infer architecture, and the environment.

---

## 0. Why transformers+harness for BOTH train and inference (vLLM dropped)

**vLLM online LoRA is a NO-OP on qwen3.5.** vLLM 0.22 serves `Qwen3_5ForConditionalGeneration` but its
`--lora-modules` path silently serves the BASE weights for this arch (verified: vLLM base==lora exact
logprobs; PEFT base vs base+lora differ by ~0.064). So every vLLM-lora eval was invalid.

**Decision: do not rely on vLLM for the LoRA. Use the transformers stack end-to-end:**
- **Inference / rollouts / eval**: a transformers + PEFT OpenAI-compatible shim (`scripts/tf_agentic/tf_shim.py`)
  that loads base + attaches the LoRA adapter (`LORA_ADAPTER` env) and serves an OpenAI `/v1` API the
  agent (OpenClaw) talks to.
- **Training**: PEFT GRPO directly on transformers (`train/train_meeting_grpo_step.py`,
  `--lora-path` to continue-train; logprobs via `train/compute_rollout_logprobs.py` WITH the same adapter).
- **The LoRA itself is kept** (all-linear r32, alpha=64). "Dropped LoRA" = dropped *vLLM's lora serving*,
  not the adapter. Everything runs through transformers because that's the only path where qwen3.5 LoRA
  actually takes effect.

Shim robustness lessons baked in: self-redirect stdout→file (ssh drop can't BrokenPipe it),
`SHIM_DEFAULT_TEMP` (OpenClaw sends temp=None; rollout=1.0 sampling / eval=0 greedy),
`torch.cuda.empty_cache()` after generate, SSE streaming, multi-turn tool_call args as dict.

---

## 1. Harness-check methodology (catch low-level bugs BEFORE trusting any score)

Agentic-RL scores are worthless if the harness silently failed. **A written report that scores 0 is
almost always a harness artifact (timeout / workspace-sync / wrong path), NOT a model failure.** Two
real bugs this cost us a whole on-policy round:
- **Rollout timeout too short** (180×2=360s) for the 71K-char advisory doc → all 4 rollouts timed out →
  2 no-write + 2 wrote-but-killed-before-grade → false `automated_score=0` → advisory filtered out of
  training entirely. (Proof it was harness: re-running the SAME grade() on the now-synced file → 0.78.)
- **Eval served the wrong adapter** (ssh dropped mid-launch → stale shim serving a previous adapter).

**Checks (now automated in `scripts/tf_agentic/rollout_healthcheck.py`, run RIGHT after rollout, BEFORE
scoring/training; exits non-zero on CRITICAL so the driver stops before wasting a train):**
1. per task-group: `timed_out` count, empty/no-write count, automated range.
2. **ALL-TIMEOUT** → timeout too short (raise `ROLLOUT_TIMEOUT_MULT`).
3. **WRITTEN-BUT-AUTO=0** → grading/sync bug (a written report with auto 0 is a red flag).
4. **MOSTLY-NO-WRITE** (≥n-1 empty) → agent not saving / doc too long.
5. before any eval: verify the LIVE shim's adapter via `tr '\0' '\n' </proc/$(pgrep -f tf_shim)/environ | grep LORA_ADAPTER`
   AND confirm transcripts are fresh (mtime = now), not stale leftovers.

Manual one-liners that root-caused the bugs:
```bash
# what adapter is the running shim actually serving?
pid=$(pgrep -f tf_shim|head -1); tr '\0' '\n' </proc/$pid/environ | grep LORA_ADAPTER
# did a written report really score 0? re-run grade() on its workspace:
python3 -c "import json; t=[x for x in json.load(open('val3_plus6_train.json')) if x['task_id']=='...'][0]; ns={}; exec(t['grading']['grade_function'],ns); print(ns['grade'](transcript, workspace_path))"
# per-rollout timeout/empty/auto from graded_trajectories.jsonl (see rollout_healthcheck.py)
```

---

## 2. RULER-based RANK for GRPO (with reference, with rubrics) — the core reward

Built on OpenPipe RULER's insight: **relative (listwise) ranking of a group is easier + more stable for
an LLM judge than absolute scoring, and GRPO only needs within-group relative values.** RULER natively
supports a custom rubric — so we plug in pinchbench's hand-written `llm_rubric`, not just a generic one.

`scripts/tf_agentic/ruler_reward.py` — for each GRPO group (K rollouts of one task):
- **listwise relative scoring**: all K deliverables anonymized + shuffled, scored 0-1 in ONE judge call
  relative to each other.
- **heterogeneous committee** (kills correlated single-model bias): `ds-v4-flash` + `qwen3-max` +
  `MiniMax-M3` (one per family), averaged. Reasoning judges (flash/minimax) need max_tokens headroom +
  `reasoning_content` fallback.
- **rubric**: the task's hand-written `llm_rubric` when present (str or list→normalized), else a generic
  RULER fallback. A thin anti-hallucination + anti-duplicate overlay applies on top either way; the
  "concise>verbose" clause lives ONLY in the generic fallback so it never fights a completeness rubric
  (e.g., advisory). RULER explicitly allows custom rubrics — relative ranking + hand rubric is the
  intended use, not a contradiction.
- **base-model REFERENCE anchor (放法B)**: inject one base-model report for THIS task into the judge
  prompt as a CALIBRATION reference — "score relative to this baseline". It is NOT scored and NOT put in
  the GRPO advantage normalization (only the K policy rollouts are ranked/normalized — preserves
  within-group spread). It anchors the judge's scale across rounds and encodes "beat base".
- caveat: base-ref + diverse on-policy sampling are complementary; the anchor alone won't create
  gradient — within-group spread must come from on-policy temp>0 diversity.

Eval-side uses **pairwise** (A-vs-B, order-consistency debias, committee, null-calibration) via
`scripts/tf_agentic/committee_judge.py` — pairwise for the careful one-off verdict, listwise for the
per-group training reward (cheaper + handles degenerate/empty rollouts better, length-independent).

---

## 3. Hybrid reward: automated + RULER (committee)

The deterministic `automated_score` (regex/entity/structure/file-checks, per-task) and the relative
committee score are **complementary and check each other**:
- automated alone → rewards coverage/length → verbosity hacking (round-1e: tech inflated to auto 0.944).
- committee alone → rewards conciseness/quality → over-shortens tasks that need completeness (advisory).

**Final GRPO reward = `AUTO_W·automated + (1-AUTO_W)·committee`** (ADDITIVE — multiplicative rejected:
too harsh, non-standard, couples component weights). `inject_committee_reward.py` computes committee
(rubric+ref) then blends with the row's `automated_score` and writes it as the GRPO `score`.

**⚠️ automated ↔ committee CONFLICT (empirically established, see committee_reward_ablation.md):** the two
signals conflict on the coverage/verbosity axis. automated = presence/coverage (length-friendly, blind to
padding/duplication/shallow quality); committee = quality/grounding (penalizes padding). They AGREE at the
floor (grounded report vs empty/hallucinated) but CONFLICT at the top (max coverage ≠ max quality).
Proof: AUTO_W 0.5→0.7 (committee_w2→w3) made BOTH committee AND automated WORSE — pushing advisory coverage
dragged gov/tech quality down. **Therefore keep AUTO_W LOW (≤0.3) or use automated as a soft FLOOR/gate, not
a high-weight co-equal term; committee is the primary driver of quality.** AUTO_W=0.5 (committee_w2) is the
best balance found; 0.7 is a dead end.

---

## 4. Train ↔ infer architecture (both transformers + harness)

```
rollout:  tf_shim(base+LoRA, temp=1.0) ←OpenAI/v1← OpenClaw agent → transcripts + automated grade
              │                                                          │
              │ rollout_healthcheck.py (gate)                            │
score:    inject_committee_reward.py: committee(listwise, rubric, base-ref) ⊕ automated → blend score
train:    select variance>thr → compute_rollout_logprobs(WITH adapter) → train_meeting_grpo_step
              (PPO: clip 0.2, kl 0.05; --lora-path to continue-train) → new LoRA
eval:     tf_shim(base+new LoRA, temp=0) → Val3 bench → committee_judge (pairwise) + stable_rejudge
              (automated/hybrid 三口径); ALWAYS verify shim LORA_ADAPTER env + fresh transcripts
```
On-policy iteration = rollout from current policy + continue-train from it (logprobs use that adapter).
Everything is a pod-self-contained nohup script (RunPod ssh is unstable; never host a pipeline over ssh).

---

## 5. Environment

- GPU: 1× NVIDIA A100-SXM4-80GB
- Python 3.10.12; torch 2.11.0+cu130; transformers 5.9.0; peft 0.19.1; vllm 0.22.0 (NOT used for LoRA)
- model: qwen3.5-4b, arch `Qwen3_5ForConditionalGeneration` (served as multimodal-arch text model),
  local path `/tmp/qwen3.5-4b` (NEVER on MFS — MFS silently corrupts large downloads).
- judges (API): deepseek-v4-flash, qwen3-max (DASHSCOPE), MiniMax-M3; keys in `~/.pinchbench_env`
  (pod + mac); unified entry `experiments/lib.py` (`family_chat`/`ds_chat`).
- OpenClaw execution: LOCAL (`PINCHBENCH_FORCE_LOCAL_OPENCLAW=1`; unset OPENCLAW_HOST/ECS_HOST —
  remote ECS needs ssh keys we don't have and breaks workspace sync).

## 6. Scripts (scripts/tf_agentic/ unless noted)
- `tf_shim.py` — transformers+PEFT OpenAI shim (serve base+LoRA).
- `ruler_reward.py` — RULER listwise committee scorer (rubric + base-ref).
- `committee_judge.py` — pairwise committee eval verdict (consistency + null-calibration).
- `inject_committee_reward.py` — committee score ⊕ automated blend → GRPO score.
- `rollout_healthcheck.py` — post-rollout harness gate.
- `retrain_committee.sh` — logprobs (+adapter) → GRPO (cold or continue via INIT_LORA).
- `run_onpolicy.sh` — full on-policy round driver (shim→rollout→healthcheck→inject→train).
- `eval_val3_adapter.sh` / `stable_rejudge.py` — eval rollouts + automated/hybrid dual-column.
- `train/generate_ledger_online_rollouts.py` — rollout gen (timeout via ROLLOUT_TIMEOUT_MULT).
- See `committee_reward_ablation.md` for the full result lineage (committee_w1→blend→w2→w3).
