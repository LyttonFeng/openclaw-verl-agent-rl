#!/usr/bin/env python3
"""
Offline GRPO training step for meeting_analysis with proper masking.

Takes graded trajectories (JSONL from generate_meeting_rollouts.py),
groups by task_id, computes GRPO advantages, and does one LoRA update.

Masking:
  - mask=1: model thinking, model tool_calls, model text responses (train these)
  - mask=0: system prompt, user messages, tool results (do NOT train these)

GRPO advantage for response i of prompt j:
    advantage_i = (reward_i - mean_reward_j) / max(std_reward_j, 1.0)

Only tasks with reward variance > 0 contribute gradient.

Usage:
    python train_meeting_grpo_step.py \
        --graded-file rollouts/graded_trajectories.jsonl \
        --model-path Qwen/Qwen3-4B \
        --output-dir checkpoints/round_1 \
        --lr 5e-6 --lora-rank 16 --grad-accum-steps 4
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def load_graded_trajectories(graded_file: str) -> list[dict]:
    """Load JSONL file of graded trajectories."""
    records = []
    with open(graded_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def compute_grpo_advantages(records: list[dict]) -> list[dict]:
    """
    Compute GRPO advantages: normalize rewards within each task group.
    Only returns records where variance exists (at least 2 different scores).
    """
    groups = defaultdict(list)
    for r in records:
        groups[r["task_id"]].append(r)

    output = []
    tasks_with_signal = 0
    tasks_without_signal = 0

    for task_id, group in groups.items():
        scores = [r["score"] for r in group]
        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        std_score = max(variance ** 0.5, 1e-6)

        if variance < 1e-8:
            tasks_without_signal += 1
            logger.info(f"  {task_id}: all scores={scores[0]:.3f}, skipping (no variance)")
            continue

        tasks_with_signal += 1
        for r in group:
            r["advantage"] = (r["score"] - mean_score) / max(std_score, 1.0)
            output.append(r)

    logger.info(f"  Tasks with learning signal: {tasks_with_signal}")
    logger.info(f"  Tasks without signal (skipped): {tasks_without_signal}")
    logger.info(f"  Training samples: {len(output)}")
    return output


def build_messages_from_transcript(transcript_path: str) -> list[dict[str, Any]]:
    """
    Extract chat messages from OpenClaw transcript JSONL.

    Returns list of message dicts in OpenAI format:
      [{"role": "user/assistant/tool", "content": "..."}]

    For assistant messages with tool_calls, content includes the thinking + tool call.
    Tool results are returned as separate tool role messages.
    """
    messages = []
    if not transcript_path or not Path(transcript_path).exists():
        return messages

    for line in open(transcript_path):
        entry = json.loads(line)
        if entry.get("type") != "message":
            continue
        msg = entry.get("message", {})
        role = msg.get("role", "")
        content = msg.get("content", [])

        if not isinstance(content, list):
            if content:
                messages.append({"role": role, "content": str(content)})
            continue

        # Build text content for this message
        parts = []
        for part in content:
            if not isinstance(part, dict):
                continue
            ptype = part.get("type", "")
            if ptype == "thinking":
                # Model thinking → include as <think> block
                think_text = part.get("thinking", "")
                if think_text:
                    parts.append(f"<think>\n{think_text}\n</think>")
            elif ptype == "text":
                text = part.get("text", "")
                if text:
                    parts.append(text)
            elif ptype == "toolCall":
                # Model tool call → format as tool_call
                name = part.get("name", "")
                args = part.get("arguments", {})
                if isinstance(args, str):
                    args_str = args
                else:
                    args_str = json.dumps(args, ensure_ascii=False)
                parts.append(f'<tool_call>\n{{"name": "{name}", "arguments": {args_str}}}\n</tool_call>')
            elif ptype == "toolResult":
                # This is a tool result embedded in the same message
                result_text = part.get("content", part.get("text", ""))
                parts.append(str(result_text))

        if parts:
            combined = "\n".join(parts)
            messages.append({"role": role, "content": combined})

    return messages


def build_token_ids_and_mask(
    messages: list[dict],
    tokenizer,
    max_length: int = 32768,
) -> tuple[list[int], list[int], list[int]]:
    """
    Tokenize messages and build response_mask + assistant turn index per token.

    mask=1 for tokens from assistant messages (thinking + tool calls + text)
    mask=0 for tokens from user messages, system prompt, tool results

    turn_index_per_token: 0,1,2,... for tokens belonging to the k-th assistant
    message; -1 for non-assistant tokens. Used by PRM per-turn credit assignment.

    Returns (token_ids, response_mask, turn_index_per_token) all as lists of ints.
    """
    # We need to tokenize each message separately to know boundaries
    all_ids = []
    all_mask = []
    all_turn_idx = []

    # Add BOS if tokenizer has one
    if tokenizer.bos_token_id is not None:
        all_ids.append(tokenizer.bos_token_id)
        all_mask.append(0)
        all_turn_idx.append(-1)

    assistant_turn_counter = 0
    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        # Format as chat turn
        # For Qwen3: <|im_start|>role\ncontent<|im_end|>\n
        turn_text = f"<|im_start|>{role}\n{content}<|im_end|>\n"
        turn_ids = tokenizer.encode(turn_text, add_special_tokens=False)

        # Determine mask: 1 for assistant, 0 for others
        if role == "assistant":
            mask_value = 1  # Train on model's output
            turn_idx_value = assistant_turn_counter
            assistant_turn_counter += 1
        else:
            mask_value = 0  # Don't train on user/system/tool
            turn_idx_value = -1

        all_ids.extend(turn_ids)
        all_mask.extend([mask_value] * len(turn_ids))
        all_turn_idx.extend([turn_idx_value] * len(turn_ids))

    # Truncate
    if len(all_ids) > max_length:
        all_ids = all_ids[:max_length]
        all_mask = all_mask[:max_length]
        all_turn_idx = all_turn_idx[:max_length]

    return all_ids, all_mask, all_turn_idx


def train_grpo_step(
    records: list[dict],
    model_path: str,
    lora_path: str | None,
    output_dir: str,
    lr: float,
    lora_rank: int,
    grad_accum_steps: int,
    max_seq_length: int,
    rope_scaling_factor: float | None = None,
    prm_alpha: float = 1.0,
    prm_beta: float = 0.0,
    prm_mode: str = "additive",
    old_logprobs_file: str | None = None,
    clip_eps: float = 0.2,
    kl_beta: float = 0.0,
):
    """
    Do one GRPO training step with proper masking.

    Per-token advantage modes:
      - additive (default):
            a_token = α·terminal_adv + β·turn_score[turn_idx]
      - multiplicative ("mult_b"):
            if terminal_adv > 0:
                a_token = terminal_adv · (1 + β·turn_score[turn_idx])
            else:
                a_token = terminal_adv         # PRM does NOT touch failures
        Rationale: PRM sits as a multiplier on already group-baselined
        terminal_adv, sidestepping the additive bias problem (PRM has no
        natural per-position group baseline because turn counts vary).

    Loss = -mean_over_trainable_tokens(a_token * log_prob(token)) / grad_accum

    If a record lacks `prm_turn_scores`, the PRM contribution is 0 for that
    sample (additive: +0; mult_b: factor 1).
    """
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, PeftModel

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    lora_output = output_path / "lora_adapter"

    logger.info(f"  Loading model: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    from_pretrained_kwargs: dict[str, Any] = dict(
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        device_map={"": 0},
    )
    if rope_scaling_factor is not None and rope_scaling_factor > 1.0:
        # Match inference-side rope-scaling so train/infer stay consistent.
        # 'dynamic' YARN: kicks in only past the base context window.
        rope_scaling_cfg = {"type": "dynamic", "factor": float(rope_scaling_factor)}
        from_pretrained_kwargs["rope_scaling"] = rope_scaling_cfg
        logger.info(f"  rope_scaling enabled: {rope_scaling_cfg}")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        **from_pretrained_kwargs,
    )

    # Apply LoRA
    if lora_path and Path(lora_path).exists():
        logger.info(f"  Loading existing LoRA: {lora_path}")
        model = PeftModel.from_pretrained(model, lora_path, is_trainable=True)
    else:
        logger.info(f"  Creating new LoRA (rank={lora_rank})")
        lora_config = LoraConfig(
            r=lora_rank,
            lora_alpha=32,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)

    model.train()
    model.enable_input_require_grads()  # Required for LoRA + gradient checkpointing
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        weight_decay=0.01,
    )

    # Load old policy log_probs (for PPO ratio + KL). If None, vanilla PG.
    old_logprobs_map = {}
    if old_logprobs_file:
        with open(old_logprobs_file) as f:
            for line in f:
                rec = json.loads(line)
                key = (rec['task_id'], rec['response_idx'])
                old_logprobs_map[key] = rec.get('trainable_log_probs', [])
        logger.info(f"  Loaded old logprobs for {len(old_logprobs_map)} rollouts (PPO mode)")
        logger.info(f"  PPO clip_eps={clip_eps} kl_beta={kl_beta}")
    else:
        logger.info("  No old_logprobs_file → vanilla PG (no ratio, no KL)")

    # ── Training loop (batch_size=1, gradient accumulation) ──────────────

    total_loss = 0.0
    total_kl = 0.0
    n_steps = 0
    n_skipped = 0
    optimizer.zero_grad()

    for i, record in enumerate(records):
        advantage = record["advantage"]

        # Skip near-zero advantages
        if abs(advantage) < 0.01:
            n_skipped += 1
            continue

        # Build messages from transcript
        transcript_path = record.get("transcript_path", "")
        messages = build_messages_from_transcript(transcript_path)
        if len(messages) < 2:
            logger.warning(f"  Sample {i}: too few messages ({len(messages)}), skipping")
            n_skipped += 1
            continue

        # Tokenize with masking + per-token assistant turn index
        token_ids, response_mask, turn_idx_per_token = build_token_ids_and_mask(
            messages, tokenizer, max_length=max_seq_length,
        )

        # Check if there are any trainable tokens
        n_trainable = sum(response_mask)
        if n_trainable < 5:
            logger.warning(f"  Sample {i}: only {n_trainable} trainable tokens, skipping")
            n_skipped += 1
            continue

        # ── Build per-token advantage (additive or multiplicative B) ────────
        # Number of assistant turns ACTUALLY present in tokenized messages
        # (may be less than len(prm_turn_scores) if truncation cut some turns,
        # or if the parser dropped empty messages).
        n_assistant_turns = max(turn_idx_per_token) + 1 if any(t >= 0 for t in turn_idx_per_token) else 0
        prm_turn_scores = record.get("prm_turn_scores", []) or []

        if prm_mode == "multiplicative":
            # Multiplicative B: PRM as factor on already-baselined terminal_adv.
            # Only applied when terminal_adv > 0 (don't let PRM rescue/punish
            # failures; failures keep their negative gradient untouched).
            adv_f = float(advantage)
            if adv_f > 0:
                per_turn_factor = []
                for k in range(n_assistant_turns):
                    ts = prm_turn_scores[k] if k < len(prm_turn_scores) else 0
                    per_turn_factor.append(1.0 + prm_beta * float(ts))
                per_token_adv_list = [
                    adv_f * (per_turn_factor[t] if 0 <= t < n_assistant_turns else 1.0)
                    for t in turn_idx_per_token
                ]
            else:
                # Failure rollouts: terminal-only gradient, PRM ignored
                per_token_adv_list = [adv_f for _ in turn_idx_per_token]
        else:
            # Additive (legacy): a_token = α·terminal_adv + β·turn_score[k]
            per_turn_extra = []
            for k in range(n_assistant_turns):
                ts = prm_turn_scores[k] if k < len(prm_turn_scores) else 0
                per_turn_extra.append(prm_beta * float(ts))
            terminal_term = prm_alpha * float(advantage)
            per_token_adv_list = [
                terminal_term + (per_turn_extra[t] if 0 <= t < n_assistant_turns else 0.0)
                for t in turn_idx_per_token
            ]

        # Convert to tensors
        input_ids = torch.tensor([token_ids], dtype=torch.long, device="cuda")
        mask = torch.tensor([response_mask], dtype=torch.float32, device="cuda")
        per_token_adv = torch.tensor(
            [per_token_adv_list], dtype=torch.float32, device="cuda"
        )  # [B, T]

        # Forward pass — at long context (>32K) we MUST avoid materializing
        # the [T, V] logits tensor. For Qwen3-4B with V≈152K, T=75K → 22 GB
        # in bf16 just for `outputs.logits`, and the backward pass needs
        # another 22 GB for the gradient — OOMs a 80 GB GPU.
        #
        # Trick: run only the transformer body (no lm_head), gather the
        # trainable hidden states (mask is sparse — assistant tokens only),
        # then apply lm_head + cross_entropy on that small slice.
        targets = input_ids[:, 1:]           # shifted targets [B, T-1]
        mask_shifted = mask[:, 1:]           # align mask [B, T-1]
        adv_shifted = per_token_adv[:, 1:]   # align advantage to targets [B, T-1]
        trainable_idx = mask_shifted.bool()  # [B, T-1] bool
        n_train = int(trainable_idx.sum().item())
        if n_train == 0:
            n_skipped += 1
            continue

        # PEFT wraps the underlying Qwen3ForCausalLM at .base_model.model.
        # Its .model attribute is the transformer body (returns hidden_states),
        # and .lm_head projects hidden -> vocab.
        causal_lm = model.base_model.model
        body = causal_lm.model
        lm_head = causal_lm.lm_head

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            body_out = body(input_ids=input_ids)
            hidden = body_out.last_hidden_state[:, :-1, :]  # [B, T-1, H]
            train_hidden = hidden[trainable_idx]            # [N_train, H]
            train_targets = targets[trainable_idx]          # [N_train]
            train_advantages = adv_shifted[trainable_idx]   # [N_train]
            # lm_head only on the (sparse) trainable positions: [N_train, V]
            train_logits = lm_head(train_hidden)
            # Fused log_softmax + nll, reduction='none' → no full softmax
            # tensor allocated; we cast logits to fp32 for numerical safety.
            neg_log_probs = torch.nn.functional.cross_entropy(
                train_logits.float(), train_targets, reduction="none"
            )
            token_log_probs = -neg_log_probs                # [N_train] under π_θ

            # ── PPO: ratio + clip + KL（启用 --logprobs-file 时） ────────────
            kl_value = torch.tensor(0.0, device=token_log_probs.device)
            if old_logprobs_file:
                key = (record['task_id'], record['response_idx'])
                old_lp_list = old_logprobs_map.get(key)
                if old_lp_list is None or len(old_lp_list) != token_log_probs.shape[0]:
                    logger.warning(
                        f"  Sample {i} ({key}): old_logprobs missing or len mismatch "
                        f"({len(old_lp_list) if old_lp_list else 'None'} vs {token_log_probs.shape[0]}); fallback to vanilla"
                    )
                    loss = -(train_advantages * token_log_probs).mean() / grad_accum_steps
                else:
                    old_log_probs = torch.tensor(old_lp_list, dtype=torch.float32, device=token_log_probs.device)
                    log_ratio = (token_log_probs - old_log_probs).clamp(-20.0, 20.0)
                    ratio = torch.exp(log_ratio)
                    surr1 = ratio * train_advantages
                    surr2 = ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * train_advantages
                    pg_loss = -torch.min(surr1, surr2).mean()
                    # KL k3 estimator: KL(π_θ || π_ref) where ref=old in our setup.
                    # k3 = exp(log_ratio_ref) - log_ratio_ref - 1, where
                    # log_ratio_ref = log_p_ref - log_p_θ = -log_ratio
                    kl_value = (torch.exp(-log_ratio) + log_ratio - 1.0).mean()
                    loss = (pg_loss + kl_beta * kl_value) / grad_accum_steps
            else:
                # Vanilla policy gradient (legacy path)
                loss = -(train_advantages * token_log_probs).mean() / grad_accum_steps

        loss.backward()

        step_in_accum = (i - n_skipped) % grad_accum_steps
        if step_in_accum == grad_accum_steps - 1 or i == len(records) - 1:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            n_steps += 1

        total_loss += loss.item() * grad_accum_steps
        total_kl += float(kl_value.item()) if old_logprobs_file else 0.0

        if (i + 1) % 5 == 0:
            n_pos_turns = sum(1 for s in prm_turn_scores if s > 0)
            n_neg_turns = sum(1 for s in prm_turn_scores if s < 0)
            kl_str = f" kl={float(kl_value.item()):.4f}" if old_logprobs_file else ""
            logger.info(
                f"    sample {i+1}/{len(records)}: loss={loss.item()*grad_accum_steps:.4f}{kl_str} "
                f"adv={advantage:.3f} tokens={len(token_ids)} "
                f"trainable={n_trainable} ({n_trainable/len(token_ids)*100:.0f}%) "
                f"prm_turns=+{n_pos_turns}/-{n_neg_turns}/{n_assistant_turns}"
            )

    # Final step if leftover gradients
    if (len(records) - n_skipped) % grad_accum_steps != 0:
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        optimizer.zero_grad()
        n_steps += 1

    # ── Save LoRA ────────────────────────────────────────────────────────

    avg_loss = total_loss / max(len(records) - n_skipped, 1)
    logger.info(f"  Training done: {n_steps} optimizer steps, {n_skipped} skipped, avg_loss={avg_loss:.4f}")

    model.save_pretrained(str(lora_output))
    tokenizer.save_pretrained(str(lora_output))
    logger.info(f"  LoRA saved: {lora_output}")

    meta = {
        "n_records": len(records),
        "n_skipped": n_skipped,
        "n_optimizer_steps": n_steps,
        "avg_loss": avg_loss,
        "lr": lr,
        "lora_rank": lora_rank,
        "grad_accum_steps": grad_accum_steps,
        "max_seq_length": max_seq_length,
        "prm_alpha": prm_alpha,
        "prm_beta": prm_beta,
        "prm_mode": prm_mode,
        "ppo_enabled": bool(old_logprobs_file),
        "clip_eps": clip_eps if old_logprobs_file else None,
        "kl_beta": kl_beta if old_logprobs_file else None,
        "avg_kl": (total_kl / max(len(records) - n_skipped, 1)) if old_logprobs_file else None,
    }
    (output_path / "training_meta.json").write_text(json.dumps(meta, indent=2))


def main():
    parser = argparse.ArgumentParser(description="GRPO training step with masking")
    parser.add_argument("--graded-file", required=True)
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--lora-path", default="", help="Existing LoRA to continue from")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--lr", type=float, default=5e-6)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=4)
    parser.add_argument("--max-seq-length", type=int, default=32768)
    parser.add_argument(
        "--rope-scaling-factor",
        type=float,
        default=None,
        help="If set (e.g. 2.0), enables dynamic YARN rope_scaling on HF model load. "
             "Must match the vLLM rollout side to keep train/infer consistent.",
    )
    parser.add_argument(
        "--prm-alpha",
        type=float,
        default=1.0,
        help="Coefficient on terminal advantage in per-token advantage formula: "
             "a_token = alpha*terminal_adv + beta*turn_score[k].",
    )
    parser.add_argument(
        "--prm-beta",
        type=float,
        default=0.0,
        help="Coefficient on per-turn process score (default 0 = terminal-only). "
             "Roadmap PRM v1 spec: beta=0.1 with turn_score in {-1, 0, +1}.",
    )
    parser.add_argument(
        "--prm-mode",
        type=str,
        default="additive",
        choices=["additive", "multiplicative"],
        help="How to fold PRM into per-token advantage. "
             "additive: a_token = α·terminal + β·prm[k]. "
             "multiplicative: a_token = terminal · (1 + β·prm[k]) when terminal>0 "
             "else terminal (PRM does not touch failures).",
    )
    parser.add_argument(
        "--logprobs-file",
        type=str,
        default=None,
        help="JSONL of token log_probs from the rollout-time policy (P_old). "
             "When provided, switches loss from vanilla PG to PPO with ratio + clip. "
             "In our setup ref=old, so KL(π_θ || π_old) is also taken from the same file. "
             "Generate via rl/train/compute_rollout_logprobs.py.",
    )
    parser.add_argument("--clip-eps", type=float, default=0.2, help="PPO clip epsilon (default 0.2)")
    parser.add_argument("--kl-beta", type=float, default=0.0, help="KL penalty coefficient (default 0=off; recommend 0.02 when PPO enabled)")
    args = parser.parse_args()

    logger.info("Loading graded trajectories...")
    records = load_graded_trajectories(args.graded_file)
    logger.info(f"  Loaded {len(records)} trajectories")

    logger.info("Computing GRPO advantages...")
    advantaged = compute_grpo_advantages(records)

    if not advantaged:
        logger.error("No training signal (all tasks have zero variance). Exiting.")
        sys.exit(1)

    logger.info("Training GRPO step...")
    train_grpo_step(
        records=advantaged,
        model_path=args.model_path,
        lora_path=args.lora_path or None,
        output_dir=args.output_dir,
        lr=args.lr,
        lora_rank=args.lora_rank,
        grad_accum_steps=args.grad_accum_steps,
        max_seq_length=args.max_seq_length,
        rope_scaling_factor=args.rope_scaling_factor,
        prm_alpha=args.prm_alpha,
        prm_beta=args.prm_beta,
        prm_mode=args.prm_mode,
        old_logprobs_file=args.logprobs_file,
        clip_eps=args.clip_eps,
        kl_beta=args.kl_beta,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
