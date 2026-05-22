#!/usr/bin/env python3
"""LoRA SFT on Qwen3-4B from ChatML-format JSONL.

Usage:
  python train_qwen3_lora.py \
      --data /workspace/verl_port/sft_v1/chatml.jsonl \
      --model /root/.cache/huggingface/hub/models--Qwen--Qwen3-4B/snapshots/.../ \
      --out /workspace/verl_port/sft_ckpt_v1 \
      --epochs 2 --lr 1e-4 --lora-r 32

Loss is masked to assistant-only tokens (incl. `<think>...</think>` if present),
so the model learns DSv4 Flash's reasoning + tool-calling style without copying
user prompts or tool outputs.
"""
import argparse
import json
import os
from pathlib import Path

import torch
from torch.utils.data import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)
from peft import LoraConfig, get_peft_model


IGNORE_INDEX = -100


def load_jsonl(path):
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def build_masked_example(tokenizer, messages, max_len: int):
    """Apply chat template incrementally; tokens emitted in an assistant turn are trainable.

    Strategy: for each prefix ending in assistant, compute templated ids both
    with and without that assistant turn; the diff is what gets labelled.
    """
    input_ids: list[int] = []
    labels: list[int] = []
    prev_len = 0

    n = len(messages)
    for i, m in enumerate(messages):
        prefix = messages[: i + 1]
        # add_generation_prompt only when the *next* expected role is assistant
        # but here we already have the assistant turn included; so no.
        try:
            full = tokenizer.apply_chat_template(
                prefix, tokenize=True, add_generation_prompt=False
            )
        except Exception as e:
            # Some tokenizers may fail on partial; abort sample
            return None
        new_ids = full[prev_len:]
        if m["role"] == "assistant":
            new_labels = list(new_ids)
        else:
            new_labels = [IGNORE_INDEX] * len(new_ids)
        input_ids.extend(new_ids)
        labels.extend(new_labels)
        prev_len = len(full)
        if len(input_ids) > max_len:
            break

    input_ids = input_ids[:max_len]
    labels = labels[:max_len]
    if not any(t != IGNORE_INDEX for t in labels):
        return None
    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": [1] * len(input_ids),
    }


class SFTDataset(Dataset):
    def __init__(self, records, tokenizer, max_len):
        self.examples = []
        n_dropped = 0
        for r in records:
            ex = build_masked_example(tokenizer, r["messages"], max_len)
            if ex is None:
                n_dropped += 1
                continue
            self.examples.append(ex)
        print(f"SFTDataset: kept {len(self.examples)} / {len(records)} (dropped {n_dropped})")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, i):
        return self.examples[i]


class PadCollator:
    def __init__(self, pad_id):
        self.pad_id = pad_id

    def __call__(self, batch):
        max_len = max(len(b["input_ids"]) for b in batch)
        out = {"input_ids": [], "labels": [], "attention_mask": []}
        for b in batch:
            pad = max_len - len(b["input_ids"])
            out["input_ids"].append(b["input_ids"] + [self.pad_id] * pad)
            out["labels"].append(b["labels"] + [IGNORE_INDEX] * pad)
            out["attention_mask"].append(b["attention_mask"] + [0] * pad)
        return {k: torch.tensor(v) for k, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--model", required=True, help="HF model dir")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=64)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--max-len", type=int, default=24000)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--save-steps", type=int, default=50)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--bf16", action="store_true", default=True)
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading tokenizer from {args.model}")
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    print(f"Loading dataset from {args.data}")
    records = load_jsonl(args.data)
    print(f"  {len(records)} raw records")
    ds = SFTDataset(records, tok, args.max_len)

    if len(ds) == 0:
        raise SystemExit("ERROR: no usable training examples")

    print(f"Loading base model from {args.model}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if args.bf16 else torch.float32,
        attn_implementation="flash_attention_2",
        device_map="auto",
        trust_remote_code=True,
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        task_type="CAUSAL_LM",
        bias="none",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    train_args = TrainingArguments(
        output_dir=str(out_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",
        bf16=args.bf16,
        logging_steps=1,
        save_steps=args.save_steps,
        save_total_limit=3,
        save_strategy="steps",
        report_to=[],
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        dataloader_pin_memory=False,
    )

    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=ds,
        data_collator=PadCollator(tok.pad_token_id),
        tokenizer=tok,
    )

    trainer.train()
    model.save_pretrained(str(out_dir / "final_lora"))
    tok.save_pretrained(str(out_dir / "final_lora"))
    print(f"DONE — LoRA saved to {out_dir / 'final_lora'}")


if __name__ == "__main__":
    main()
