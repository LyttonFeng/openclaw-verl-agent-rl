#!/usr/bin/env python3
"""
为 PPO 训练计算 P_old：用 rollout 时的 LoRA 跑现有 transcripts 的 forward pass，
保存每个 trainable token 的 log_prob。

为什么不直接让 vLLM 输出 logprobs？
  - vLLM 当前 rollout 不开启 logprobs，改了等于全部 rollout 重跑（贵）
  - 离线重算同一份 LoRA 的 log_probs 等价于 P_old（同模型同输入）
  - bf16 重算 vs vLLM 实际 bf16 输出，per-token log_prob 几乎相同
    （sanity check: 中位 ratio = 1.0000，~1-3% outlier 在 PPO clip 范围外，
    被 PPO clip 自然吸收）

输出格式：JSONL，每行 {task_id, response_idx, n_trainable, trainable_log_probs: [...]}

用法：
  CUDA_VISIBLE_DEVICES=0,1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \\
  python3 rl/train/compute_rollout_logprobs.py \\
      --graded-file $ROUND_DIR/selection/graded_trajectories_prm_pos_only.jsonl \\
      --lora-path   $PREV_ROUND/checkpoint/lora_adapter \\
      --output      $ROUND_DIR/rollout_logprobs.jsonl \\
      --max-seq-length 65536 --rope-scaling-factor 2.0
"""
import argparse
import json
import sys
from pathlib import Path

# 复用训练脚本的 tokenize 逻辑（保证 token_ids/mask 完全一致）
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from train_meeting_grpo_step import (  # noqa: E402
    build_messages_from_transcript,
    build_token_ids_and_mask,
)


def main():
    ap = argparse.ArgumentParser(description='离线算 rollout 时刻的 token log_probs（P_old）')
    ap.add_argument('--graded-file', required=True, help='训练用的 jsonl（含 transcript_path）')
    ap.add_argument('--model-path', default='Qwen/Qwen3-4B')
    ap.add_argument('--lora-path', default=None,
                    help='Rollout 时所用的 LoRA（即 P_old）。空时用 base 模型（适用 R1 从 base 起跑）。')
    ap.add_argument('--max-seq-length', type=int, default=65536)
    ap.add_argument('--rope-scaling-factor', type=float, default=2.0)
    ap.add_argument('--output', required=True, help='输出 JSONL 路径')
    ap.add_argument(
        '--dtype', choices=['bf16', 'fp32'], default='bf16',
        help='bf16 与 vLLM rollout 一致且省显存；fp32 仅在显存允许时使用',
    )
    ap.add_argument(
        '--max-memory-per-gpu', default='75GiB',
        help='device_map="auto" 的每卡上限',
    )
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import PeftModel

    tok = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    dtype_map = {'bf16': torch.bfloat16, 'fp32': torch.float32}
    fp_kwargs = dict(
        torch_dtype=dtype_map[args.dtype],
        trust_remote_code=True,
    )
    n_gpus = torch.cuda.device_count()
    if n_gpus >= 2:
        fp_kwargs['device_map'] = 'auto'
        fp_kwargs['max_memory'] = {i: args.max_memory_per_gpu for i in range(n_gpus)}
    else:
        fp_kwargs['device_map'] = {'': 0}

    if args.rope_scaling_factor and args.rope_scaling_factor > 1.0:
        fp_kwargs['rope_scaling'] = {'type': 'dynamic', 'factor': float(args.rope_scaling_factor)}

    print(f'Loading base model {args.model_path} ({args.dtype}, {n_gpus} GPU)...')
    model = AutoModelForCausalLM.from_pretrained(args.model_path, **fp_kwargs)
    if args.lora_path:
        print(f'Loading LoRA from {args.lora_path}...')
        model = PeftModel.from_pretrained(model, args.lora_path, is_trainable=False)
    else:
        print('No LoRA path provided; computing P_old with base model.')
    model.eval()

    # PEFT wraps the causal LM at model.base_model.model. Plain HF causal LMs
    # also expose .base_model, but that is the bare transformer body, not a
    # CausalLM with .model/.lm_head.
    if args.lora_path and hasattr(model, "base_model") and hasattr(model.base_model, "model"):
        causal_lm = model.base_model.model
    else:
        causal_lm = model
    body = causal_lm.model
    lm_head = causal_lm.lm_head

    records = [json.loads(l) for l in open(args.graded_file)]
    print(f'Computing log_probs for {len(records)} records...')

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out_f = open(args.output, 'w')
    n_done = 0
    with torch.no_grad():
        for r in records:
            tp = r['transcript_path']
            messages = build_messages_from_transcript(tp)
            if len(messages) < 2:
                out_f.write(json.dumps({
                    'task_id': r['task_id'],
                    'response_idx': r['response_idx'],
                    'trainable_log_probs': [],
                    'note': 'too_few_messages',
                }) + '\n')
                continue
            token_ids, response_mask, _ = build_token_ids_and_mask(
                messages, tok, max_length=args.max_seq_length,
            )
            input_ids = torch.tensor([token_ids], dtype=torch.long, device='cuda:0')
            mask = torch.tensor([response_mask], dtype=torch.float32, device='cuda:0')
            targets = input_ids[:, 1:]
            mask_shifted = mask[:, 1:]
            trainable_idx = mask_shifted.bool()
            n_train = int(trainable_idx.sum().item())
            if n_train == 0:
                out_f.write(json.dumps({
                    'task_id': r['task_id'],
                    'response_idx': r['response_idx'],
                    'trainable_log_probs': [],
                    'note': 'no_trainable',
                }) + '\n')
                continue
            body_out = body(input_ids=input_ids)
            hidden = body_out.last_hidden_state[:, :-1, :]
            out_dev = hidden.device
            tidx = trainable_idx.to(out_dev)
            train_hidden = hidden[tidx]
            train_targets = targets.to(out_dev)[tidx]
            train_logits = lm_head(train_hidden)
            logits_dev = train_logits.device
            train_targets = train_targets.to(logits_dev)
            log_probs_full = torch.log_softmax(train_logits.float(), dim=-1)
            token_log_probs = log_probs_full.gather(
                -1, train_targets.unsqueeze(-1)
            ).squeeze(-1)
            log_probs_list = token_log_probs.cpu().tolist()
            out_f.write(json.dumps({
                'task_id': r['task_id'],
                'response_idx': r['response_idx'],
                'n_trainable': n_train,
                'trainable_log_probs': log_probs_list,
            }) + '\n')
            n_done += 1
            if n_done % 5 == 0:
                print(f'  done {n_done}/{len(records)}')

    out_f.close()
    print(f'Wrote {args.output}')


if __name__ == '__main__':
    main()
