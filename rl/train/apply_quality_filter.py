#!/usr/bin/env python3
"""
Race-to-bottom 防御：对**正 advantage 样本**做质量过滤。

只过滤 positive-advantage 样本，理由：
  - GRPO 把正样本当"好榜样"训练（imitate）
  - GRPO 把负样本当"避免信号"训练（push away）
  - 质量低的负样本反而是好的对比例，应保留
  - 只有正样本需要质量审查

实证：R3 v1 用 vanilla GRPO 退化到 43.3%（vs R2 46.4%，-3.1pp）。
诊断发现 16 个 useful group 中 4 个是 race-to-bottom（max(reward) < 0.4），
GRPO 仍然按 group-relative 把"两个都差但稍微不那么差"的那条标为正 advantage，
模型学到了 lazy 模式。加上本过滤后 R3 v2 恢复到 46.2%（持平 R2）。

用法：在 select_grpo_samples.py 之后、训练之前插入此过滤。

  python3 rl/train/apply_quality_filter.py \\
      --input  $ROUND_DIR/selection/graded_trajectories_prm_valid.jsonl \\
      --output $ROUND_DIR/selection/graded_trajectories_prm_pos_only.jsonl \\
      --report $ROUND_DIR/selection/quality_report.json
"""
import argparse
import json
import re
import statistics
from collections import defaultdict
from pathlib import Path

# 默认阈值（实测保守值，避免误伤）
DEFAULT_MIN_GROUP_MAX_SCORE   = 0.4   # 整组 max reward 至少 0.4，否则整组扔
DEFAULT_MIN_TOTAL_OUTPUT_CHARS = 500  # final reply + 所有 written file 总字符
DEFAULT_MIN_TOOL_SUCCESS       = 1    # 至少一次成功 tool call


def analyze_transcript(path):
    """从 transcript JSONL 提取质量特征。"""
    final_text = ''
    n_tool_calls = 0
    n_tool_success = 0
    n_tool_errors = 0
    written_files = []  # (path, content_chars)
    try:
        for line in open(path):
            r = json.loads(line)
            if r.get('type') != 'message':
                continue
            m = r['message']
            c = m.get('content', [])
            if not isinstance(c, list):
                continue
            role = m.get('role')
            if role == 'assistant':
                for x in c:
                    if x.get('type') == 'text':
                        final_text = x.get('text', '')
                    elif x.get('type') == 'toolCall':
                        n_tool_calls += 1
                        if x.get('name') in ('write', 'Write'):
                            a = x.get('arguments', {})
                            written_files.append(
                                (a.get('path', '?'), len(a.get('content', '')))
                            )
            elif role == 'toolResult':
                is_err = bool(m.get('isError'))
                if not is_err:
                    for x in c:
                        if isinstance(x, dict) and x.get('isError'):
                            is_err = True
                            break
                if is_err:
                    n_tool_errors += 1
                else:
                    n_tool_success += 1
    except Exception:
        return None
    return {
        'final_text_chars': len(final_text),
        'n_tool_calls': n_tool_calls,
        'n_tool_success': n_tool_success,
        'n_tool_errors': n_tool_errors,
        'written_files': written_files,
    }


def quality_check(rec, group_max, args):
    """对一个正 advantage 样本做质量审查。返回 (pass, reasons, feats)。"""
    reasons = []
    if group_max < args.min_group_max_score:
        reasons.append(f'group_max_too_low ({group_max:.3f}<{args.min_group_max_score})')
    feats = analyze_transcript(rec['transcript_path'])
    if feats is None:
        reasons.append('transcript_unreadable')
        return False, reasons, None
    total_output = feats['final_text_chars'] + sum(ln for _, ln in feats['written_files'])
    if total_output < args.min_total_output_chars:
        reasons.append(f'total_output_too_short ({total_output}<{args.min_total_output_chars})')
    if feats['n_tool_success'] < args.min_tool_success:
        reasons.append(
            f'no_tool_success (calls={feats["n_tool_calls"]} ok={feats["n_tool_success"]})'
        )
    return len(reasons) == 0, reasons, feats


def main():
    ap = argparse.ArgumentParser(description='Race-to-bottom 防御：过滤低质量正 advantage 样本。')
    ap.add_argument('--input', required=True, help='Selection 后的 valid.jsonl')
    ap.add_argument('--output', required=True, help='过滤后输出（直接喂给训练）')
    ap.add_argument('--report', required=True, help='quality_report.json')
    ap.add_argument('--min-group-max-score', type=float, default=DEFAULT_MIN_GROUP_MAX_SCORE)
    ap.add_argument('--min-total-output-chars', type=int, default=DEFAULT_MIN_TOTAL_OUTPUT_CHARS)
    ap.add_argument('--min-tool-success', type=int, default=DEFAULT_MIN_TOOL_SUCCESS)
    ap.add_argument('--prm-reward-gate', type=float, default=None,
                    help='若设置，对 score >= 阈值的 trajectory 把 prm_turn_scores 清零'
                         '（避免 PRM 干扰已及格的 trajectory）。')
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.input)]

    # 按 task_id 分组算 advantage
    groups = defaultdict(list)
    for r in records:
        groups[r['task_id']].append(r)

    for tid, gr in groups.items():
        scores = [g['score'] for g in gr]
        mu = sum(scores) / len(scores)
        sigma = statistics.pstdev(scores)
        gmax = max(scores)
        for g in gr:
            g['_advantage'] = ((g['score'] - mu) / sigma) if sigma > 1e-8 else 0.0
            g['_group_max'] = gmax

    kept = []
    dropped = []
    for r in records:
        adv = r['_advantage']
        if adv <= 0.01:
            kept.append(r)  # 负 / 零 advantage 全部保留
            continue
        passes, reasons, feats = quality_check(r, r['_group_max'], args)
        if passes:
            kept.append(r)
        else:
            dropped.append({
                'task_id': r['task_id'],
                'response_idx': r['response_idx'],
                'score': r['score'],
                'group_max': r['_group_max'],
                'advantage': round(adv, 3),
                'reasons': reasons,
                'feats': feats,
            })

    # 写训练文件（PRM 正向化以兼容下游）
    n_gated = 0
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as f:
        for r in kept:
            out = {k: v for k, v in r.items() if not k.startswith('_')}
            out['prm_turn_scores'] = [
                max(0, int(s)) for s in out.get('prm_turn_scores', [])
            ]
            # Reward gate：score >= 阈值的 trajectory 清零 PRM
            if args.prm_reward_gate is not None and out.get('score', 0.0) >= args.prm_reward_gate:
                out['prm_turn_scores'] = [0] * len(out['prm_turn_scores'])
                out['prm_zeroed_by_reward_gate'] = True
                n_gated += 1
            f.write(json.dumps(out) + '\n')

    if args.prm_reward_gate is not None:
        print(f'PRM reward gate: zeroed PRM for {n_gated}/{len(kept)} trajectories with score >= {args.prm_reward_gate}')

    pos_total = sum(1 for r in records if r['_advantage'] > 0.01)
    neg_total = sum(1 for r in records if r['_advantage'] < -0.01)
    zero_total = len(records) - pos_total - neg_total

    report = {
        'input_total': len(records),
        'kept_total': len(kept),
        'dropped_total': len(dropped),
        'pos_adv_total': pos_total,
        'neg_adv_total': neg_total,
        'zero_adv': zero_total,
        'pos_adv_kept': pos_total - len(dropped),
        'pos_adv_dropped': len(dropped),
        'thresholds': {
            'min_group_max_score': args.min_group_max_score,
            'min_total_output_chars': args.min_total_output_chars,
            'min_tool_success': args.min_tool_success,
        },
        'dropped_records': dropped,
    }
    with open(args.report, 'w') as f:
        json.dump(report, f, indent=2)

    print(f'Input:   {len(records)} (pos={pos_total}, neg={neg_total}, zero={zero_total})')
    print(f'Kept:    {len(kept)} (pos_kept={pos_total - len(dropped)}, neg_kept={neg_total})')
    print(f'Dropped: {len(dropped)} (all positive-adv low-quality)')
    print(f'Report:  {args.report}')


if __name__ == '__main__':
    main()
