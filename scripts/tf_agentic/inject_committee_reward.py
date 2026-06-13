"""Replace the GRPO `score` in graded_trajectories with the RULER-style listwise
committee reward (the validated stable, length-independent, anti-hacking reward).

- groups rows by task_id, scores each group's `response` texts relative to each other
  via the heterogeneous committee (ds-flash/qwen-max/minimax-M3), anonymized+shuffled
- writes graded_committee.jsonl with score replaced; keeps score_old/committee_score/
  automated_score for audit.
"""
import sys, os, json
from statistics import mean, pstdev
sys.path.insert(0, os.path.expanduser("~/r1_rollouts"))
import ruler_reward as R
C = R.C

IN = os.path.expanduser("~/r1_rollouts/graded_w1e.jsonl")
OUT = os.path.expanduser("~/r1_rollouts/graded_committee.jsonl")
MAXREP = 6000


def shuffles_for(k):
    base = list(range(k))
    return [base, base[::-1], base[1:] + base[:1]]  # identity, reverse, rotate-1


def main():
    rows = [json.loads(l) for l in open(IN) if l.strip()]
    groups = {}
    for idx, r in enumerate(rows):
        groups.setdefault(r["task_id"], []).append(idx)
    members = list(C.MEMBERS)
    for tid, idxs in groups.items():
        items = [{"dlv": (rows[i].get("response") or "")[:MAXREP], "len": len(rows[i].get("response") or "")} for i in idxs]
        task = rows[idxs[0]].get("prompt", "")
        reward, jm = R.listwise_group(task, items, members, shuffles_for(len(items)))
        rs = [reward[p] for p in range(len(items)) if reward[p] is not None]
        spread = pstdev(rs) if len(rs) > 1 else 0.0
        print(f"=== {tid}  n={len(items)}  spread={spread:.3f} ===", flush=True)
        for p, i in enumerate(idxs):
            new = reward[p] if reward[p] is not None else 0.0
            print(f"  resp_idx={rows[i].get('response_idx')} len={items[p]['len']:6d}  old_score={rows[i].get('score'):.3f} -> committee={new:.3f}")
            rows[i]["score_old"] = rows[i].get("score")
            rows[i]["committee_score"] = new
            rows[i]["score"] = new
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nWROTE {OUT}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
