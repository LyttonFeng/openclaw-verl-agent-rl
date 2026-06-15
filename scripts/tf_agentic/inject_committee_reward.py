"""Replace the GRPO `score` in graded_trajectories with a RULER-style listwise
committee reward, optionally BLENDED with the deterministic automated score.

- groups rows by task_id; scores each group's `response` texts RELATIVE to each
  other via the heterogeneous committee (ds-flash/qwen-max/minimax-M3),
  anonymized + shuffled (RULER).
- the committee judges against the task's hand-written llm_rubric when available
  (loaded from TASKS_FILE), else a generic anti-hacking rubric. A thin
  anti-hallucination + anti-duplicate overlay applies on top either way.
- final GRPO score = AUTO_W * automated_score + (1-AUTO_W) * committee_score.
  AUTO_W=0 -> pure committee; AUTO_W=0.5 -> the agreed blend.

env: GRADED_IN, GRADED_OUT, AUTO_W, TASKS_FILE (for llm_rubric),
     RULER_DIR (where ruler_reward.py lives).
"""
import sys, os, json
from statistics import pstdev
sys.path.insert(0, os.environ.get("RULER_DIR", os.path.expanduser("~/r1_rollouts")))
import ruler_reward as R
C = R.C

IN = os.environ.get("GRADED_IN", os.path.expanduser("~/r1_rollouts/graded_w1e.jsonl"))
OUT = os.environ.get("GRADED_OUT", os.path.expanduser("~/r1_rollouts/graded_blend.jsonl"))
TASKS_FILE = os.environ.get("TASKS_FILE", "")
# graded file of BASE-model rollouts -> per-task calibration reference (放法B anchor).
BASE_REF_FILE = os.environ.get("BASE_REF_FILE", "")
AUTO_W = float(os.environ.get("AUTO_W", "0.5"))
MAXREP = 6000


def load_base_refs(path):
    """task_id -> best base-model response (highest automated_score) as calibration anchor."""
    if not path or not os.path.exists(path):
        return {}
    best = {}
    for l in open(path):
        if not l.strip():
            continue
        r = json.loads(l)
        tid = r.get("task_id"); resp = r.get("response") or ""
        a = float(r.get("automated_score") or 0)
        if tid and resp and (tid not in best or a > best[tid][0]):
            best[tid] = (a, resp)
    return {t: v[1] for t, v in best.items()}


def shuffles_for(k):
    base = list(range(k))
    return [base, base[::-1], base[1:] + base[:1]]  # identity, reverse, rotate-1


def load_rubrics(path):
    """task_id -> llm_rubric (str or list). Empty dict if no file."""
    if not path or not os.path.exists(path):
        return {}
    d = json.load(open(path))
    tasks = d if isinstance(d, list) else d.get("tasks", [d])
    out = {}
    for t in tasks:
        if not isinstance(t, dict):
            continue
        tid = t.get("task_id") or t.get("id")
        rub = (t.get("grading", {}) or {}).get("llm_rubric") or t.get("llm_rubric")
        if tid and rub:
            out[tid] = rub
    return out


def main():
    rows = [json.loads(l) for l in open(IN) if l.strip()]
    rubrics = load_rubrics(TASKS_FILE)
    base_refs = load_base_refs(BASE_REF_FILE)
    print(f"[inject] AUTO_W={AUTO_W}  rubrics={len(rubrics)}  base_refs={len(base_refs)}  rows={len(rows)}", flush=True)
    groups = {}
    for idx, r in enumerate(rows):
        groups.setdefault(r["task_id"], []).append(idx)
    members = list(C.MEMBERS)
    for tid, idxs in groups.items():
        items = [{"dlv": (rows[i].get("response") or "")[:MAXREP], "len": len(rows[i].get("response") or "")} for i in idxs]
        task = rows[idxs[0]].get("prompt", "")
        rub = rubrics.get(tid)  # None -> generic fallback inside ruler_reward
        ref = base_refs.get(tid)  # base calibration anchor (放法B); None -> no anchor
        committee, _ = R.listwise_group(task, items, members, shuffles_for(len(items)), rubric=rub, reference=ref)
        finals = []
        print(f"=== {tid}  n={len(items)}  rubric={'llm' if rub else 'generic'}  base_ref={'yes' if ref else 'no'} ===", flush=True)
        for p, i in enumerate(idxs):
            c = committee[p] if committee[p] is not None else 0.0
            a = float(rows[i].get("automated_score") or 0.0)
            blend = AUTO_W * a + (1 - AUTO_W) * c
            finals.append(blend)
            print(f"  resp_idx={rows[i].get('response_idx')} len={items[p]['len']:6d}  auto={a:.3f} committee={c:.3f} -> score={blend:.3f}")
            rows[i]["score_old"] = rows[i].get("score")
            rows[i]["committee_score"] = c
            rows[i]["score"] = blend
        print(f"  group spread={pstdev(finals) if len(finals)>1 else 0.0:.3f}")
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(f"\nWROTE {OUT}  ({len(rows)} rows)")


if __name__ == "__main__":
    main()
