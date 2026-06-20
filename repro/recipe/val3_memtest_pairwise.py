#!/usr/bin/env python3
"""Committee pairwise on the Val3 mem-test deliverables: pure-base (mode 'base') vs base+new-mem (mode 'lora').
Reuses committee_judge.py's full protocol (3 heterogeneous members, order-consistency, deliberation, sign-test)
by overriding load_reports to feed deliverables from the memtest rollout JSONLs.
'lora' winning => the new action-item mem HELPS base on that Val3 task."""
import os, sys, json
sys.path.insert(0, "/Users/lytton/eval_val3_transcripts")
os.environ.setdefault("JUDGE_LIB_DIR", "/Users/lytton/work/reinforement_learning/timeline/data_agent/experiments")
import committee_judge as cj

def load_side(path):
    d = {}
    for l in open(path):
        if not l.strip(): continue
        r = json.loads(l); tid = r.get("task_id")
        d.setdefault(tid, []).append((r.get("response") or "")[:cj.MAXREP])
    return d

nomem  = load_side("/tmp/memtest_nomem.jsonl")
newmem = load_side("/tmp/memtest_newmem.jsonl")
prompts = {t["task_id"]: t.get("prompt", "") for t in json.load(open("/tmp/val3_3_nomem.json"))}

def load_reports(tid):
    b = [x for x in nomem.get(tid, []) if len(x) >= 50][:3]
    l = [x for x in newmem.get(tid, []) if len(x) >= 50][:3]
    return prompts.get(tid, ""), {"base": b, "lora": l}
cj.load_reports = load_reports

print("Val3 mem-test:  base = PURE base (no mem)   |   lora = base + NEW action-item mem (gated)")
print("'lora' win => new mem helps base on that task.\n")
for tid in cj.TASKS:
    n_b = len(nomem.get(tid, [])); n_l = len(newmem.get(tid, []))
    print(f"[{tid.replace('task_meeting_','')}] nomem_runs={n_b} newmem_runs={n_l}")
    if n_b and n_l:
        cj.show(cj.committee_task(tid, "base", "lora", deliberate=True))
    else:
        print("  SKIP (missing deliverables)")
