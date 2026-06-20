#!/usr/bin/env python3
"""Committee pairwise on HELD-OUT Tampa council: base (nomem) vs base+mem. Empties filtered.
'lora' win => mem helps base on that held-out task."""
import os, sys, json
sys.path.insert(0, "/Users/lytton/eval_val3_transcripts")
os.environ.setdefault("JUDGE_LIB_DIR", "/Users/lytton/work/reinforement_learning/timeline/data_agent/experiments")
import committee_judge as cj

def load_side(path):
    d = {}
    for l in open(path):
        if not l.strip(): continue
        r = json.loads(l); d.setdefault(r.get("task_id"), []).append((r.get("response") or "")[:cj.MAXREP])
    return d
nomem = load_side("/tmp/council_nomem.jsonl")
mem   = load_side("/tmp/council_mem.jsonl")
prompts = {t["task_id"]: t["prompt"] for t in json.load(open("/tmp/council_3_nomem.json"))}
TASKS = ["task_meeting_council_votes", "task_meeting_council_upcoming", "task_meeting_council_contact_info"]

def load_reports(tid):
    b = [x for x in nomem.get(tid, []) if len(x) >= 50][:4]
    l = [x for x in mem.get(tid, []) if len(x) >= 50][:4]
    return prompts.get(tid, ""), {"base": b, "lora": l}
cj.load_reports = load_reports

print("HELD-OUT council:  base = no mem  |  lora = base+mem.  'lora' win => mem helps held-out.\n")
for tid in TASKS:
    b = [x for x in nomem.get(tid, []) if len(x) >= 50]; l = [x for x in mem.get(tid, []) if len(x) >= 50]
    print(f"[{tid.replace('task_meeting_council_','')}] valid base={len(b)} mem={len(l)}")
    if len(b) >= 1 and len(l) >= 1 and (len(b) >= 2 or len(l) >= 2):
        cj.show(cj.committee_task(tid, "base", "lora", deliberate=True))
    else:
        print("  SKIP (too few valid — timeout-polluted)\n")
