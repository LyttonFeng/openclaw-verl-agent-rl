#!/usr/bin/env python3
"""Distill GENERIC, transferable how-to hints PER Val3 task type from the collaborator's real
in-distribution rollouts (top committee_score responses). RED LINE: generic method only, NO names/
numbers/quotes/answers. Prints candidates for HUMAN REVIEW; stores nothing."""
import os, re, json, urllib.request
from collections import defaultdict

key=None
for l in open(os.path.expanduser("~/.pinchbench_env")):
    m=re.search(r"DEEPSEEK_API_KEY\s*=\s*[\"']?([^\"'\s]+)",l)
    if m: key=m.group(1); break

rows=[json.loads(l) for l in open("/tmp/val3_agent_graded.jsonl")]
g=defaultdict(list)
for r in rows: g[r["task_id"]].append(r)

def tasklabel(tid):
    if "advisory" in tid: return "stakeholder analysis of an advisory/policy meeting"
    if "gov" in tid or "speaker" in tid: return "per-speaker summary of a public/government meeting"
    if "tech" in tid or "action" in tid: return "action-item extraction from a team meeting"
    return "meeting analysis"

def distill(tid, rs):
    rs.sort(key=lambda r: float(r.get("committee_score",0) or 0), reverse=True)
    top=[ (r.get("response") or "")[:2200] for r in rs[:4] ]
    blob="\n\n----- GOOD REPORT -----\n\n".join(top)
    prompt=(
      f"Below are several high-quality example reports for the task: {tasklabel(tid)}.\n"
      "Write short, GENERAL how-to hints that would help an agent do THIS KIND of task well on ANY meeting.\n\n"
      "STRICT RULES (reviewer rejects violations):\n"
      "- GENERAL METHOD only, not facts about these specific meetings.\n"
      "- NO proper names, NO specific dates/numbers, NO organizations, NO quotes, NO answers.\n"
      "- Read like professional best-practice an expert already knows.\n"
      "- 3 hints, each <= 22 words, imperative voice.\n\n"
      'Return ONLY JSON: {"hints":["...","...","..."]}\n\n'
      f"EXAMPLES:\n{blob[:9000]}"
    )
    data=json.dumps({"model":"deepseek-chat","messages":[{"role":"user","content":prompt}],
                     "temperature":0.2,"max_tokens":400,"response_format":{"type":"json_object"}}).encode()
    req=urllib.request.Request("https://api.deepseek.com/v1/chat/completions",data=data,
        headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"})
    txt=json.loads(urllib.request.urlopen(req,timeout=90).read())["choices"][0]["message"]["content"]
    return json.loads(txt).get("hints",[])

SUSP=re.compile(r"\b(NASA|UAP|CSMAC|NTIA|GitLab|spectrum|MHz|Q[1-4]|\d{4})\b",re.I)
allh={}
for tid,rs in g.items():
    hs=distill(tid,rs); allh[tid]=hs
    print(f"\n=== {tid.replace('task_meeting_','')} ===")
    for i,h in enumerate(hs,1):
        flag="  [!] AUTO-FLAG specific" if SUSP.search(h) else ""
        print(f"  {i}. {h}{flag}")
json.dump(allh,open("/tmp/val3_mem_candidates.json","w"),ensure_ascii=False,indent=2)
print("\n(candidates -> /tmp/val3_mem_candidates.json; NOT stored)")
