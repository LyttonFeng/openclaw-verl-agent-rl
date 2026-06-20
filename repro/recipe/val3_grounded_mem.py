#!/usr/bin/env python3
"""Val3-GROUNDED mem: for each Val3 task, get a strong-model REF report, diff vs BASE's report,
extract generic transferable hints capturing what base lacks ON VAL3. RED LINE: generic method, no
answers/entities. Prints candidates for review; stores nothing."""
import os, re, json, urllib.request

key=None
for l in open(os.path.expanduser("~/.pinchbench_env")):
    m=re.search(r"DEEPSEEK_API_KEY\s*=\s*[\"']?([^\"'\s]+)",l)
    if m: key=m.group(1); break

def ds(prompt, max_tokens=1200, jmode=False):
    body={"model":"deepseek-chat","messages":[{"role":"user","content":prompt}],"temperature":0.2,"max_tokens":max_tokens}
    if jmode: body["response_format"]={"type":"json_object"}
    req=urllib.request.Request("https://api.deepseek.com/v1/chat/completions",data=json.dumps(body).encode(),
        headers={"Authorization":f"Bearer {key}","Content-Type":"application/json"})
    return json.loads(urllib.request.urlopen(req,timeout=180).read())["choices"][0]["message"]["content"]

TASKS={
 "task_meeting_advisory_stakeholders":("/tmp/val3tx_2012-05-30-meeting-transcript-ntia-csmac.md","stakeholder analysis of an advisory/policy meeting"),
 "task_meeting_gov_speaker_summary":("/tmp/val3tx_2025-07-30-nasa-holds-first-public-meeting-on-ufos-transcript.md","per-speaker summary of a public meeting"),
 "task_meeting_tech_action_items":("/tmp/val3tx_2021-06-28-gitlab-product-marketing-meeting.md","action-item extraction from a team meeting"),
}
prompts={t["task_id"]:t["prompt"] for t in json.load(open("/tmp/val3_3_nomem.json"))}
base={}
for l in open("/tmp/indist_nomem.jsonl"):
    r=json.loads(l); resp=r.get("response") or ""
    if len(resp)>=50: base.setdefault(r["task_id"],resp)

allh={}
for tid,(txf,label) in TASKS.items():
    tx=open(txf).read()[:90000]
    prm=prompts[tid].split("**Optional")[0].strip()
    ref=ds(f"You are an expert analyst. {prm}\n\nTRANSCRIPT:\n{tx}", max_tokens=2500)
    bse=base.get(tid,"(base produced no report)")[:6000]
    diff=ds(
      f"Task: {label}.\nBelow are a STRONG reference report and a WEAKER base-model report for the SAME meeting.\n"
      "Identify what the STRONG report does that the WEAK one misses, and turn it into GENERAL method hints "
      "that would help on ANY meeting of this kind.\n"
      "STRICT: general METHOD only; NO proper names, dates, numbers, quotes, or answers. 3 hints, <=22 words each, imperative.\n"
      'Return ONLY JSON {"hints":["...","...","..."]}\n\n'
      f"STRONG REFERENCE:\n{ref[:6000]}\n\n----\nWEAKER BASE:\n{bse}", max_tokens=500, jmode=True)
    hs=json.loads(diff).get("hints",[])
    allh[tid]=hs
    print(f"\n=== {tid.replace('task_meeting_','')} (ref_len={len(ref)}) ===")
    for i,h in enumerate(hs,1): print(f"  {i}. {h}")
json.dump(allh, open("/tmp/val3_grounded_candidates.json","w"), ensure_ascii=False, indent=2)
print("\n(candidates -> /tmp/val3_grounded_candidates.json; NOT stored)")
