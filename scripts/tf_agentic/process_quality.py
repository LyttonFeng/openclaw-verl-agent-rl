"""Post-rollout PROCESS-QUALITY analysis (informational, not a hard gate).
For each archived trajectory: did the agent save the report (write/create tool)?
how many read-offset errors? how many turns? how much text was drafted in the
message (not saved)? Cross with the final score. Surfaces *why* rollouts failed
and can later inform filtering. Pure transcript analysis — no GPU.

Usage: process_quality.py <transcripts_dir> [graded_trajectories.jsonl]
"""
import json, sys, os, glob, re

TDIR = sys.argv[1] if len(sys.argv) > 1 else "/tmp/nma_round1/val3plus6_w1e/rollouts/transcripts"
GRADED = sys.argv[2] if len(sys.argv) > 2 else "/tmp/nma_round1/val3plus6_w1e/rollouts/graded_trajectories.jsonl"

# scores by (task, resp-ish) — graded keys vary; build a fallback by task list
scores = {}
if os.path.exists(GRADED):
    for l in open(GRADED):
        r = json.loads(l)
        scores.setdefault(r.get("task_id"), []).append(r.get("score"))


def analyze(fp):
    rows = []
    for l in open(fp):
        l = l.strip()
        if l:
            try:
                rows.append(json.loads(l))
            except Exception:
                pass
    used_write = 0
    read_errs = 0
    turns = 0
    msg_chars = 0
    for o in rows:
        if o.get("type") != "message":
            continue
        m = o.get("message", o)
        role = m.get("role")
        c = m.get("content")
        items = c if isinstance(c, list) else [c]
        if role == "assistant":
            turns += 1
        for x in items:
            if isinstance(x, dict):
                if x.get("type") == "toolCall":
                    if x.get("name") in ("write", "create"):
                        used_write += 1
                elif x.get("type") in ("toolResult", "tool_result"):
                    if "beyond end of file" in str(x.get("content", "")) or '"status": "error"' in str(x.get("content", "")):
                        read_errs += 1
                elif role == "assistant" and x.get("text"):
                    msg_chars += len(x["text"])
            elif role == "assistant" and isinstance(x, str):
                msg_chars += len(x)
    return dict(turns=turns, used_write=used_write, read_errs=read_errs, msg_chars=msg_chars)


# only the round's own transcripts (the named-with-timestamp ones, newest per task-resp)
fps = sorted(f for f in glob.glob(TDIR + "/*.jsonl") if "__" not in os.path.basename(f) or True)
# prefer resp-named files
resp_files = sorted(f for f in glob.glob(TDIR + "/*resp[0-9]*.jsonl"))
use = resp_files or fps
print("%-40s turns write read_err msg_chars" % "trajectory")
print("-" * 80)
agg = {}
for f in sorted(use):
    a = analyze(f)
    name = os.path.basename(f).replace("task_meeting_", "").replace(".jsonl", "")[:40]
    flag = "  ⚠NO-WRITE" if a["used_write"] == 0 else ""
    flag += "  ⚠READ-FUMBLE(%d)" % a["read_errs"] if a["read_errs"] >= 2 else ""
    print("%-40s  %3d  %4d   %5d   %7d%s" % (name, a["turns"], a["used_write"], a["read_errs"], a["msg_chars"], flag))
print("\nPROCESS_QUALITY_DONE")
