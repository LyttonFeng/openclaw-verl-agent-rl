"""Stable re-judge of Val3 eval transcripts (base vs lora), DISCRETE judge
(lib_grading now forces 0/0.5/1.0). Reports TWO columns per task:
  - automated-only  (deterministic, zero judge noise)  -> the stable yardstick
  - hybrid          (automated + discrete-judge rubric) -> richer, low-noise now
Reuses saved eval transcripts; API-only, no GPU.
"""
import os, sys, json, glob, tempfile, shutil
from pathlib import Path
from statistics import mean

os.environ["DEEPSEEK_API_KEY"] = os.environ.get("DEEPSEEK_API_KEY", "")
sys.path.insert(0, "/workspace/openclaw-naive-meeting-analysis-github/scripts")
from lib_tasks import TaskLoader
from lib_grading import grade_task

ASSETS = "/workspace/openclaw-naive-meeting-analysis-github/data/eval/assets"
TASKS = ["task_meeting_advisory_stakeholders", "task_meeting_gov_speaker_summary", "task_meeting_tech_action_items"]
JKW = dict(judge_model="deepseek-v4-flash", judge_backend="api",
           judge_base_url="https://api.deepseek.com/v1",
           judge_api_key=os.environ["DEEPSEEK_API_KEY"], skill_dir=Path("/tmp"))
tasks = {t.task_id: t for t in TaskLoader(Path("/tmp/meeting_analysis_tasks_skilltest")).load_all_tasks()}


def load_tr(fp):
    return [json.loads(l) for l in open(fp) if l.strip()]


def build_ws(task, tr):
    ws = tempfile.mkdtemp()
    for o in tr:
        if o.get("type") != "message": continue
        m = o.get("message", o); c = m.get("content")
        for x in (c if isinstance(c, list) else [c]):
            if isinstance(x, dict) and x.get("type") == "toolCall" and x.get("name") in ("write", "create"):
                a = x.get("arguments", {}) or {}
                open(os.path.join(ws, os.path.basename(a.get("path", "r.md"))), "w").write(str(a.get("content", "")))
    for wf in getattr(task, "workspace_files", []) or []:
        sp = os.path.join(ASSETS, wf["source"])
        if os.path.exists(sp):
            shutil.copy2(sp, os.path.join(ws, os.path.basename(wf["dest"])))
    return ws


def newest3(mode, tid):
    fs = sorted(glob.glob(f"/tmp/eval_val3_{mode}/0001_transcripts/{tid}__*.jsonl"), key=os.path.getmtime)
    return fs[-3:]


res = {}
for mode in ["base", "lora"]:
    res[mode] = {}
    for tid in TASKS:
        task = tasks[tid]
        autos, hybrids = [], []
        for fp in newest3(mode, tid):
            tr = load_tr(fp); ws = build_ws(task, tr)
            try:
                gr = grade_task(task=task, execution_result={"status": "success", "transcript": tr, "workspace": ws}, **JKW)
                au = {k: v for k, v in gr.breakdown.items() if k.startswith("automated")}
                autos.append(mean(au.values()) if au else 0.0)
                hybrids.append(gr.score)
            except Exception as e:
                print("ERR", mode, tid, str(e)[:60])
            shutil.rmtree(ws, ignore_errors=True)
        res[mode][tid] = (mean(autos) if autos else 0, mean(hybrids) if hybrids else 0, len(autos))
        print("[%s] %-20s auto=%.3f hybrid=%.3f (n=%d)" % (mode, tid.replace("task_meeting_", ""), res[mode][tid][0], res[mode][tid][1], res[mode][tid][2]), flush=True)

print("\n==== STABLE (discrete judge) base vs lora ====")
print("%-14s | auto base/lora | hybrid base/lora" % "task")
ab = al = hb = hl = 0
for tid in TASKS:
    b, l = res["base"][tid], res["lora"][tid]
    print("%-14s |  %.3f / %.3f  |  %.3f / %.3f" % (tid.replace("task_meeting_", ""), b[0], l[0], b[1], l[1]))
    ab += b[0]; al += l[0]; hb += b[1]; hl += l[1]
n = len(TASKS)
print("%-14s |  %.3f / %.3f  |  %.3f / %.3f" % ("OVERALL", ab/n, al/n, hb/n, hl/n))
print("AUTO delta lora-base=%+.3f  HYBRID delta=%+.3f" % (al/n - ab/n, hl/n - hb/n))
print("STABLE_REJUDGE_DONE")
