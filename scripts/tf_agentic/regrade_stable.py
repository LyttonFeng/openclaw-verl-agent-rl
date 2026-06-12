"""Noise-averaged definitive re-judge: for every RUNS=3 trajectory (base+lora,
3 tasks x 3 runs each), judge the SAME report NJUDGE times with deepseek-v4-flash
and average -> per-run stable score; then average the 3 runs -> per-task; then
overall. Beats down the judge's per-call non-determinism. API-only.
"""
import os, sys, json, glob, tempfile, shutil
from pathlib import Path
from statistics import mean, pstdev

assert os.environ.get("DEEPSEEK_API_KEY"), "set DEEPSEEK_API_KEY (e.g. source ~/.pinchbench_env)"
sys.path.insert(0, "/workspace/openclaw-naive-meeting-analysis-github/scripts")
from lib_tasks import TaskLoader
from lib_grading import grade_task

ASSETS = "/workspace/openclaw-naive-meeting-analysis-github/data/eval/assets"
JM = "deepseek-v4-flash"
NJUDGE = 3
TASK_IDS = ["task_meeting_advisory_stakeholders", "task_meeting_gov_speaker_summary", "task_meeting_tech_action_items"]
JKW = dict(judge_model=JM, judge_backend="api", judge_base_url="https://api.deepseek.com/v1",
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
    fs = sorted(glob.glob(f"/tmp/agentic_val3_{mode}/0001_transcripts/{tid}__*.jsonl"), key=os.path.getmtime)
    return fs[-3:]


print(f"judge={JM} njudge={NJUDGE}", flush=True)
final = {}
for mode in ["base", "lora"]:
    print(f"\n===== {mode} =====", flush=True)
    task_means = []
    for tid in TASK_IDS:
        task = tasks[tid]
        run_scores = []
        for fp in newest3(mode, tid):
            ws = build_ws(task, load_tr(fp))
            tr = load_tr(fp)
            js = []
            for _ in range(NJUDGE):
                gr = grade_task(task=task, execution_result={"status": "success", "transcript": tr, "workspace": ws}, **JKW)
                js.append(gr.score)
            run_scores.append(mean(js))
            print("  %-14s run: %.3f  (judge calls=%s)" % (tid.replace("task_meeting_", ""), mean(js), [round(x, 2) for x in js]), flush=True)
            shutil.rmtree(ws, ignore_errors=True)
        tm = mean(run_scores)
        task_means.append(tm)
        print("  >>> %-14s mean=%.3f (n=%d runs)" % (tid.replace("task_meeting_", ""), tm, len(run_scores)), flush=True)
    final[mode] = mean(task_means)
    print("  ===== %s OVERALL = %.3f =====" % (mode, final[mode]), flush=True)
print("\nDELTA lora-base = %+.3f" % (final["lora"] - final["base"]), flush=True)
print("REGRADE_STABLE_DONE", flush=True)
