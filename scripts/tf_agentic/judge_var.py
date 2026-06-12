import os, sys, json, glob, tempfile, shutil
from pathlib import Path
assert os.environ.get("DEEPSEEK_API_KEY"), "set DEEPSEEK_API_KEY (e.g. source ~/.pinchbench_env)"
sys.path.insert(0, "/workspace/openclaw-naive-meeting-analysis-github/scripts")
from lib_tasks import TaskLoader
from lib_grading import grade_task

ASSETS = "/workspace/openclaw-naive-meeting-analysis-github/data/eval/assets"
tasks = {t.task_id: t for t in TaskLoader(Path("/tmp/meeting_analysis_tasks_skilltest")).load_all_tasks()}
task = tasks["task_meeting_advisory_stakeholders"]
fp = glob.glob("/tmp/agentic_val3_lora/0001_transcripts/*advisory*1781247237615.jsonl")[0]
tr = [json.loads(l) for l in open(fp) if l.strip()]

ws = tempfile.mkdtemp()
for o in tr:
    if o.get("type") != "message": continue
    m = o.get("message", o); c = m.get("content")
    for x in (c if isinstance(c, list) else [c]):
        if isinstance(x, dict) and x.get("type") == "toolCall" and x.get("name") in ("write", "create"):
            a = x.get("arguments", {}) or {}
            open(os.path.join(ws, os.path.basename(a.get("path", "r.md"))), "w").write(str(a.get("content", "")))
for wf in getattr(task, "workspace_files", []) or []:
    shutil.copy2(os.path.join(ASSETS, wf["source"]), os.path.join(ws, os.path.basename(wf["dest"])))

JM = sys.argv[1] if len(sys.argv) > 1 else "deepseek-v4-pro"
print("JUDGE MODEL:", JM, flush=True)
scores = []
for i in range(4):
    er = {"status": "success", "transcript": tr, "workspace": ws}
    gr = grade_task(task=task, execution_result=er, judge_model=JM, judge_backend="api",
                    judge_base_url="https://api.deepseek.com/v1", judge_api_key=os.environ["DEEPSEEK_API_KEY"],
                    skill_dir=Path("/tmp"), verbose=False)
    j = {k.split(".")[-1]: v for k, v in gr.breakdown.items() if k.startswith("llm_judge")}
    scores.append(gr.score)
    print("call %d: hybrid=%.3f  judge=%s" % (i, gr.score, j), flush=True)
print("SAME REPORT, %d judge calls: min=%.3f max=%.3f spread=%.3f" % (len(scores), min(scores), max(scores), max(scores) - min(scores)), flush=True)
print("JUDGE_VAR_DONE", flush=True)
