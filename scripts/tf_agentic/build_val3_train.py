"""Build a training TASKS_FILE = 3 Val3 tasks (converted from .md via TaskLoader,
so grading matches eval) + 6 ledger companions (from canonical_26), mapped to the
Val3 capability tree. Prints a self-check summary, then writes the combined JSON.
"""
import json, re, sys
from pathlib import Path

REPO = "/workspace/openclaw-naive-meeting-analysis-github"
sys.path.insert(0, f"{REPO}/scripts")
from lib_tasks import TaskLoader
from lib_grading import _extract_grading_code, _format_grading_criteria

VAL3 = ["task_meeting_advisory_stakeholders", "task_meeting_gov_speaker_summary", "task_meeting_tech_action_items"]
COMPANIONS = ["ledger_commitment_gitlab", "ledger_commitment_nasa", "ledger_commitment_ntia",
              "ledger_speaker_gitlab", "ledger_speaker_nasa", "ledger_decision_ntia"]

tasks = {t.task_id: t for t in TaskLoader(Path("/tmp/meeting_analysis_tasks_skilltest")).load_all_tasks()}


def wf_list(t):
    out = []
    for w in (t.workspace_files or []):
        src = w.get("source") if isinstance(w, dict) else getattr(w, "source", None)
        dst = w.get("dest") if isinstance(w, dict) else getattr(w, "dest", None)
        out.append({"source": src, "dest": dst})
    return out


def rubric_list(t):
    rub = getattr(t, "llm_judge_rubric", None)
    # Val3 tasks store the rubric as a markdown string — keep it verbatim so the
    # rollout judge uses the SAME rubric the eval judge uses (run_llm_judge now
    # accepts a string rubric directly).
    if isinstance(rub, str) and rub.strip():
        return rub
    if isinstance(rub, list) and rub:
        out = []
        for it in rub:
            if isinstance(it, dict):
                out.append({"name": it.get("name", "criterion"), "weight": it.get("weight", round(1.0 / len(rub), 3)), "anchors": it.get("anchors", {})})
            else:
                out.append({"name": str(it), "weight": round(1.0 / len(rub), 3), "anchors": {}})
        return out
    gc = getattr(t, "grading_criteria", None) or []
    if gc:
        return [{"name": c, "weight": round(1.0 / len(gc), 3), "anchors": {}} for c in gc]
    return []


def to_canonical(t):
    code = _extract_grading_code(t) or ""
    m = re.search(r'/\s*["\']([A-Za-z0-9_\-]+\.md)["\']', code)
    out_file = m.group(1) if m else "report.md"
    return {
        "task_id": t.task_id, "name": t.task_id, "source": "val3", "split": "train",
        "category": "meeting", "grading_type": "hybrid",
        "prompt": t.prompt, "workspace_files": wf_list(t),
        "expected_output_file": out_file, "timeout_seconds": 180,
        "grading": {"weights": dict(t.grading_weights or {"automated": 0.5, "llm_judge": 0.5}),
                    "grade_function": code, "llm_rubric": rubric_list(t)},
        # No hard require_output_file gate: the Val3 grade fns accept several report
        # filenames (e.g. speaker_summary/summary/speakers.md) but a strict gate on
        # one name false-zeros valid reports. Let automated.report_created + judge
        # score report presence naturally instead.
        "reward_contract": {"score_range": [0.0, 1.0], "primary_score": "final_score",
                            "process_gate": {}},
        "rl_grouping": {"group_id": t.task_id, "rollouts_per_group": 4},
        "target_capability": t.task_id.replace("task_meeting_", ""),
        "meeting_family": (wf_list(t)[0]["source"] if wf_list(t) else ""),
    }


val3_entries = []
print("=== Val3 转换自检 ===")
for tid in VAL3:
    if tid not in tasks:
        print("  !! MISSING task def:", tid); continue
    e = to_canonical(tasks[tid])
    val3_entries.append(e)
    print("  %-34s out=%-22s wf=%s weights=%s rubric_n=%d grade_fn=%dchars" % (
        tid, e["expected_output_file"], [w["dest"] for w in e["workspace_files"]],
        e["grading"]["weights"], len(e["grading"]["llm_rubric"]), len(e["grading"]["grade_function"])))

canon = {t["task_id"]: t for t in json.load(open(f"{REPO}/data/meeting_analysis_val3_slim_train/canonical_26_tasks.json"))}
comp_entries = []
print("=== 配料 ===")
for cid in COMPANIONS:
    if cid not in canon:
        print("  !! MISSING companion:", cid); continue
    comp_entries.append(canon[cid])
    print("  %-26s out=%s cap=%s" % (cid, canon[cid].get("expected_output_file"), canon[cid].get("target_capability")))

combined = val3_entries + comp_entries
outp = f"{REPO}/data/meeting_analysis_val3_slim_train/val3_plus6_train.json"
json.dump(combined, open(outp, "w"), ensure_ascii=False, indent=2)
print("=== 写出 %d 个任务 -> %s ===" % (len(combined), outp))
print("BUILD_DONE")
