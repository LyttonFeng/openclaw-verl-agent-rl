"""Committee + relative + semantic terminal reward for meeting tasks (veRL drop-in).

Replaces the brittle reward in meeting_reward.py (keyword grade() + single ABSOLUTE
deepseek judge) with what the no-GT verifier research found robust:

  reward = spec_gate * ( 0.7 * committee_relative + 0.3 * semantic_coverage )

  - spec_gate (deterministic): a non-trivial deliverable was produced.
  - committee_relative: 3 families (deepseek + qwen3-max + minimax) each RULER-score
    the deliverable RELATIVE to a built good-anchor + an empty bad-anchor; take the
    deliverable's score; MEDIAN across families. Relative committee is robust where a
    single absolute judge is fooled by fluent-but-wrong answers (the "validation trap").
  - semantic_coverage: a model judges which grading_criteria the answer EXPRESSES
    (rephrasing allowed) and cites the span; keyword match is a one-way positive only,
    never a 0. (Keyword/regex grading injects run-to-run false-negative reward noise.)

veRL config:
    reward.custom_reward_function.path=rewards/meeting_reward_committee.py
    reward.custom_reward_function.name=compute_score

Keys (env, falls back to ~/.pinchbench_env; degrades to fewer families if missing):
    DEEPSEEK_API_KEY, DASHSCOPE_API_KEY, MINIMAX_API_KEY
Env switches: COMMITTEE_FAMILIES="deepseek,qwen,minimax" ; MEETING_REWARD_DEBUG=1
"""
from __future__ import annotations
import json as _json
import os, re, sys, statistics as _st, urllib.request
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

# reuse the existing helpers (task loading, workspace reading, summarization)
from meeting_reward import _load_task, _read_workspace_files, _summarize_transcript, _truncate  # noqa: E402

_FAMILY = {
    "deepseek": {"key": "DEEPSEEK_API_KEY", "model": "deepseek-chat",
                 "url": "https://api.deepseek.com/v1/chat/completions"},
    "qwen": {"key": "DASHSCOPE_API_KEY", "model": "qwen3-max",
             "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions"},
    "minimax": {"key": "MINIMAX_API_KEY", "model": "MiniMax-M3", "reasoning": True,
                "url": "https://api.minimaxi.chat/v1/chat/completions"},
}
_KEYS: dict = {}


def _load_key(name: str) -> str:
    if name in os.environ and os.environ[name]:
        return os.environ[name]
    envf = Path(os.path.expanduser("~/.pinchbench_env"))
    if envf.exists():
        for line in envf.read_text().splitlines():
            line = line.strip()
            if line.startswith("export "):
                line = line[7:]
            if line.startswith(name + "="):
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    return ""


def _families():
    want = os.environ.get("COMMITTEE_FAMILIES", "deepseek,qwen,minimax").split(",")
    out = []
    for f in [w.strip() for w in want if w.strip()]:
        if f not in _FAMILY:
            continue
        if f not in _KEYS:
            _KEYS[f] = _load_key(_FAMILY[f]["key"])
        if _KEYS[f]:
            out.append(f)
    return out


def _chat(family: str, messages, max_tokens=600) -> str:
    cfg = _FAMILY[family]
    body = {"model": cfg["model"], "messages": messages, "temperature": 0.0,
            "max_tokens": max_tokens + (4096 if cfg.get("reasoning") else 0),
            "response_format": {"type": "json_object"}}
    req = urllib.request.Request(cfg["url"], data=_json.dumps(body).encode(),
                                 headers={"Authorization": "Bearer " + _KEYS[family],
                                          "Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        txt = _json.load(r)["choices"][0]["message"]["content"]
    return re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()


def _json_obj(txt: str) -> dict:
    try:
        return _json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, re.S)
        return _json.loads(m.group()) if m else {}


_RUBRIC = ("- Achieving the goal scores much higher than not.\n"
           "- More complete / better-substantiated / better-targeted scores higher.\n"
           "- Small quality gaps => small score gaps; partial credit for partial progress.")


def _ruler_one(family, goal, context, deliverables) -> dict:
    blocks = "\n\n".join(f'<deliverable id="{i}">\n{_truncate(t, 3500)}\n</deliverable>' for i, t in deliverables)
    ctx = f"\n\n<context>\n{_truncate(context, 2500)}\n</context>" if context else ""
    msg = [{"role": "system", "content":
            "You grade AI deliverables relative to each other for the same goal. "
            f"Standards:\n{_RUBRIC}\n"
            'Return JSON {"scores":[{"id":<int>,"score":<float 0..1>}]} for every deliverable.'},
           {"role": "user", "content": f"<goal>\n{_truncate(goal, 1500)}\n</goal>{ctx}\n\n"
            f"All deliverables share the same goal. Score each 0..1.\n\n{blocks}"}]
    out = {}
    try:
        for s in _json_obj(_chat(family, msg)).get("scores", []):
            out[int(s["id"])] = float(s["score"])
    except Exception:
        pass
    return out


def _committee_relative(goal, context, deliverable, good_anchor, families) -> float:
    """deliverable scored RELATIVE to a good + empty anchor; median across families."""
    delivs = [(0, deliverable), (1, good_anchor), (2, "")]
    vals = []
    for f in families:
        s = _ruler_one(f, goal, context, delivs)
        if 0 in s:
            vals.append(s[0])
    return round(_st.median(vals), 4) if vals else 0.0


def _semantic_coverage(criteria, deliverable, family) -> float:
    """Fraction of grading_criteria the deliverable EXPRESSES (model-judged, cited).
    Keyword is a one-way positive fast-path; absence never -> 0 (goes to the model)."""
    if not criteria:
        return 0.0
    dl_low = re.sub(r"\s+", " ", deliverable or "").lower()
    pending, covered = [], 0
    for i, c in enumerate(criteria):
        toks = [t for t in re.findall(r"[a-z0-9][a-z0-9\-']+", c.lower()) if len(t) > 3][:5]
        if toks and all(t in dl_low for t in toks):
            covered += 1
        else:
            pending.append((i, c))
    if pending:
        listing = "\n".join(f"{i}: {c}" for i, c in pending)
        msg = [{"role": "system", "content":
                "For each CRITERION decide if the ANSWER addresses it (rephrasing allowed). "
                'Return JSON {"present":[{"id":<int>,"quote":<verbatim span from the ANSWER>}]}. '
                "Include an id only if genuinely addressed; quote copied verbatim from the ANSWER."},
               {"role": "user", "content": f"ANSWER:\n{_truncate(deliverable, 7000)}\n\nCRITERIA:\n{listing}"}]
        try:
            for p in _json_obj(_chat(family, msg, max_tokens=1200)).get("present", []):
                q = (p.get("quote") or "").strip().lower()
                if len(q) >= 8 and q in dl_low:
                    covered += 1
        except Exception:
            pass
    return round(covered / len(criteria), 4)


def _deliverable_text(solution_str, extra_info) -> str:
    ws = (extra_info or {}).get("workspace_path", "")
    body = _read_workspace_files(ws) if ws else ""
    return body if len(body) > 40 else (solution_str or "")


def compute_score(data_source: str, solution_str: str, ground_truth: str,
                  extra_info: dict | None = None, **kwargs) -> dict:
    extra_info = extra_info or {}
    task_id = ground_truth or extra_info.get("task_id", "")
    if not task_id:
        return {"score": 0.0, "error": "no task_id", "task_id": ""}
    try:
        task = _load_task(task_id)
    except FileNotFoundError as e:
        return {"score": 0.0, "error": str(e), "task_id": task_id}

    deliverable = _deliverable_text(solution_str, extra_info)
    context = (extra_info.get("transcript") and _summarize_transcript(extra_info["transcript"])) or ""
    goal = f"{task.prompt}\n\nExpected:\n{task.expected_behavior or ''}"
    criteria = task.grading_criteria or []
    good_anchor = ("Reference answer covering: " + "; ".join(criteria)) if criteria else (task.expected_behavior or "")

    # gate: deterministic — a non-trivial deliverable exists
    gate = 1.0 if len(re.sub(r"\s+", " ", deliverable).strip()) >= 80 else 0.0
    if gate == 0.0:
        return {"score": 0.0, "gate": 0.0, "task_id": task_id, "note": "no deliverable"}

    fams = _families()
    if not fams:
        return {"score": 0.0, "error": "no committee keys", "task_id": task_id}
    committee = _committee_relative(goal, context, deliverable, good_anchor, fams)
    coverage = _semantic_coverage(criteria, deliverable, fams[0])
    content = 0.7 * committee + 0.3 * coverage
    score = round(gate * content, 4)
    out = {"score": score, "gate": gate, "committee_relative": committee,
           "semantic_coverage": coverage, "families": fams, "task_id": task_id}
    if os.environ.get("MEETING_REWARD_DEBUG") == "1":
        out["deliverable_chars"] = len(deliverable)
    return out
