"""Stable + accurate EVAL verdict (base vs lora) via a PAIRWISE judge COMMITTEE.

Goal (per user): reference RULER's relative-scoring insight, but (1) use PAIRWISE
comparisons to avoid absolute scoring, and (2) an original COMMITTEE protocol, to
make the non-automated judge both ACCURATE and STABLE.

Pillars
-------
1. PAIRWISE (relative, not absolute): for a task, each base report is compared
   against each lora report -> {winner, reason}. No 0-1 absolute scale to drift.
2. ORDER-CONSISTENCY debias: every pair is judged in BOTH orders (A=base/B=lora and
   swapped). A side wins ONLY if chosen in both orders; an order-flip => TIE.
   Position bias becomes a tie, not false signal.
3. COMMITTEE of heterogeneous judges (ds-v4-flash / qwen-max / minimax-M3 ->
   uncorrelated family bias):
     Round 1: each member independently renders consistency-checked pair verdicts.
     Round 2 (DELIBERATION): only for pairs the committee is NOT unanimous on, each
       member sees peers' verdict+reason and casts a FINAL vote.
     Chair: per-pair committee verdict = majority of final votes (else TIE).
4. SIGN TEST significance over the 9 pairs (two-sided binomial on non-tie pairs).
   Plus NULL CALIBRATION: base-vs-base should be ~all ties (validates no residual bias).

API-only, mac-local, thread-parallel. No GPU, no ssh pipeline.
"""
import sys, os, json, glob, re, math, itertools, urllib.request
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.environ.get("JUDGE_LIB_DIR",
    "/Users/lytton/work/reinforement_learning/timeline/data_agent/experiments"))
import lib  # provides load_key + parse_trajectory; multi-family judge entry

TDIR = os.environ.get("EVAL_TRANSCRIPTS_DIR", os.path.expanduser("~/eval_val3_transcripts"))
TASKS = ["task_meeting_advisory_stakeholders", "task_meeting_gov_speaker_summary", "task_meeting_tech_action_items"]
MAXREP = 8000

# committee members: name -> (url, key_name, model, is_reasoning)
MEMBERS = {
    "ds-flash":   ("https://api.deepseek.com/v1/chat/completions", "DEEPSEEK_API_KEY", "deepseek-v4-flash", True),
    "qwen-max":   ("https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions", "DASHSCOPE_API_KEY", "qwen3-max", False),
    "minimax-M3": ("https://api.minimaxi.chat/v1/chat/completions", "MINIMAX_API_KEY", "MiniMax-M3", True),
}
_KEYS = {}

SYS_PAIR = (
    "You are a strict, impartial evaluator. Two reports, A and B, were written for the SAME task. "
    "Decide which report BETTER fulfills the task. Judge on: accuracy, completeness, grounding in the "
    "source meeting transcript, and structure. If the two reports are essentially equal in quality, "
    'answer "tie". Output STRICT JSON only: {"winner":"A"|"B"|"tie","reason":"<one short sentence>"}.'
)


def _call(member, messages, max_tokens=512):
    url, keyname, model, reasoning = MEMBERS[member]
    if member not in _KEYS:
        _KEYS[member] = lib.load_key(keyname)
    body = {"model": model, "messages": messages, "temperature": 0.0,
            "max_tokens": max_tokens + (3072 if reasoning else 0),
            "response_format": {"type": "json_object"}}
    data = json.dumps(body).encode()
    last = None
    for i in range(4):
        try:
            req = urllib.request.Request(url, data=data,
                headers={"Authorization": "Bearer " + _KEYS[member], "Content-Type": "application/json"})
            d = json.load(urllib.request.urlopen(req, timeout=180))
            m = d["choices"][0]["message"]
            txt = (m.get("content") or "").strip()
            if not txt:  # reasoning models can leave content empty -> fall back
                txt = (m.get("reasoning_content") or "")
            txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
            try:
                return json.loads(txt)
            except Exception:
                mm = re.search(r"\{.*\}", txt, re.S)
                if mm:
                    return json.loads(mm.group(0))
                raise ValueError("no json: " + txt[:80])
        except Exception as e:
            last = e
            import time; time.sleep(1.5 * (i + 1))
    raise RuntimeError(f"_call({member}) failed: {last!r}")


def pair_msgs(task, repA, repB, peer_note=""):
    user = f"<task>\n{(task or '')[:6000]}\n</task>\n\n=== REPORT A ===\n{repA}\n\n=== REPORT B ===\n{repB}"
    if peer_note:
        user += "\n\n" + peer_note
    return [{"role": "system", "content": SYS_PAIR}, {"role": "user", "content": user}]


def newest3(mode, tid):
    return sorted(glob.glob(f"{TDIR}/{mode}/{tid}__*.jsonl"), key=os.path.getmtime)[-3:]


def load_reports(tid):
    out = {"base": [], "lora": []}
    task_prompt = None
    for mode in ("base", "lora"):
        for fp in newest3(mode, tid):
            tr = lib.parse_trajectory(fp)
            task_prompt = task_prompt or tr["prompt"]
            out[mode].append((tr["deliverable"] or "")[:MAXREP])
    return task_prompt, out


def consistent_verdict(member, task, left, right):
    """Judge (A=left,B=right) and (A=right,B=left). Return 'left'|'right'|'tie' + reasons."""
    o1 = _call(member, pair_msgs(task, left, right))
    o2 = _call(member, pair_msgs(task, right, left))
    w1, w2 = o1.get("winner", "tie"), o2.get("winner", "tie")
    s1 = "left" if w1 == "A" else ("right" if w1 == "B" else "tie")
    s2 = "right" if w2 == "A" else ("left" if w2 == "B" else "tie")  # swapped order
    side = s1 if s1 == s2 else "tie"  # consistent only
    return side, [o1.get("reason", ""), o2.get("reason", "")]


def sign_test(wins_lora, wins_base):
    """Two-sided binomial sign test on decisive (non-tie) pairs."""
    n = wins_lora + wins_base
    if n == 0:
        return 1.0
    k = max(wins_lora, wins_base)
    p = sum(math.comb(n, i) for i in range(k, n + 1)) / (2 ** n) * 2
    return min(1.0, p)


def committee_task(tid, mode_a="base", mode_b="lora", deliberate=True, pool=6):
    """Run the pairwise committee. mode_a vs mode_b reports (use base vs base for null)."""
    task, reps = load_reports(tid)
    A, B = reps[mode_a], reps[mode_b]
    if mode_a == mode_b:  # null: distinct-index pairs only (3 choose 2 = 3)
        pairs = [(i, j) for i, j in itertools.combinations(range(3), 2)]
    else:
        pairs = [(i, j) for i in range(3) for j in range(3)]  # 3x3 = 9

    members = list(MEMBERS)
    jobs = [(m, pi) for pi in range(len(pairs)) for m in members]

    def run_one(job):
        m, pi = job
        i, j = pairs[pi]
        side, reasons = consistent_verdict(m, task, A[i], B[j])  # 'left'=mode_a, 'right'=mode_b
        return (pi, m, side, reasons)

    with ThreadPoolExecutor(max_workers=pool) as ex:
        r1 = list(ex.map(run_one, jobs))

    # collect round-1 verdicts per pair
    by_pair = {pi: {} for pi in range(len(pairs))}
    reasons_by = {pi: {} for pi in range(len(pairs))}
    for pi, m, side, reasons in r1:
        by_pair[pi][m] = side
        reasons_by[pi][m] = reasons

    # deliberation on non-unanimous pairs
    if deliberate:
        disputed = [pi for pi in by_pair if len(set(by_pair[pi].values())) > 1]
        def deliberate_one(arg):
            pi, m = arg
            i, j = pairs[pi]
            peer = "Other evaluators' verdicts on this same A-vs-B pair (A is the first report, B the second):\n"
            for pm in members:
                if pm == m:
                    continue
                sv = by_pair[pi][pm]
                wv = "A" if sv == "left" else ("B" if sv == "right" else "tie")
                peer += f"- {pm}: winner={wv}; reason={reasons_by[pi][pm][0][:120]}\n"
            peer += "Reconsider carefully and give your FINAL verdict."
            o = _call(m, pair_msgs(task, A[i], B[j], peer_note=peer))
            w = o.get("winner", "tie")
            return pi, m, ("left" if w == "A" else ("right" if w == "B" else "tie"))
        delib_jobs = [(pi, m) for pi in disputed for m in members]
        with ThreadPoolExecutor(max_workers=pool) as ex:
            for pi, m, side in ex.map(deliberate_one, delib_jobs):
                by_pair[pi][m] = side

    # chair: per-pair majority
    a_wins = b_wins = ties = 0
    rows = []
    for pi in range(len(pairs)):
        votes = list(by_pair[pi].values())
        cnt = {"left": votes.count("left"), "right": votes.count("right"), "tie": votes.count("tie")}
        if cnt["left"] > cnt["right"] and cnt["left"] >= 2:
            v = "left"; a_wins += 1
        elif cnt["right"] > cnt["left"] and cnt["right"] >= 2:
            v = "right"; b_wins += 1
        else:
            v = "tie"; ties += 1
        rows.append((pairs[pi], cnt, v))
    return {"task": tid, "mode_a": mode_a, "mode_b": mode_b, "pairs": len(pairs),
            "a_wins": a_wins, "b_wins": b_wins, "ties": ties, "rows": rows}


def show(res):
    a, b = res["mode_a"], res["mode_b"]
    print(f"\n=== {res['task'].replace('task_meeting_','')}  ({a} vs {b}) ===")
    for (i, j), cnt, v in res["rows"]:
        vlabel = a if v == "left" else (b if v == "right" else "tie")
        print(f"  {a}[{i}] vs {b}[{j}]: votes L/R/T={cnt['left']}/{cnt['right']}/{cnt['tie']} -> {vlabel}")
    p = sign_test(res["b_wins"], res["a_wins"]) if (a != b) else sign_test(res["a_wins"], res["b_wins"])
    print(f"  TALLY  {a}-wins={res['a_wins']}  {b}-wins={res['b_wins']}  ties={res['ties']}  (of {res['pairs']})  sign-p={p:.3f}")
    if a == b:
        verdict = "OK null (~ties expected)" if res["a_wins"] + res["b_wins"] <= 1 else "WARN residual bias"
    else:
        if p < 0.10 and res["b_wins"] > res["a_wins"]:
            verdict = f"{b} > {a} (significant)"
        elif p < 0.10 and res["a_wins"] > res["b_wins"]:
            verdict = f"{a} > {b} (significant)"
        else:
            verdict = "TIE / no real difference"
    print(f"  VERDICT: {verdict}")
    return verdict


if __name__ == "__main__":
    smoke = "--smoke" in sys.argv
    null = "--null" in sys.argv
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    tasks = only if only else TASKS
    for tid in tasks:
        if smoke:
            task, reps = load_reports(tid)
            side, rs = consistent_verdict("ds-flash", task, reps["base"][0], reps["lora"][0])
            print(f"[smoke] {tid}: ds-flash base[0] vs lora[0] -> {side}; reasons={rs}")
        elif null:
            show(committee_task(tid, "base", "base", deliberate=False))
        else:
            show(committee_task(tid, "base", "lora", deliberate=True))
    print("\nCOMMITTEE_DONE")
