"""RULER-style RELATIVE committee reward scorer for GRPO training.

Generalizes the EVAL insight (relative > absolute; committee kills idiosyncratic
bias; bake in anti-reward-hacking grounding rules) into a TRAINING reward:

For each task-group of K rollouts:
  - extract the K deliverables, ANONYMIZE + SHUFFLE (judge can't tell order/origin)
  - each committee judge scores ALL K relative to each other, 0-1, in ONE call
    (RULER: relative ranking is easier + GRPO only needs within-group relative values)
  - reward[rollout] = mean over judges over shuffles
  - the RUBRIC explicitly rewards GROUNDING and penalizes DUPLICATES / HALLUCINATION
    / PADDING -- the exact failure mode the pairwise committee caught on tech.

Optional --pairwise: within-group pairwise (C(K,2) pairs x 2 orders) -> win-count ->
reward (more stable at small K, but O(K^2) calls).

API-only, mac-local, thread-parallel. Debug tool: prints per-rollout scores + spread
so we can sanity-check BEFORE feeding GRPO. No GPU, no ssh pipeline.
"""
import sys, os, json, glob, re, itertools
from statistics import mean, pstdev
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.environ.get("COMMITTEE_DIR", os.path.expanduser("~/eval_val3_transcripts")))
import committee_judge as C  # reuse MEMBERS, _call, lib, pair_msgs, consistent_verdict
lib = C.lib

RDIR = os.environ.get("ROLLOUTS_DIR", os.path.expanduser("~/r1_rollouts"))
MAXREP = 6000
SHUFFLES4 = [[0, 1, 2, 3], [2, 0, 3, 1], [3, 1, 0, 2]]  # deterministic perms of K=4

# Generic fallback rubric (RULER-style) — used ONLY when a task has no hand-written llm_rubric.
RUBRIC = (
    "- A report that fully and ACCURATELY fulfills the task scores highest.\n"
    "- GROUNDING is paramount: every item/claim must be supported by the source meeting transcript. "
    "Strongly penalize hallucinated or unsupported content.\n"
    "- Penalize DUPLICATE entries and redundant PADDING: extra length that does not add new, grounded "
    "information must LOWER the score, never raise it.\n"
    "- Reward completeness of genuinely-grounded items, correct owners/deadlines where stated, and clear structure.\n"
    "- A concise, fully-grounded report MUST score higher than a longer report that adds duplicates or ungrounded items."
)


def normalize_rubric(rub):
    """Turn a task llm_rubric (markdown str OR list of {name,weight,anchors}) into a text block.
    Returns the generic RUBRIC when rub is empty/None."""
    if not rub:
        return RUBRIC
    if isinstance(rub, str):
        return rub.strip()
    lines = []
    for c in rub:
        if not isinstance(c, dict):
            continue
        nm = c.get("name", "criterion"); w = c.get("weight", "")
        lines.append(f"### {nm}" + (f" (weight {w})" if w != "" else ""))
        anchors = c.get("anchors", {}) or {}
        for k in ("1.0", "0.5", "0", "0.0"):
            if k in anchors:
                lines.append(f"- Score {k}: {anchors[k]}")
    return "\n".join(lines) if lines else RUBRIC


def _sys_list(rubric_text, has_rubric=True):
    # Anti-hacking overlay that NEVER conflicts with completeness:
    overlay = ("\n\nIn addition, penalize content that is hallucinated or NOT grounded in the source "
               "transcript, and penalize DUPLICATE entries (the same item repeated). Do not reward length "
               "for its own sake.")
    # The "concise must beat longer" preference lives only in the generic fallback rubric,
    # so it does not fight a task rubric that legitimately rewards comprehensiveness.
    return (
        "All of the reports below were written for the SAME task. Consider each and assign it a score "
        "between 0 and 1 for how well it fulfills the task, judging them relative to each other.\n\n"
        "Grading standards (apply these criteria faithfully):\n" + rubric_text + overlay +
        '\n\nReturn STRICT JSON only: {"scores":[{"id":1,"score":0.0,"reason":"<short>"}, ...]} with one '
        "entry for EVERY report id."
    )


def groups():
    g = {}
    for fp in glob.glob(f"{RDIR}/*_resp*.jsonl"):
        m = re.match(r"(.+)_resp(\d+)\.jsonl$", os.path.basename(fp))
        if m:
            g.setdefault(m.group(1), {})[int(m.group(2))] = fp
    return {k: [v[i] for i in sorted(v)] for k, v in sorted(g.items())}


def deliverables(files):
    out = []
    task = None
    for fp in files:
        tr = lib.parse_trajectory(fp)
        task = task or tr["prompt"]
        out.append({"dlv": (tr["deliverable"] or "")[:MAXREP], "len": len(tr["deliverable"] or ""),
                    "writes": tr["num_writes"]})
    return task, out


def listwise_once(member, task, items, order, rubric=None, reference=None):
    rubric_text = normalize_rubric(rubric)  # task llm_rubric if given, else generic RUBRIC
    sys_p = _sys_list(rubric_text, has_rubric=bool(rubric))
    parts = [f'<report id="{pos+1}">\n{items[idx]["dlv"]}\n</report>' for pos, idx in enumerate(order)]
    # 放法B: inject a base-model reference for CALIBRATION only. Do NOT score it; it just
    # anchors the judge's scale ("how do these compare to a known baseline answer").
    ref_block = ""
    if reference:
        ref_block = ("<reference_baseline>\nThe following is a baseline answer for THIS task. Do NOT "
                     "score it. Use it only to calibrate your scale: a report clearly better than this "
                     "baseline should score higher, one clearly worse should score lower.\n"
                     + str(reference)[:MAXREP] + "\n</reference_baseline>\n\n")
    user = f"<task>\n{(task or '')[:5000]}\n</task>\n\n{ref_block}Reports to score:\n\n" + "\n\n".join(parts)
    obj = C._call(member, [{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
                  max_tokens=1500)
    pos_sc = {int(e["id"]): float(e["score"]) for e in obj.get("scores", [])}
    return {idx: pos_sc.get(pos + 1) for pos, idx in enumerate(order)}  # item_idx -> score


def listwise_group(task, items, members, shuffles, pool=6, rubric=None, reference=None):
    jobs = [(m, si) for m in members for si in range(len(shuffles))]

    def run(job):
        m, si = job
        try:
            return (m, listwise_once(m, task, items, shuffles[si], rubric=rubric, reference=reference))
        except Exception as e:
            print(f"    [{m}] shuffle{si} ERR {repr(e)[:80]}", flush=True)
            return (m, None)

    with ThreadPoolExecutor(max_workers=pool) as ex:
        res = list(ex.map(run, jobs))
    # per-judge mean over shuffles, then panel mean
    K = len(items)
    by_judge = {m: {i: [] for i in range(K)} for m in members}
    for m, sc in res:
        if sc:
            for i in range(K):
                if sc.get(i) is not None:
                    by_judge[m][i].append(sc[i])
    judge_mean = {m: {i: (mean(by_judge[m][i]) if by_judge[m][i] else None) for i in range(K)} for m in members}
    reward = {}
    for i in range(K):
        vals = [judge_mean[m][i] for m in members if judge_mean[m][i] is not None]
        reward[i] = mean(vals) if vals else None
    return reward, judge_mean


def pairwise_group(task, items, members, pool=6):
    """C(K,2) pairs x 2 orders x committee -> win count -> normalized reward."""
    K = len(items)
    pairs = list(itertools.combinations(range(K), 2))
    jobs = [(m, pi) for pi in range(len(pairs)) for m in members]

    def run(job):
        m, pi = job
        a, b = pairs[pi]
        side, _ = C.consistent_verdict(m, task, items[a]["dlv"], items[b]["dlv"])
        return (pi, m, side)  # 'left'=a, 'right'=b, 'tie'

    with ThreadPoolExecutor(max_workers=pool) as ex:
        res = list(ex.map(run, jobs))
    wins = {i: 0.0 for i in range(K)}
    for pi, m, side in res:
        a, b = pairs[pi]
        if side == "left":
            wins[a] += 1
        elif side == "right":
            wins[b] += 1
        else:
            wins[a] += 0.5; wins[b] += 0.5
    maxw = (K - 1) * len(members)
    return {i: wins[i] / maxw for i in range(K)}, None


if __name__ == "__main__":
    pairwise = "--pairwise" in sys.argv
    smoke = "--smoke" in sys.argv
    only = [a for a in sys.argv[1:] if not a.startswith("--")]
    G = groups()
    members = list(C.MEMBERS)[:1] if smoke else list(C.MEMBERS)
    shuffles = SHUFFLES4[:1] if smoke else SHUFFLES4
    names = only if only else list(G)
    allrows = {}
    for name in names:
        files = G[name]
        task, items = deliverables(files)
        if pairwise:
            reward, jm = pairwise_group(task, items, members)
        else:
            reward, jm = listwise_group(task, items, members, shuffles)
        rs = [reward[i] for i in range(len(items)) if reward[i] is not None]
        spread = pstdev(rs) if len(rs) > 1 else 0.0
        print(f"\n=== {name}  ({'pairwise' if pairwise else 'listwise'})  spread={spread:.3f} ===")
        for i in range(len(items)):
            jstr = ""
            if jm:
                jstr = "  [" + " ".join(f"{m.split('-')[0]}={jm[m][i]:.2f}" if jm[m][i] is not None else f"{m}=NA" for m in members) + "]"
            print(f"  resp{i}: reward={reward[i] if reward[i] is None else round(reward[i],3)}  len={items[i]['len']:6d} writes={items[i]['writes']}{jstr}")
        allrows[name] = reward
    print("\nRULER_REWARD_DONE")
