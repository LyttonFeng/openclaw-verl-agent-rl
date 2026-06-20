"""Shared helpers for no-GT reward experiments.

Parses OpenClaw trajectories, talks to DeepSeek, and provides
retrieval + correlation utilities. No PinchBench grading is ever read here.
"""
import re, json, os, time, urllib.request, urllib.error, hashlib, glob

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_key(name="DEEPSEEK_API_KEY", path="/Users/lytton/.pinchbench_env"):
    for line in open(path):
        m = re.match(r"\s*export\s+" + name + r"\s*=\s*'([^']+)'", line)
        if m:
            return m.group(1)
    raise RuntimeError("key not found: " + name)

_KEY = None
def ds_chat(messages, model="deepseek-chat", max_tokens=2048, temperature=0.0,
            json_mode=False, retries=4):
    """Single DeepSeek chat call with retry. Returns assistant text."""
    global _KEY
    if _KEY is None:
        _KEY = load_key()
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    data = json.dumps(body).encode()
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(
                "https://api.deepseek.com/v1/chat/completions", data=data,
                headers={"Authorization": "Bearer " + _KEY,
                         "Content-Type": "application/json"})
            r = urllib.request.urlopen(req, timeout=180)
            d = json.load(r)
            return d["choices"][0]["message"]["content"]
        except Exception as e:
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError("ds_chat failed: %r" % last)

def ds_json(messages, **kw):
    """Call DeepSeek in JSON mode and parse. Falls back to brace-extraction."""
    txt = ds_chat(messages, json_mode=True, **kw)
    try:
        return json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            return json.loads(m.group(0))
        raise

# ---------- multi-family judge layer (L2 adversarial) ----------
_FAMILY = {
    "deepseek": {"key": "DEEPSEEK_API_KEY",
                 "url": "https://api.deepseek.com/v1/chat/completions",
                 "model": "deepseek-chat"},
    "qwen": {"key": "DASHSCOPE_API_KEY",
             "url": "https://dashscope.aliyuncs.com/compatible-mode/v1/chat/completions",
             "model": "qwen3-max"},
    "minimax": {"key": "MINIMAX_API_KEY",
                "url": "https://api.minimaxi.chat/v1/chat/completions",
                "model": "MiniMax-M3", "reasoning": True},
}
_KEYS = {}
def family_chat(family, messages, max_tokens=512, temperature=0.0,
                json_mode=False, retries=4):
    """Unified judge call routed to a model family. Returns assistant text."""
    cfg = _FAMILY[family]
    if family not in _KEYS:
        _KEYS[family] = load_key(cfg["key"])
    reasoning = cfg.get("reasoning", False)
    body = {"model": cfg["model"], "messages": messages,
            # reasoning models burn tokens in <think>; give headroom
            "max_tokens": max_tokens + (4096 if reasoning else 0),
            "temperature": temperature}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    data = json.dumps(body).encode()
    last = None
    for i in range(retries):
        try:
            req = urllib.request.Request(cfg["url"], data=data,
                headers={"Authorization": "Bearer " + _KEYS[family],
                         "Content-Type": "application/json"})
            r = urllib.request.urlopen(req, timeout=180)
            d = json.load(r)
            txt = d["choices"][0]["message"]["content"]
            # strip reasoning blocks so downstream JSON/parse sees only the answer
            txt = re.sub(r"<think>.*?</think>", "", txt, flags=re.S).strip()
            return txt
        except Exception as e:
            last = e
            time.sleep(2 * (i + 1))
    raise RuntimeError("family_chat(%s) failed: %r" % (family, last))

def family_json(family, messages, **kw):
    txt = family_chat(family, messages, json_mode=True, **kw)
    try:
        return json.loads(txt)
    except Exception:
        m = re.search(r"\{.*\}", txt, re.S)
        if m:
            return json.loads(m.group(0))
        raise

# ---------- trajectory parsing ----------
def parse_trajectory(path):
    """Extract prompt, read behaviour, and the model's final deliverable."""
    lines = [json.loads(l) for l in open(path) if l.strip()]
    prompt = None
    reads = []          # list of dicts: {args, result_chars, used_offset}
    writes = []         # list of dicts: {path, content}
    final_text = ""     # last assistant text block (the closing message)
    pending_read = None
    for o in lines:
        if o.get("type") != "message":
            continue
        m = o["message"]; role = m["role"]; blocks = m.get("content", [])
        if role == "user" and prompt is None:
            for b in blocks:
                if b.get("type") == "text":
                    prompt = b["text"]; break
        if role == "assistant":
            for b in blocks:
                if b.get("type") == "text" and b.get("text", "").strip():
                    final_text = b["text"]
                if b.get("type") == "toolCall":
                    nm = (b.get("name") or "").lower()
                    args = b.get("arguments", {}) or {}
                    if nm in ("read", "readfile", "read_file"):
                        pending_read = {"args": args,
                                        "used_offset": bool(args.get("offset")),
                                        "result_chars": None}
                        reads.append(pending_read)
                    if nm in ("write", "writefile", "write_file", "edit"):
                        writes.append({"path": args.get("path") or args.get("file_path"),
                                       "content": args.get("content", "")})
        if role == "toolResult":
            txt = ""
            for b in blocks:
                if b.get("type") == "text":
                    txt += b.get("text", "")
            if pending_read is not None and pending_read["result_chars"] is None:
                pending_read["result_chars"] = len(txt)
                pending_read = None
    # the deliverable: prefer the largest written file; else final assistant text
    deliverable = ""
    if writes:
        deliverable = max(writes, key=lambda w: len(w["content"]))["content"]
    if len(deliverable) < len(final_text):
        deliverable = final_text if len(final_text) > len(deliverable) else deliverable
    max_read = max([r["result_chars"] or 0 for r in reads], default=0)
    return {
        "path": path,
        "prompt": prompt or "",
        "deliverable": deliverable,
        "final_text": final_text,
        "num_reads": len(reads),
        "max_read_chars": max_read,
        "used_offset": any(r["used_offset"] for r in reads),
        "num_writes": len(writes),
        "raw_lines": len(lines),
    }

def task_trajectory_files(task_id, tdir):
    """All run files for a task, sorted by trailing timestamp (run order)."""
    pat = os.path.join(tdir, task_id + "__*.jsonl")
    files = sorted(glob.glob(pat), key=lambda p: re.findall(r"_(\d+)\.jsonl$", p))
    return files

# ---------- retrieval grounding ----------
def keyword_windows(source, phrase, window=1500, max_hits=2):
    """Find context windows in source around the salient tokens of a phrase."""
    toks = [t for t in re.findall(r"[A-Za-z0-9][A-Za-z0-9\-']+", phrase) if len(t) > 3]
    toks = sorted(set(toks), key=lambda t: -len(t))[:6]
    src_low = source.lower()
    hits = []
    for t in toks:
        idx = src_low.find(t.lower())
        if idx >= 0:
            s = max(0, idx - window); e = min(len(source), idx + window)
            hits.append((s, e))
        if len(hits) >= max_hits:
            break
    if not hits:
        return ""
    # merge overlapping
    hits.sort()
    merged = [hits[0]]
    for s, e in hits[1:]:
        if s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return "\n...\n".join(source[s:e] for s, e in merged)

def regurgitation_ratio(deliverable, source, win=100):
    """Fraction of the deliverable that is verbatim contiguous source text.

    High => the answer is a copy-paste dump of the transcript, not extraction.
    Defeats the reward-hack where pasting raw source yields precision=1.0.
    """
    d = re.sub(r"\s+", " ", deliverable).strip()
    s = re.sub(r"\s+", " ", source).lower()
    if len(d) < 40:
        return 0.0
    if len(d) < win:
        return 1.0 if d.lower() in s else 0.0
    covered = total = 0
    for i in range(0, len(d) - win, win):
        total += 1
        if d[i:i+win].lower() in s:
            covered += 1
    return covered / max(1, total)

def regurgitation_penalty_factor(deliverable, source, free=0.35):
    """Multiplier in [0,1]: 1 below `free` copy-ratio, ramps to 0 at full copy."""
    r = regurgitation_ratio(deliverable, source)
    if r <= free:
        return 1.0
    return max(0.0, 1.0 - (r - free) / (1.0 - free))

def substring_grounded(quote, source, min_len=12):
    """Cheap programmatic check: is a quote literally in the source?"""
    q = re.sub(r"\s+", " ", quote).strip().lower()
    if len(q) < min_len:
        return None  # too short to be meaningful
    s = re.sub(r"\s+", " ", source).lower()
    return q in s

# ---------- stats ----------
def spearman(xs, ys):
    n = len(xs)
    if n < 2:
        return float("nan")
    def rank(v):
        order = sorted(range(len(v)), key=lambda i: v[i])
        r = [0.0] * len(v)
        i = 0
        while i < len(v):
            j = i
            while j + 1 < len(v) and v[order[j+1]] == v[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1
            for k in range(i, j + 1):
                r[order[k]] = avg
            i = j + 1
        return r
    rx, ry = rank(xs), rank(ys)
    mx = sum(rx)/n; my = sum(ry)/n
    num = sum((rx[i]-mx)*(ry[i]-my) for i in range(n))
    dx = (sum((rx[i]-mx)**2 for i in range(n)))**0.5
    dy = (sum((ry[i]-my)**2 for i in range(n)))**0.5
    return num/(dx*dy) if dx and dy else float("nan")

def pearson(xs, ys):
    n = len(xs)
    if n < 2: return float("nan")
    mx=sum(xs)/n; my=sum(ys)/n
    num=sum((xs[i]-mx)*(ys[i]-my) for i in range(n))
    dx=(sum((xs[i]-mx)**2 for i in range(n)))**0.5
    dy=(sum((ys[i]-my)**2 for i in range(n)))**0.5
    return num/(dx*dy) if dx and dy else float("nan")
