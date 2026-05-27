#!/usr/bin/env python3
"""Build a static HTML viewer for comparing val_5 OpenClaw trajectories.

The default inputs are the current val_5 DSv4 Flash, Qwen3-4B base, and
v38 ckpt9 RL runs on the benchmark pod. The script has no third-party
dependencies and embeds all parsed transcript data into one HTML file.
"""

from __future__ import annotations

import argparse
import html
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


VAL5_TASKS = [
    "task_meeting_advisory_stakeholders",
    "task_meeting_council_votes",
    "task_meeting_gov_speaker_summary",
    "task_meeting_tech_action_items",
    "task_meeting_sentiment_analysis",
]

DEFAULT_RUNS = [
    "DSv4 Flash=/workspace/verl_port/bench/dsv4flash_20260521_052408/r1",
    "Base 4B=/workspace/verl_port/ahe_v05_20260520_222239/baseline/r1",
    "RL v38 ckpt9=/workspace/verl_port/bench/v38_ckpt9_r2",
]


def text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return json.dumps(content, ensure_ascii=False) if content is not None else ""
    parts: list[str] = []
    for item in content:
        if isinstance(item, dict):
            if item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif item.get("type") == "thinking":
                parts.append(str(item.get("thinking") or ""))
            elif item.get("type") == "toolCall":
                parts.append(json.dumps(item.get("arguments") or {}, ensure_ascii=False))
    return "".join(parts)


def compact_path(value: Any) -> Any:
    if isinstance(value, str):
        return re.sub(r"/tmp/pinchbench/\d+/agent_workspace/", "", value)
    if isinstance(value, list):
        return [compact_path(v) for v in value]
    if isinstance(value, dict):
        return {k: compact_path(v) for k, v in value.items()}
    return value


def parse_transcript(path: Path) -> dict[str, Any]:
    meta: dict[str, Any] = {"path": str(path), "exists": path.exists()}
    messages: list[dict[str, Any]] = []
    event_counts: Counter[str] = Counter()
    model_id = None
    session_id = None
    cwd = None

    if not path.exists():
        return {"meta": meta, "messages": messages, "metrics": {}, "error": "missing transcript"}

    for line_no, line in enumerate(path.read_text("utf-8", errors="replace").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            messages.append({
                "kind": "error",
                "role": "parse_error",
                "title": f"JSON parse error at line {line_no}",
                "text": str(exc),
            })
            continue

        event_type = event.get("type") or "unknown"
        event_counts[event_type] += 1
        if event_type == "session":
            session_id = event.get("id")
            cwd = event.get("cwd")
            meta.update({"session_id": session_id, "cwd": cwd, "timestamp": event.get("timestamp")})
            continue
        if event_type == "model_change":
            model_id = event.get("modelId")
            meta.update({"provider": event.get("provider"), "model_id": model_id})
            continue
        if event_type != "message":
            continue

        msg = event.get("message") or {}
        role = msg.get("role") or "unknown"
        row: dict[str, Any] = {
            "kind": "message",
            "role": role,
            "id": event.get("id"),
            "timestamp": event.get("timestamp"),
            "blocks": [],
            "tool_calls": [],
            "text": "",
            "thinking": "",
            "is_error": bool(msg.get("isError")),
        }

        if role == "toolResult":
            row["tool_name"] = msg.get("toolName")
            row["tool_call_id"] = msg.get("toolCallId")
            row["text"] = text_from_content(msg.get("content"))
            row["details"] = compact_path(msg.get("details"))
        else:
            for item in msg.get("content") or []:
                if not isinstance(item, dict):
                    continue
                item_type = item.get("type")
                if item_type == "text":
                    text = str(item.get("text") or "")
                    row["text"] += text
                    row["blocks"].append({"type": "text", "text": text})
                elif item_type == "thinking":
                    thinking = str(item.get("thinking") or "")
                    row["thinking"] += thinking
                    row["blocks"].append({"type": "thinking", "text": thinking})
                elif item_type == "toolCall":
                    call = {
                        "id": item.get("id"),
                        "name": item.get("name"),
                        "arguments": compact_path(item.get("arguments") or {}),
                    }
                    row["tool_calls"].append(call)
                    row["blocks"].append({"type": "toolCall", "tool_call": call})

        messages.append(row)

    tool_calls = [
        call
        for msg in messages
        for call in msg.get("tool_calls", [])
    ]
    tool_results = [msg for msg in messages if msg.get("role") == "toolResult"]
    metrics = {
        "events": sum(event_counts.values()),
        "messages": len(messages),
        "assistant_turns": sum(1 for msg in messages if msg.get("role") == "assistant"),
        "tool_calls": len(tool_calls),
        "tool_results": len(tool_results),
        "errors": sum(1 for msg in messages if msg.get("is_error")),
        "tool_counts": dict(Counter(call.get("name") or "unknown" for call in tool_calls)),
        "event_counts": dict(event_counts),
    }
    meta.setdefault("model_id", model_id)
    meta.setdefault("session_id", session_id)
    meta.setdefault("cwd", cwd)
    return {"meta": meta, "messages": messages, "metrics": metrics}


def load_result_json(run_dir: Path) -> dict[str, Any]:
    jsons = sorted(p for p in run_dir.glob("*.json") if not p.name.endswith(".partial.json"))
    if not jsons:
        return {}
    try:
        return json.loads(jsons[0].read_text("utf-8", errors="replace"))
    except Exception as exc:
        return {"_error": str(exc), "_path": str(jsons[0])}


def score_by_task(result: dict[str, Any]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for task in result.get("tasks") or []:
        task_id = task.get("task_id")
        if not task_id:
            continue
        grading = task.get("grading") or {}
        run = (grading.get("runs") or [{}])[0]
        out[task_id] = {
            "score": run.get("score", grading.get("mean")),
            "status": task.get("status"),
            "timed_out": task.get("timed_out"),
            "execution_time": task.get("execution_time"),
            "transcript_length": task.get("transcript_length"),
            "usage": task.get("usage") or {},
            "notes": run.get("notes") or "",
            "breakdown": run.get("breakdown") or {},
            "workspace": task.get("workspace"),
        }
    return out


def find_transcript_dir(run_dir: Path) -> Path | None:
    if run_dir.name.endswith("_transcripts") and run_dir.is_dir():
        return run_dir
    transcript_dirs = sorted(run_dir.glob("*_transcripts"))
    if transcript_dirs:
        return transcript_dirs[0]
    nested = sorted(run_dir.glob("*/**/*_transcripts"))
    return nested[0] if nested else None


def parse_run_spec(spec: str) -> tuple[str, Path]:
    if "=" in spec:
        label, path = spec.split("=", 1)
        return label.strip(), Path(path.strip())
    path = Path(spec)
    return path.name, path


def load_runs(run_specs: list[str], tasks: list[str]) -> dict[str, Any]:
    runs = []
    for spec in run_specs:
        label, run_dir = parse_run_spec(spec)
        transcript_dir = find_transcript_dir(run_dir)
        result = load_result_json(run_dir if not run_dir.name.endswith("_transcripts") else run_dir.parent)
        scores = score_by_task(result)
        run_data = {
            "label": label,
            "run_dir": str(run_dir),
            "transcript_dir": str(transcript_dir) if transcript_dir else None,
            "model": result.get("model"),
            "run_id": result.get("run_id"),
            "summary": {
                "category_scores": result.get("category_scores") or {},
                "suite": result.get("suite"),
                "timestamp": result.get("timestamp"),
            },
            "tasks": {},
        }
        for task_id in tasks:
            path = transcript_dir / f"{task_id}.jsonl" if transcript_dir else Path("__missing__")
            parsed = parse_transcript(path)
            parsed["score"] = scores.get(task_id, {})
            run_data["tasks"][task_id] = parsed
        runs.append(run_data)
    return {"tasks": tasks, "runs": runs}


def render_html(data: dict[str, Any]) -> str:
    data_json = json.dumps(data, ensure_ascii=False)
    escaped = html.escape(data_json, quote=False)
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>val_5 Trajectory Compare</title>
  <style>
    :root {{
      --bg: #f7f7f4;
      --panel: #ffffff;
      --ink: #22252a;
      --muted: #6d7280;
      --line: #d9d9d2;
      --accent: #0f766e;
      --assistant: #eef6ff;
      --tool: #f4f1ff;
      --user: #fff7df;
      --thinking: #f1f5f9;
      --error: #fff1f2;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      background: var(--bg);
      color: var(--ink);
      font-size: 14px;
      line-height: 1.45;
    }}
    header {{
      position: sticky;
      top: 0;
      z-index: 10;
      background: rgba(247, 247, 244, 0.96);
      border-bottom: 1px solid var(--line);
      padding: 14px 18px 12px;
    }}
    h1 {{
      margin: 0 0 10px;
      font-size: 20px;
      font-weight: 700;
      letter-spacing: 0;
    }}
    .controls, .tabs {{
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      align-items: center;
    }}
    .tabs {{ margin-top: 10px; }}
    button, label.toggle {{
      border: 1px solid var(--line);
      background: var(--panel);
      color: var(--ink);
      border-radius: 7px;
      padding: 7px 10px;
      cursor: pointer;
      font: inherit;
    }}
    button.active {{
      border-color: var(--accent);
      background: #e6f4f1;
      color: #064e47;
      font-weight: 650;
    }}
    input[type="search"] {{
      flex: 1 1 280px;
      max-width: 520px;
      border: 1px solid var(--line);
      border-radius: 7px;
      padding: 8px 10px;
      font: inherit;
      background: var(--panel);
    }}
    main {{ padding: 16px 18px 28px; }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(3, minmax(240px, 1fr));
      gap: 10px;
      margin-bottom: 14px;
    }}
    .summary-card {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 12px;
    }}
    .summary-card h2 {{
      margin: 0 0 8px;
      font-size: 15px;
    }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 6px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 6px;
      background: #fbfbfa;
      min-height: 48px;
    }}
    .metric b {{ display: block; font-size: 16px; }}
    .metric span {{ color: var(--muted); font-size: 12px; }}
    .columns {{
      display: grid;
      grid-template-columns: repeat(3, minmax(320px, 1fr));
      gap: 12px;
      align-items: start;
    }}
    .column {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      min-width: 0;
      overflow: hidden;
    }}
    .column-head {{
      position: sticky;
      top: 104px;
      background: var(--panel);
      border-bottom: 1px solid var(--line);
      padding: 10px 12px;
      z-index: 3;
    }}
    .column-head h3 {{
      margin: 0 0 5px;
      font-size: 16px;
    }}
    .small {{ color: var(--muted); font-size: 12px; }}
    .score {{
      font-weight: 700;
      color: var(--accent);
    }}
    .timeline {{ padding: 10px; }}
    .event {{
      border: 1px solid var(--line);
      border-radius: 8px;
      margin-bottom: 9px;
      overflow: hidden;
      background: #fff;
    }}
    .event.user {{ background: var(--user); }}
    .event.assistant {{ background: var(--assistant); }}
    .event.toolResult {{ background: var(--tool); }}
    .event.parse_error, .event.error {{ background: var(--error); }}
    .event-head {{
      display: flex;
      justify-content: space-between;
      gap: 8px;
      padding: 7px 9px;
      border-bottom: 1px solid rgba(0,0,0,0.07);
      font-weight: 650;
    }}
    .event-body {{ padding: 8px 9px; }}
    .block {{
      margin: 6px 0;
      border-radius: 6px;
      background: rgba(255,255,255,0.62);
      border: 1px solid rgba(0,0,0,0.08);
      overflow: hidden;
    }}
    .block-title {{
      padding: 5px 7px;
      font-size: 12px;
      color: var(--muted);
      border-bottom: 1px solid rgba(0,0,0,0.07);
      background: rgba(255,255,255,0.55);
    }}
    .block-body {{
      padding: 7px;
      white-space: pre-wrap;
      word-break: break-word;
      max-height: 320px;
      overflow: auto;
    }}
    .thinking-block {{ background: var(--thinking); }}
    .tool-name {{
      display: inline-block;
      padding: 2px 6px;
      border-radius: 999px;
      background: #e7e5ff;
      color: #3730a3;
      font-size: 12px;
      font-weight: 700;
    }}
    .notes {{
      margin-top: 8px;
      color: #3f4652;
      max-height: 120px;
      overflow: auto;
      white-space: pre-wrap;
    }}
    details {{ margin-top: 8px; }}
    summary {{ cursor: pointer; color: var(--accent); font-weight: 650; }}
    pre {{
      margin: 0;
      white-space: pre-wrap;
      word-break: break-word;
      font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace;
      font-size: 12px;
    }}
    .hidden {{ display: none !important; }}
    @media (max-width: 1100px) {{
      .summary, .columns {{ grid-template-columns: 1fr; }}
      .column-head {{ position: static; }}
    }}
  </style>
</head>
<body>
<header>
  <h1>val_5 Trajectory Compare</h1>
  <div class="controls">
    <input id="search" type="search" placeholder="Filter text, tool name, arguments, notes">
    <label class="toggle"><input id="showThinking" type="checkbox" checked> Thinking</label>
    <label class="toggle"><input id="showToolResults" type="checkbox" checked> Tool results</label>
    <label class="toggle"><input id="showUser" type="checkbox" checked> User prompt</label>
  </div>
  <div id="tabs" class="tabs"></div>
</header>
<main>
  <section id="summary" class="summary"></section>
  <section id="columns" class="columns"></section>
</main>
<script id="data" type="application/json">{escaped}</script>
<script>
const DATA = JSON.parse(document.getElementById('data').textContent);
let activeTask = DATA.tasks[0];

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;', "'":'&#39;'}}[c]));
const pct = (v) => (typeof v === 'number' ? (v * 100).toFixed(1) + '%' : 'n/a');
const num = (v) => (typeof v === 'number' ? (Math.round(v * 10) / 10).toString() : 'n/a');
const json = (v) => esc(JSON.stringify(v ?? {{}}, null, 2));

function initTabs() {{
  const tabs = $('tabs');
  tabs.innerHTML = '';
  DATA.tasks.forEach(task => {{
    const btn = document.createElement('button');
    btn.textContent = task.replace('task_meeting_', '');
    btn.dataset.task = task;
    btn.onclick = () => {{
      activeTask = task;
      render();
    }};
    tabs.appendChild(btn);
  }});
}}

function renderSummary() {{
  const wrap = $('summary');
  wrap.innerHTML = DATA.runs.map(run => {{
    const item = run.tasks[activeTask] || {{}};
    const score = item.score || {{}};
    const metrics = item.metrics || {{}};
    const usage = score.usage || {{}};
    return `<article class="summary-card">
      <h2>${{esc(run.label)}} <span class="small">${{esc(run.model || '')}}</span></h2>
      <div class="metric-grid">
        <div class="metric"><b>${{pct(score.score)}}</b><span>score</span></div>
        <div class="metric"><b>${{metrics.assistant_turns ?? 'n/a'}}</b><span>assistant</span></div>
        <div class="metric"><b>${{metrics.tool_calls ?? 'n/a'}}</b><span>tool calls</span></div>
        <div class="metric"><b>${{num(score.execution_time)}}</b><span>seconds</span></div>
      </div>
      <div class="small" style="margin-top:8px">tools: ${{esc(Object.entries(metrics.tool_counts || {{}}).map(([k,v]) => `${{k}}:${{v}}`).join(', ') || 'none')}}</div>
      <div class="small">requests: ${{usage.request_count ?? 'n/a'}} / tokens: ${{usage.total_tokens ?? 'n/a'}}</div>
      <div class="notes">${{esc(score.notes || '')}}</div>
      <details><summary>grading breakdown</summary><pre>${{json(score.breakdown)}}</pre></details>
    </article>`;
  }}).join('');
}}

function blockHtml(block) {{
  if (block.type === 'thinking') {{
    return `<div class="block thinking-block thinking"><div class="block-title">thinking</div><div class="block-body">${{esc(block.text)}}</div></div>`;
  }}
  if (block.type === 'text') {{
    return `<div class="block"><div class="block-title">text</div><div class="block-body">${{esc(block.text)}}</div></div>`;
  }}
  if (block.type === 'toolCall') {{
    const call = block.tool_call || {{}};
    return `<div class="block"><div class="block-title"><span class="tool-name">${{esc(call.name)}}</span> tool call</div><div class="block-body"><pre>${{json(call.arguments)}}</pre></div></div>`;
  }}
  return '';
}}

function messageHtml(msg, idx) {{
  const role = msg.role || msg.kind || 'unknown';
  const title = role === 'toolResult'
    ? `<span class="tool-name">${{esc(msg.tool_name || 'tool')}}</span> result`
    : esc(role);
  let body = '';
  if (msg.blocks && msg.blocks.length) body += msg.blocks.map(blockHtml).join('');
  if (role === 'toolResult') {{
    body += `<div class="block"><div class="block-title">${{msg.is_error ? 'error' : 'tool output'}}</div><div class="block-body">${{esc(msg.text || '')}}</div></div>`;
    if (msg.details) body += `<details><summary>details</summary><pre>${{json(msg.details)}}</pre></details>`;
  }}
  if (!body && msg.text) body = `<div class="block"><div class="block-body">${{esc(msg.text)}}</div></div>`;
  const haystack = esc(JSON.stringify(msg).toLowerCase());
  return `<article class="event ${{esc(role)}}${{msg.is_error ? ' error' : ''}}" data-role="${{esc(role)}}" data-search="${{haystack}}">
    <div class="event-head"><span>#${{idx + 1}} ${{title}}</span><span class="small">${{esc(msg.timestamp || '')}}</span></div>
    <div class="event-body">${{body}}</div>
  </article>`;
}}

function renderColumns() {{
  const q = $('search').value.trim().toLowerCase();
  const columns = $('columns');
  columns.innerHTML = DATA.runs.map(run => {{
    const item = run.tasks[activeTask] || {{}};
    const score = item.score || {{}};
    const messages = item.messages || [];
    return `<section class="column">
      <div class="column-head">
        <h3>${{esc(run.label)}} <span class="score">${{pct(score.score)}}</span></h3>
        <div class="small">${{esc(item.meta?.path || '')}}</div>
      </div>
      <div class="timeline">
        ${{messages.map(messageHtml).join('') || '<div class="small">No transcript loaded.</div>'}}
      </div>
    </section>`;
  }}).join('');

  document.querySelectorAll('.event').forEach(el => {{
    const role = el.dataset.role;
    const text = el.dataset.search || '';
    let hide = false;
    if (!$('showToolResults').checked && role === 'toolResult') hide = true;
    if (!$('showUser').checked && role === 'user') hide = true;
    if (q && !text.includes(q)) hide = true;
    el.classList.toggle('hidden', hide);
  }});
  document.querySelectorAll('.thinking').forEach(el => {{
    el.classList.toggle('hidden', !$('showThinking').checked);
  }});
}}

function render() {{
  document.querySelectorAll('#tabs button').forEach(btn => btn.classList.toggle('active', btn.dataset.task === activeTask));
  renderSummary();
  renderColumns();
}}

['search', 'showThinking', 'showToolResults', 'showUser'].forEach(id => $(id).addEventListener('input', renderColumns));
initTabs();
render();
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a val_5 trajectory comparison HTML page.")
    parser.add_argument(
        "--run",
        action="append",
        dest="runs",
        help="Run spec LABEL=PATH. PATH can be a run dir or a *_transcripts dir. Repeat for each model.",
    )
    parser.add_argument("--output", default="/tmp/val5_trajectory_compare.html", help="Output HTML path.")
    parser.add_argument("--task", action="append", dest="tasks", help="Task id to include. Defaults to val_5.")
    args = parser.parse_args()

    run_specs = args.runs or DEFAULT_RUNS
    tasks = args.tasks or VAL5_TASKS
    data = load_runs(run_specs, tasks)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_html(data), "utf-8")
    print(f"Wrote {output}")
    print("Runs:")
    for run in data["runs"]:
        print(f"  - {run['label']}: {run['run_dir']}")
    print("Tasks:")
    for task in tasks:
        print(f"  - {task}")


if __name__ == "__main__":
    main()
