#!/usr/bin/env python3
"""Make the Qwen3.5 chat template default to NON-THINK.

Stock template: when `enable_thinking` is omitted it emits a bare `<think>\\n`
(thinking ON). OpenClaw/pi-ai (openai-completions.js) only sends
`chat_template_kwargs.enable_thinking` when `model.reasoning=true`; with
reasoning=false it omits the field entirely, so the stock template silently
runs thinking-ON -- which on Qwen3.5-4B is degenerate (burns the token budget
without closing </think>, returns empty content).

This patch flips the default: thinking only when a caller explicitly passes
enable_thinking=true; otherwise inject a closed empty `<think></think>` block
(non-think). Idempotent.

Usage: patch_qwen35_template_nothink.py [path/to/chat_template.jinja]
       (default: /tmp/qwen3.5-4b/chat_template.jinja)
"""
import sys, shutil, pathlib

p = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "/tmp/qwen3.5-4b/chat_template.jinja")
s = p.read_text()

OLD = (
    "    {%- if enable_thinking is defined and enable_thinking is false %}\n"
    "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
    "    {%- else %}\n"
    "        {{- '<think>\\n' }}\n"
    "    {%- endif %}\n"
)
NEW = (
    "    {%- if enable_thinking is defined and enable_thinking is true %}\n"
    "        {{- '<think>\\n' }}\n"
    "    {%- else %}\n"
    "        {{- '<think>\\n\\n</think>\\n\\n' }}\n"
    "    {%- endif %}\n"
)

if NEW in s:
    print("already patched (default=non-think):", p)
    sys.exit(0)
if OLD not in s:
    print("ANCHOR NOT FOUND in", p, "-- template differs; aborting", file=sys.stderr)
    sys.exit(1)

shutil.copy2(p, str(p) + ".think_default.bak")
p.write_text(s.replace(OLD, NEW, 1))
print("patched: default is now NON-THINK (think only when enable_thinking=true):", p)
