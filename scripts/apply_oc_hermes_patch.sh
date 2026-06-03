#!/usr/bin/env bash
# Patch OpenClaw's OpenAI-compatible provider so Qwen3 hermes text fallback
# tool calls are converted into OpenClaw toolCall blocks.
#
# Why this exists:
#   Qwen3-4B can emit <tool_call>{...}</tool_call> text. vLLM's hermes parser
#   usually converts that into structured tool_calls, but OpenClaw multi-turn
#   requests can reach vLLM with an empty tools list. In that case vLLM leaves
#   the <tool_call> XML in text, and OpenClaw will not execute the tool unless
#   this fallback patch extracts it.

set -euo pipefail

PROVIDER="${OC_PROVIDER_JS:-/usr/local/lib/node_modules/openclaw/node_modules/@mariozechner/pi-ai/dist/providers/openai-completions.js}"
BAK="$(dirname "$PROVIDER")/.oc_hermes_patch.bak"

if [ ! -f "$PROVIDER" ]; then
  echo "ERROR: OpenClaw provider not found at $PROVIDER" >&2
  echo "Set OC_PROVIDER_JS=/path/to/openai-completions.js if OpenClaw is installed elsewhere." >&2
  exit 1
fi

if grep -q "PATCH-B" "$PROVIDER"; then
  echo "PATCH-B already applied: $PROVIDER"
  exit 0
fi

cp "$PROVIDER" "$BAK"
echo "Backup: $BAK"

python3 <<'PY'
import os
import sys

path = os.environ.get(
    "OC_PROVIDER_JS",
    "/usr/local/lib/node_modules/openclaw/node_modules/@mariozechner/pi-ai/dist/providers/openai-completions.js",
)
src = open(path, encoding="utf-8").read()
old = "            finishCurrentBlock(currentBlock);\n            if (options?.signal?.aborted) {"
new = """            finishCurrentBlock(currentBlock);
            // PATCH-B: extract <tool_call>...</tool_call> from text blocks.
            // This is a hermes-parser fallback for Qwen3 when vLLM leaves
            // valid tool_call XML in plain text during multi-turn sessions.
            try {
                const _re = /<tool_call>\\s*([\\s\\S]*?)\\s*<\\/tool_call>/g;
                const _newContent = [];
                let _patched = 0;
                for (const _block of output.content) {
                    if (_block.type === 'text' && typeof _block.text === 'string' && _block.text.indexOf('<tool_call>') !== -1) {
                        let _lastIdx = 0;
                        let _m;
                        _re.lastIndex = 0;
                        while ((_m = _re.exec(_block.text)) !== null) {
                            const _before = _block.text.slice(_lastIdx, _m.index);
                            if (_before.trim()) _newContent.push({ type: 'text', text: _before });
                            try {
                                const _tc = JSON.parse(_m[1]);
                                if (_tc && _tc.name) {
                                    _newContent.push({
                                        type: 'toolCall',
                                        id: 'extracted-' + Math.random().toString(36).slice(2, 10),
                                        name: _tc.name,
                                        arguments: _tc.arguments || _tc.parameters || {}
                                    });
                                    _patched += 1;
                                } else {
                                    _newContent.push({ type: 'text', text: _m[0] });
                                }
                            } catch (_e) {
                                _newContent.push({ type: 'text', text: _m[0] });
                            }
                            _lastIdx = _m.index + _m[0].length;
                        }
                        const _after = _block.text.slice(_lastIdx);
                        if (_after.trim()) _newContent.push({ type: 'text', text: _after });
                    } else {
                        _newContent.push(_block);
                    }
                }
                if (_patched > 0) {
                    console.error('[PATCH-B] extracted ' + _patched + ' tool_call(s) from text');
                    output.content = _newContent;
                }
            } catch (_err) {
                console.error('[PATCH-B] err: ' + (_err && _err.message));
            }
            if (options?.signal?.aborted) {"""

if old not in src:
    print(f"ERROR: patch anchor not found in {path}", file=sys.stderr)
    sys.exit(1)

open(path, "w", encoding="utf-8").write(src.replace(old, new))
print(f"PATCH-B applied: {path}")
PY

shopt -s nullglob
for f in /tmp/jiti/providers-openai-completions.*.cjs; do
  echo "Removing jiti cache: $f"
  rm -f "$f"
done

echo "Restart OpenClaw/vLLM processes after applying the patch."
