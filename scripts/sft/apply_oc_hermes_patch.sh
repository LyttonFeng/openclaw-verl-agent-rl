#!/bin/bash
# Apply PATCH-B to OC's pi-ai openai-completions provider.
# See oc_hermes_patch.md for full RCA.
set -euo pipefail

PROVIDER="${OC_PROVIDER_JS:-/usr/local/lib/node_modules/openclaw/node_modules/@mariozechner/pi-ai/dist/providers/openai-completions.js}"
BAK="$(dirname "$PROVIDER")/.oc_hermes_patch.bak"

if [ ! -f "$PROVIDER" ]; then
  echo "ERROR: provider not found at $PROVIDER"
  echo "Override path with OC_PROVIDER_JS=... if installed elsewhere"
  exit 1
fi

if grep -q 'PATCH-B' "$PROVIDER"; then
  echo "PATCH-B already applied to $PROVIDER"
  exit 0
fi

cp "$PROVIDER" "$BAK"
echo "Backup: $BAK"

python3 << 'PYEOF'
import os, sys
p = os.environ.get('OC_PROVIDER_JS') or '/usr/local/lib/node_modules/openclaw/node_modules/@mariozechner/pi-ai/dist/providers/openai-completions.js'
src = open(p).read()
old = '            finishCurrentBlock(currentBlock);\n            if (options?.signal?.aborted) {'
new = '''            finishCurrentBlock(currentBlock);
            // PATCH-B: extract <tool_call>...</tool_call> from text blocks
            // (hermes-parser fallback for when vLLM doesn't extract tool_calls)
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
                                        id: 'extracted-' + Math.random().toString(36).slice(2,10),
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
            if (options?.signal?.aborted) {'''
if old not in src:
    print('ERROR: anchor pattern not found in', p); sys.exit(1)
open(p, 'w').write(src.replace(old, new))
print('OK: patched', p)
PYEOF

# Clear jiti compile cache so OC re-reads from npm source
JITI_GLOB="/tmp/jiti/providers-openai-completions.*.cjs"
shopt -s nullglob
for f in $JITI_GLOB; do
  echo "Removing jiti cache: $f"
  rm -f "$f"
done

echo
echo "PATCH-B applied. Restart any running OC / vLLM processes for it to take effect."
echo "Rollback: cp $BAK $PROVIDER && rm -f $JITI_GLOB"
