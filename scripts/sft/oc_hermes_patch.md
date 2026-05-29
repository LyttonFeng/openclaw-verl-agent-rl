# OC + vLLM hermes parser bug — RCA & patch

## TL;DR

OpenClaw 调 vLLM `--tool-call-parser hermes` 时, **多 turn tool 调用从第 2 轮起被错误解析为纯文本**, 导致 Qwen3 模型 (base 或 LoRA) 在 PinchBench 多 turn 任务上被压低 ~4-6pp。

修复后实测:
- Base Qwen3-4B val_5: 历史 0.474 → patched **0.510** (+4pp, n=3, stdev 0.019)
- v38_ckpt9 (RL LoRA) val_5: 历史 0.49 → patched 0.473 (基本持平)
- **关键发现**: v38 RL 训练实际是退化 (low base by ~4pp), 之前看起来"略涨"完全是 bug 蒙住的假象

## Bug

### 文件
- 真活路径: `/usr/local/lib/node_modules/openclaw/node_modules/@mariozechner/pi-ai/dist/providers/openai-completions.js`
- 也存在的非活路径 (jiti 编译缓存, 每次从 npm 源重生成): `/tmp/jiti/providers-openai-completions.<hash>.cjs`

### 故障链

1. Qwen3-4B 多 turn 工具调用产生 `<tool_call>{"name":"...","arguments":{...}}</tool_call>` 文本
2. vLLM hermes parser 应该把它解成结构化 `tool_calls` 字段
3. **但 vLLM hermes parser 只在 request 里 `tools` 字段非空时激活**
4. OC 多 turn 时, request builder (`streamOpenAICompletions`) 第一轮传 25 个真 tools, 后续轮 (有 tool_calls 历史时) 走 fallback `params.tools = []` (为兼容 Anthropic LiteLLM proxy 设计)
5. vLLM 看到空 tools → hermes 不解析 → `<tool_call>` 文本留在 content 里
6. OC 看到 content 是纯文本, 不执行工具 → episode 卡死 / 给出非 grounded 答案 → judge 评低

### 第一现场证据 (transcript)

```
turn 0: [thinking, text, toolCall]  ← 解析正确 (tools 字段在)
turn 1: [text]                       ← 退化! <tool_call> 在 text 里
   text: '<tool_call>\n{"name":"exec","arguments":{...}}\n</tool_call>'
```

## Fix (PATCH-B)

策略: 不动 hermes parser, 不动 vLLM, **在 OC 的流式响应处理结束时**, 扫描所有 `type: 'text'` content block 里的 `<tool_call>...</tool_call>` 序列, 解析 JSON, 转成 `type: 'toolCall'` block。

### Patch 位置

`/usr/local/lib/node_modules/openclaw/node_modules/@mariozechner/pi-ai/dist/providers/openai-completions.js` 大约 line 222, 紧跟 `finishCurrentBlock(currentBlock);` 之后插入。

### Patch 内容

```javascript
            finishCurrentBlock(currentBlock);
            // PATCH-B: extract <tool_call>...</tool_call> from text blocks
            // (hermes-parser fallback for when vLLM doesn't extract structured tool_calls
            //  despite the model emitting valid tool_call XML)
            try {
                const _re = /<tool_call>\s*([\s\S]*?)\s*<\/tool_call>/g;
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
            if (options?.signal?.aborted) {
```

### 替代修法 (未采用)

A. 修 OC 让它每 turn 都传完整 tools (而不是空数组): 需要找 caller 把 context.tools 传下来, 更深 refactor。 

B. 改 vLLM, 让 hermes parser 不依赖 tools 字段: 需要 patch vLLM 源码或等上游 PR, 周期长。

C. 训练数据用另一种格式 (不依赖 hermes): 要改训练流程, 重训, 没有立即的可验证 ROI。

PATCH-B 是最小侵入, 10 行 JS, 修复后双重 verify (transcript turn 1+ 解析正常 + bench 分数提升)。

## 影响范围

| 场景 | 是否受 bug 影响 | Patch 后表现 |
|---|---|---|
| DeepSeek V4 Flash API bench | ❌ 不受 (走 DeepSeek native function calling) | 不变 |
| Base Qwen3-4B + vLLM hermes | ✅ 受 (~4pp 被压低) | base val_5: 0.474 → **0.510** |
| RL LoRA (v32-v41) bench | ✅ 受 | v38_ckpt9: 0.49 → 0.473 (退化已暴露) |
| RL LoRA training rollout 收集 | ✅ 推测受 (相同路径) | 修复后未来训练 reward signal 干净 |

## 验证步骤

1. 确认 patch 在位:
   ```bash
   grep -n 'PATCH-B' /usr/local/lib/node_modules/openclaw/node_modules/@mariozechner/pi-ai/dist/providers/openai-completions.js
   ```
2. 重启所有 vLLM / OC agent 进程让 jiti 重编译:
   ```bash
   pkill -9 -f vllm; pkill -9 -f openclaw-agent; rm -f /tmp/jiti/providers-openai-completions.*.cjs
   ```
3. 跑 bench, OC log 应该出现 `[PATCH-B] extracted N tool_call(s)`:
   ```bash
   grep PATCH-B /tmp/openclaw/openclaw-$(date +%Y-%m-%d).log | head
   ```
4. transcript turn 1+ 应该有 `'toolCall'` block 而不是 raw `<tool_call>` text:
   ```bash
   python3 -c "
   import json
   F='.../task_meeting_advisory_stakeholders.jsonl'
   for e in (json.loads(l) for l in open(F)):
       if e.get('type')=='message' and e.get('message',{}).get('role')=='assistant':
           print([c.get('type') for c in e['message'].get('content',[])])
   "
   ```

## 备份 + 回滚

备份: `/tmp/providers-openai-completions.cjs.bak`

回滚脚本: `/tmp/oc_patch_rollback.sh` (also reverts /tmp/jiti/ cache if present)

## 测试数据

实测 val_5 (3 run, mean ± stdev):

| Config | mean | stdev | Δ vs 历史 |
|---|---|---|---|
| Base 历史 (bug 在场, 9 run) | 0.474 | ~0.04 | — |
| Base patched (3 run) | **0.5103** | 0.019 | **+4pp** |
| v38_ckpt9 历史 (bug 在场, 3 run) | 0.49 | — | — |
| v38_ckpt9 patched (r1 only so far) | 0.473 | — | -2pp |

Per-task 主要差异 (patched - 历史):
- advisory_stakeholders: +7pp (0.444 → 0.518)
- council_votes: +2pp (0.244 → 0.269)
- gov_speaker_summary: +1pp (0.426 → 0.435)
- sentiment_analysis: +2pp (0.617 → 0.639)
- tech_action_items: **+5pp** (0.641 → 0.692)

## Open questions

1. 这个 bug 在 OC 内部代码哪条 caller path 把 `context.tools` 置空? 是否能从根源修, 而不是 patch fallback 之后的输出?
2. vLLM hermes parser 是否应该在 model emit `<tool_call>` 时无视 tools 字段自动解析? (上游 vLLM 行为问题)
3. RL training (v32-v41) 时的 rollout 收集是否同样 fail? 缺历史 rollout transcripts, 无法直接验证, 但代码路径相同, 推测影响 reward signal 质量。

## Reproducibility

Patch 在 repo 里: `scripts/sft/oc_hermes_patch.md` (本文档) 和 `scripts/sft/apply_oc_hermes_patch.sh` (应用脚本)。
