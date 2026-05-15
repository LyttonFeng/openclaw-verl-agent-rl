# External library patches for verl-async + jiuwenclaw training

These patches live OUTSIDE this repo on the training pod — apply them after pulling
fresh verl / jiuwenclaw sources. Discovered during v68 debugging.

## `verl_rollouter_staleness_decrement.patch`
**Target**: `/root/verl/verl/experimental/fully_async_policy/fully_async_rollouter.py`

**Bug**: `staleness_samples` counter incremented when a sample is dispatched, but
NEVER decremented when the sample is dropped (empty response_mask or MQ-put
failure). Drops permanently occupy the pause budget → rollouter pauses at
`staleness >= max_required_samples` and never resumes → deadlock at gen plateau.

**Fix**: decrement `self.staleness_samples` in both drop branches of
`_process_single_sample_streaming`.

## `jiuwenclaw_interface_deep_progressive_tools.patch`
**Target**: `/root/jiuwen_work/jiuwenclaw/jiuwenclaw/server/runtime/agent_adapter/interface_deep.py`

**Goal**: expose OpenJiuwen's built-in `ProgressiveToolRail` via env vars so the
system prompt shrinks from ~25k tokens (~30 tool defs) to ~7.7k (just the
whitelisted core tools + meta `search_tools` / `load_tools`).

**Activation** (jiuwen stack env):
```
JIUWENCLAW_PROGRESSIVE_TOOLS=1
JIUWENCLAW_PROGRESSIVE_VISIBLE=read_file,write_file,list_files,edit_file,grep,glob,todo_create,todo_list,code,write_memory,memory_search
JIUWENCLAW_PROGRESSIVE_ALWAYS=
JIUWENCLAW_PROGRESSIVE_MAX=12
```

Applied at both `DeepAgentConfig(...)` construction sites in `interface_deep.py`
(lines ~1925 and ~2256).
