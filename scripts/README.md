# scripts

This directory contains operational scripts.

It should contain small, reproducible command wrappers for serving models, launching training jobs, running benchmarks, aggregating scores, and exporting artifacts.

Current benchmark files:

- `run_val5_bench_isolated.sh`: entrypoint for the isolated meeting-analysis Val5 benchmark. It runs the five Val5 tasks three times by default, uses a private OpenClaw home/run root/agent suffix, disables upload, disables judge cache, and grades synchronously for reproducibility.
- `benchmark.py`: benchmark runner used by the isolated wrapper. It loads task definitions, creates the OpenClaw agent, executes each task/run, invokes grading, and writes incremental/final result JSON.
- `lib_agent.py`: OpenClaw execution helper used by `benchmark.py`. It creates model-backed OpenClaw agents, writes model configuration, prepares task workspaces, invokes the OpenClaw CLI, captures transcripts, and calls judge APIs.
- `lib_grading.py`: grading helper used by `benchmark.py`. It runs automated checks from task markdown and LLM-judge grading through the configured judge backend.
- `lib_tasks.py`: task loader used by `benchmark.py`. It parses task markdown frontmatter, prompts, automated checks, and judge rubrics.
