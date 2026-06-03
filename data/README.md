# data

This directory contains task data used by the training and evaluation workflow.

It should contain training tasks, evaluation tasks, task prompts, rubrics, rules, source meeting materials, and any derived datasets that are required to reproduce the experiment.

Current evaluation data:

- `eval/val5/`: the five meeting-analysis Val5 task definitions used by `scripts/run_val5_bench_isolated.sh`.
- `eval/assets/meetings/`: the source meeting transcripts referenced by those Val5 task definitions through `workspace_files`.
