# data

This directory contains task data used by the training and evaluation workflow.

It should contain training tasks, evaluation tasks, task prompts, rubrics, rules, source meeting materials, and any derived datasets that are required to reproduce the experiment.

Current evaluation data:

- `eval/val5/`: the five meeting-analysis Val5 task definitions used by `scripts/run_val5_bench_isolated.sh`.
- `eval/assets/meetings/`: the source meeting transcripts referenced by those Val5 task definitions through `workspace_files`.

Current training data:

- `train/meeting_analysis_all_samples_split.json`: the all-samples training split for the naive RL convergence check. It intentionally includes the historical held-out Val5 tasks in `train` so we can test whether RL can overfit/converge on the full meeting-analysis task set.
- `train/tasks/`: all meeting-analysis task definitions used by that all-samples split. These task markdown files contain the prompt, workspace fixtures, automated checks, grading weights, and LLM judge rubrics needed to generate and grade training rollouts.
