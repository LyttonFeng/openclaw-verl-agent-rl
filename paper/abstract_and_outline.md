# Paper draft scaffold — DRAFT for review (not final; H-E inclusion TBD by user)

## Working title
Committee Rewards, Automated-Grader Blindness, and the Memory–RL Boundary in Agentic LLMs

## Abstract (draft)
Training small agentic LLMs to improve on long-document tasks (meeting analysis) is hard: rule-based
graders are weak, gameable proxies, and the quality gains that matter are invisible to them. We make
three contributions on qwen3.5-4b over Val3 meeting-analysis (advisory / gov / tech). (1) A stable,
heterogeneous **committee reward** (3 independent LLM judges, pairwise order-consistency + deliberation,
RULER-style listwise for training) detects quality gains that the automated/hybrid grader cannot see:
committee-reward RL beats base on the gov dimension (8:0) while the automated score stays flat — an
"automated-blindness" gap, and chasing the automated score is actively counterproductive (the highest
automated variant is the weakest by committee). (2) **Experiential memory as a training-free harness
baseline**: injecting reviewed, generalizable how-to hints (no answers/entities) retrieved per-task
matches or beats expensive RL on most dimensions, and moves advisory/tech that RL could not — because
those dimensions have near-zero within-group reward variance (flat GRPO gradient). Naively, "memory
subsumes RL." (3) We then show this is **not** the end of the story: on the memory harness, RL CAN net-beat
the memory baseline, but only with the right recipe — **cold-start from base** (a converged adapter has a
saturated, ~2x-smaller gradient) and **policy-gated memory** (the agent learns WHEN to use the hint rather
than being forced to). Gated-RL beats base+memory on tech (7:1), while a forced-memory variant with HIGHER
rollout variance ties everywhere — so the lever is the gating mechanism, not gradient magnitude. The win is
again invisible to the automated grader. We position this within the emerging RL-for-adaptive-memory line
(Memory-R1, DeltaMem) and provide a practical recipe and a set of negative results (hint-rollouts compress
reward variance and cannot be distilled into weights; long-doc instances are uncompleteable and break
on-policy iteration).

## Section outline
1. Intro — weak/gameable graders; the question of whether RL helps over cheap experiential memory.
2. Setup — qwen3.5-4b, Val3 tasks, transformers+PEFT serving (vLLM online-LoRA no-op), committee judge.
3. Committee reward + automated-blindness — gov win invisible to automated; w2 paradox; AUTO_W not a lever; deliberation de-biases reward signal (not final quality).
4. mem0 as harness baseline — red-line-safe hint injection; base+mem beats base; w7+mem ≈ base+mem ("memory subsumes RL"); mem moves the flat-gradient dims RL can't.
5. The memory–RL boundary — hint-rollouts don't restore variance / don't distill (negative); H-A saturation (base vs w7 gradient); forced vs policy-gated; gated-RL beats base+mem on tech; gating > gradient; on-policy trend (direction-robust, magnitude-noisy).
6. Related work — RL for memory gating (Memory-R1, DeltaMem, MemAgent); training-free experience injection.
7. Limitations — committee train/eval same 3 judges (no true held-out; need 4th model); small eval N (9-pair verdicts wobble); synthetic-hard-task extension (H-E) [pending]; single model/task family.
8. Conclusion — when RL is worth it over experiential memory: only where correct behavior is instance-specific and the gradient is unsaturated.

## Open decisions for the human
- Include H-E (instance-specific hard tasks) as a 4th contribution, or leave as future work?
- Venue/format (workshop vs full)? → drives ml-paper-writing template choice.
- Whether to firm the tech win with bigger eval N / a 4th held-out judge before claiming significance.
