# Harness Pitfalls in On-Policy Rollout Generation (post-mortem, 2026-06-16)

During on-policy rollout generation for the committee-reward RL rounds, ~25% of rollouts came
back EMPTY (no deliverable written, `automated_score=0`). Audit of the graded files
(`onpolicy_w2` — which w2–w6 trained on — and `committee_w7`) showed the SAME ~25% empty rate;
`round-1e` base rollouts were clean (0 empty). Three distinct causes, with fixes. **All three
silently poison the GRPO signal** (empties inject noisy 0-scores), so guard against them.

## Cause 1 (PRIMARY) — rollout sampling temperature = 1.0 → premature termination
- The rollout shim served at `SHIM_DEFAULT_TEMP=1.0` (chosen for diversity). At temp=1.0 the
  agent frequently **reads the transcript a few times then ENDS its turn WITHOUT writing the
  deliverable** (confirmed on a tech rollout: 13 events, 3 `read` calls, 0 `write`, exit_code=0
  "success", not timed out, ~20K doc — so not a length/timeout problem, purely erratic sampling).
- The eval harness used temp=0.3 and had a near-zero empty rate — that's why the pollution was
  invisible until we audited the *training* rollouts.
- **FIX:** lower rollout temp. `run_next_round.sh` now uses `ROLLOUT_TEMP=${ROLLOUT_TEMP:-0.7}`
  (still ample within-group diversity for GRPO; far fewer terminations). Do NOT use temp=1.0 for
  agentic rollouts that must end in a file write.

## Cause 2 — long transcripts (71K chars) overflow context → timeout
- advisory / NASA tasks (~71K-char transcripts) fill the context to ~94%; the LLM times out and
  the harness attempts "compaction before retry," but the old wall-clock cap (`ROLLOUT_TIMEOUT_MULT=4.0`
  → 180×4 = 720s) was too short — rollouts died at ~734s.
- **NON-FIX (corrected 2026-06-16):** raising the cap does NOT help — the 71K docs are CONTEXT-bound
  (~96% window), not time-bound; they never complete, so a higher timeout (we tried 6.0=1080s) only
  wastes ~370s per doomed rollout. DATA: every SUCCESSFUL rollout finishes ≤593s (w2 max 593, w7 max
  479). **Correct timeout = `ROLLOUT_TIMEOUT_MULT=4.0` (720s)**: covers all completable rollouts +
  margin, fails the doomed long-docs fast. The actual long-doc fix is structural (see below), not timeout:
  (a) larger effective context window; (b) pre-chunk / map-reduce the transcript; (c) partial-credit
  reward so an incomplete long-doc rollout still gives a gradient instead of a 0.
  **Do NOT "skip" perpetually-dead long-doc groups (e.g. speaker_nasa) via an exclusion list.** That is
  a landmine: a silent permanent exclusion that — if the memory/list isn't diligently maintained — means
  the ultra-long tasks are NEVER trained again, and biases the training set toward what the model already
  does (inflated metrics, hidden weakness). Instead KEEP every task in the set (so it is re-attempted and
  self-documents each round) and convert the wasted time into signal via **partial-credit reward** (score
  by completion fraction even on a timeout) or a bigger effective context window. The long-doc time is not
  "waste to skip" — it's "signal we currently throw away by scoring timeouts as 0." Fix the scoring, not the task list.

## Cause 3 (LATENT) — inconsistent transcript filename across tasks
- The 9 training tasks used THREE different transcript `dest` filenames:
  `meeting-transcript.md` (hyphen), `meeting_transcript.md` (underscore), `transcript.md`.
  Within a task, prompt↔dest were consistent, so this was not the direct cause of the empties
  observed — but it is a latent confound (an agent that has seen multiple tasks can hallucinate
  the wrong name; we saw beyond-EOF reads and a doubled `agent_workspace/agent_workspace/` path).
- **FIX:** normalized ALL tasks' transcript dest + prompt references to a single canonical name
  `meeting_transcript.md` (`fix_transcript_names.py`). Keep input transcript filenames identical
  across every task.

## Lesson (env / process)
- **Audit rollout health on the TRAINING distribution, not just eval.** Eval (temp=0.3) hid a 25%
  training-time empty rate. `rollout_healthcheck.py` flags all-timeout / no-write groups but a
  per-group ≥2-good survival can still mask 25% per-rollout waste + noisy 0-scores.
- **Empties are NOT free "diversity."** A 0-vs-0.7 spread looks like signal to GRPO but is noise
  (random termination / path fumble), teaching "write something" not the quality ranking we want.
- **Keep agentic rollout temp moderate (≤0.7).** temp=1.0 trades a 25% completion-failure tax for
  diversity you don't need.
- **One canonical input filename across all tasks.** Filename drift is a silent agent-confusion tax.
- Consider a durable **empty-rollout retry** in the rollout driver (regenerate any rollout with no
  deliverable, up to N retries) as a belt-and-suspenders fix on top of the temp reduction.
