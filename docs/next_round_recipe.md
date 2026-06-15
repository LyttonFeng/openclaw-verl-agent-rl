# Next-Round RL Recipe — continue from the committee winner (w6)

This codifies the decision after the 2026-06-15 temp=0.3 committee ablation: how to run
the *next* on-policy RL round on top of the best adapter, and the guardrails that must
not be dropped. Driver: `scripts/tf_agentic/run_next_round.sh` (run on the pod).

## What the ablation concluded (all vs canonical base@0.3, 9-pair committee)

| var | AUTO_W | delib | advisory | gov | tech | MEETING%(auto/hyb) |
|---|---|---|---|---|---|---|
| base | — | — | anchor | anchor | anchor | 79.5 |
| w5 | 0.0 | no | tie~base 5:3* | **WIN 8:1 p=.039** | tie~base 5:2 | 67.9 |
| **w6** | **0.0** | **yes** | tie~lora **6:2** | **WIN 7:1 p=.070** | tie~base 6:3 | 79.7 |
| w4 | 0.2 | no | tie~base 5:3 | **WIN 7:1 p=.070** | tie~base 5:4 | 79.3 |
| w2 | 0.5 | no | tie~lora 5:2 | tie 4:4 | tie~base 5:3 | **81.3** |
| w3 | 0.7 | no | tie~lora 6:2 | **WIN 9:0 p=.004** | tie 4:4 | 74.7 |

\* w5 advisory dragged by a policy read-loop→timeout→empty run (not harness).

- **Committee-reward beats base on gov (4/5 loras), and the win is INVISIBLE to the
  automated grader** (gov auto/hyb flat ~.85/.74 = base across all). Automated is a weak,
  even adversarial proxy — w2 has the *highest* automated (81.3%) but is the *weakest* on
  committee (gov tie). Do not optimize automated.
- **AUTO_W is not the lever** — the gov win is robust across 0.0–0.7. (The earlier temp=0
  "w3 worst" was a base-nondeterminism confound; always anchor one canonical base.)
- **Deliberation is the lever** — w6 vs w5 (same data/init, only delib differs): preserves
  advisory (clean 6:2 vs flake-tie) and tech accuracy (owner-attribution 0.50 vs 0.17 crash).
- **Best = w6 (AUTO_W=0 + deliberation).** Start the next round from it.

## The recipe (defaults in run_next_round.sh)

| knob | value | why |
|---|---|---|
| INIT_ADAPTER | committee_w6 ckpt | continue-train from the winner |
| AUTO_W | 0.0 | pure committee (automated is a weak proxy) |
| DELIBERATE | 1 | the validated stabilizer |
| reward | RULER listwise + llm_rubric + base-ref (放法B) | grounding, anti-hack |
| BASE_REF | base_ref_temp03.jsonl | base@0.3 calibration anchor (not scored) |
| LR | 2.0e-5 (lowered from 2.5e-5) | already ≥2 continue-trains deep; avoid drift |
| KL | anchors to INIT_ADAPTER | don't drift off the current policy |

```bash
# on the pod:
RUN_NAME=committee_w7 bash scripts/tf_agentic/run_next_round.sh   # uses all defaults above
```

## The THREE guardrails (must not drop)

1. **Fresh on-policy rollouts every round.** The shim serves INIT_ADAPTER at temp=1.0 and we
   regenerate K=4 rollouts per task. **Never reuse a prior round's graded_trajectories** — that
   is off-policy and makes "continue from w6" meaningless. (Note: w6 itself reused w2's rollouts;
   the next round fixes that.)
2. **Judge-overfit guard.** The committee can be reward-hacked exactly like automated was
   (round-1e). Division of labor for catching it:
   - **Automated, every round (the agent runs it):** held-out-judge validation — re-judge with a
     committee member excluded, via `JUDGE_MEMBERS`. If the committee win vanishes under the
     held-out judge, suspect overfitting.
     ```bash
     EVAL_TEMP=0.3 RUNS=3 bash scripts/tf_agentic/eval_val3_adapter.sh lora <new_ckpt>
     JUDGE_MEMBERS=qwen-max,minimax-M3 python3 scripts/tf_agentic/committee_judge.py   # ds-flash held out
     ```
   - **Agent LLM spot-read, every round:** the agent reads 1–2 deliverables vs the transcript and
     flags obvious hacking (padding / hallucination / verbose empty prose), raising it to the human.
   - **Human spot-check, periodic (NOT the agent) — scoped to OBVIOUS failures only:** the agent and
     all committee judges are LLMs and may SHARE blind spots, so a human eyeballs 1–2 reports at key
     moments (a "too good" result, or every 2–3 rounds). IMPORTANT — the human's job is to catch
     *obvious* hacking (padding, hallucination, off-topic, empty verbose prose), which humans spot
     reliably. The human does NOT adjudicate subtle ties: if a human also can't tell two reports
     apart, that IS the answer — treat it as a genuine no-difference, not a judge failure.

   **Interpreting verdicts (consequence of the above):** trust only LARGE-margin committee wins
   (gov 7:1 / 8:1 / 9:0 — clear to humans and LLMs alike). Marginal "tie~lora 6:2" is WEAK evidence
   — likely noise in a region of genuine indistinguishability; do not over-interpret or chase it.
   Where differences are not human-discernible, more committee training risks fitting judge quirks,
   not real quality. To get real, checkable gains, create CLEAR differences: use harder task variants
   that base visibly fails, so good-vs-bad is obvious to humans and judges.
3. **Timeout fix + healthcheck.** `ROLLOUT_TIMEOUT_MULT=4.0` (advisory is a 71K-char doc) and
   `rollout_healthcheck.py` stop the round BEFORE training if rollouts are all-timeout / no-write /
   written-but-auto=0. The policy read-loop failure (model re-reads a long doc forever, never
   writes, hits timeout → empty deliverable) WILL recur during rollout generation.

## Where the marginal gains are
- **gov is solved** (significant win, robust). Don't spend effort here.
- **advisory**: lora-favored but not yet significant (6:2). Push with more advisory on-policy
  diversity + fixing the read-loop failure mode.
- **tech**: committee-tie, but it has ~28% headroom — RL moves coverage; the problem is it trades
  away accuracy unless deliberation is on. To make tech *win*, use harder tech tasks (where base
  fails) to create spread/headroom, and reward accurate attribution over coverage.
