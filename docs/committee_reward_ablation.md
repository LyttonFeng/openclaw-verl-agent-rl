# Committee-Reward Training Ablation (qwen3.5-4b, Val3+6)

2026-06-13/14. All variants trained on the same Val3+6 task set (7 healthy groups, K=4).
Adapters backed up on pod `/workspace/saved_adapters/{round1e_val3plus6,committee_w1,committee_blend_w1,onpolicy_w2}`
(+ round-1e mirrored to mac `~/lora_backups/round1e_val3plus6`).
Eval = pairwise committee (committee_judge.py) + automated/hybrid (stable_rejudge.py). base = qwen3.5-4b, no adapter.
Eval is temp=0 greedy → 1 deterministic report/side; the "3×3=9 pairs" are repeated samplings of one
comparison (direction reliable, sign-p optimistic).

## Lineage / ablation table

| variant | init | rollout data | reward | base-ref | result vs base |
|---|---|---|---|---|---|
| round-1e | base (cold) | base rollouts (reused) | old flash absolute | no | tech verbosity HACK (auto inflated 0.944, quality worse); pairwise base sweeps tech |
| committee_w1 | base (cold) | base rollouts (reused) | pure committee (no rubric) | no | tech WIN (pairwise; same length, auto de-inflated 0.833) / gov tie / **advisory regress**; auto OVERALL −0.074, hybrid −0.174 |
| committee_blend | base (cold) | base rollouts (reused) | **blend 0.5·auto + 0.5·committee + llm_rubric** | no | tech win / gov tie / **advisory still regress**; auto −0.074 (unchanged), hybrid −0.094; **advisory hybrid 0.617→0.800 (+0.183 vs w1)** |
| committee_w2 | **committee_blend (continue)** | **fresh on-policy** (committee_blend, temp=1.0) | blend + llm_rubric **+ base-ref (放法B)** | yes | **INVALID — see below (harness bug, advisory never trained + eval served wrong adapter)** |

## committee_w2 (on-policy) — INVALID round, root-caused 2026-06-14

**Verdict: cannot conclude anything about advisory/net-win from this round. Two harness bugs.**

**On-policy machinery WORKED**: rollout from committee_blend @ temp=1.0 produced genuinely DIVERSE
advisory (resp assistant-text 90K–123K chars). So temp diversity is fine.

**BUG 1 (rollout grading harness → advisory never trained):** advisory's 2 written rollouts (resp2
len=15214, resp3 len=13361) were recorded with `automated_score=0` + empty breakdown DESPITE valid
reports correctly written to `stakeholder_analysis.md`. Re-running the SAME grade() on the SAME grader
workspace NOW returns **0.78** (report_created/gov/commercial/... 7/9 = 1.0). So the rollout-time
automated=0 is a harness race/timing/path artifact (grade ran before the output file was in the
workspace, or threw + was swallowed) — NOT a real grading failure. Consequence: false auto=0 dragged
the blend to ~0.24, the 2 survivors scored 0.242/0.254 → variance 0.013 < 0.02 filter → **advisory
group DROPPED from training**. Plus 2/4 advisory rollouts were empty (no-write at temp=1.0). Net: the
on-policy step trained on 22 rows = gov/tech/commitment_gitlab/commitment_ntia/speaker_gitlab only;
**advisory NOT trained** → committee_w2 advisory == committee_blend (unchanged 13856).

**BUG 2 (eval harness):** the committee_w2 eval launch ssh dropped right after the adapter backup, so
`pkill;rm;nohup eval` never ran. The shim was still serving `committee_w1` and /tmp/eval_val3_lora held
STALE 06-13 transcripts. So there is NO valid committee_w2 eval (the numbers I started to pull were
committee_w1's stale data).

**Implication:** advisory "non-recovery" is NOT evidence that on-policy fails — advisory was never
actually trained (filtered out by the false auto=0) nor validly evaluated. Must fix BUG 1 (rollout
automated grading) + reduce no-write, then re-run; and fix the eval launch (verify shim adapter path +
fresh transcripts).

### committee_w2 RE-RUN (corrected) — 2026-06-14

**Root cause of BUG 1 (false auto=0) + no-write = the SAME thing: TIMEOUT.** All 4 advisory rollouts
had `timed_out=True`. Rollout timeout was `task.timeout_seconds(180) × timeout_multiplier(2.0,
hardcoded) = 360s`; the eval used ×3=540s and base advisory wrote 20K fine. 360s wasn't enough for the
71K-char advisory doc → 2/4 didn't finish writing (no-write) and 2/4 wrote but the timeout killed the
run before the output file synced to the grader workspace → grade() saw an empty workspace → auto=0.
(Proof: re-running the SAME grade() on the now-synced file returns 0.78.)

**Fixes applied:**
1. `generate_ledger_online_rollouts.py:168` timeout_multiplier 2.0 → `float(os.environ.get("ROLLOUT_TIMEOUT_MULT","4.0"))`; set ROLLOUT_TIMEOUT_MULT=4.0 (720s) in run_onpolicy.sh.
2. NEW `scripts/tf_agentic/rollout_healthcheck.py` — runs RIGHT AFTER rollout, BEFORE scoring/train;
   flags per-group ALL-TIMEOUT / MOSTLY-NO-WRITE / WRITTEN-BUT-AUTO=0 / and exits 2 on CRITICAL so the
   driver stops before wasting a train. (The "warn early" guard the user asked for.)

**Corrected rollout health (06-14): 8/9 groups OK, advisory FIXED.**
- advisory: timeout=0, auto=[0.00–0.94]; inject scores resp0/1/3 = 0.762/0.789/0.667, resp2 empty=0,
  **group spread 0.323** (vs the old filtered 0.026) → advisory WILL train this round.
- Only `ledger_speaker_nasa` still CRITICAL (NASA doc too long, all-timeout even at ×4) → dropped (it
  was a dead group in every prior round anyway). Health check correctly caught it and halted; we then
  finish on the 8 healthy groups (speaker_nasa filtered).
- Residual: ~1 empty (no-write) per group from temp=1.0 sampling — but now USEFUL negative contrast
  (write≈0.7 vs no-write=0), not a bug.

**committee_w2 (corrected) RESULT — 2026-06-15 (best round so far, not yet a clean net win):**

eval verified correct (shim env LORA_ADAPTER=onpolicy_w2; fresh transcripts). 8 groups trained
(advisory INCLUDED this time; only speaker_nasa dropped).

| task | pairwise (base vs w2) | automated | hybrid |
|---|---|---|---|
| advisory | base3/lora2/tie4 → TIE (p=1.0) | 1.000/0.778 | 0.900/0.833 |
| gov | lora6/0/tie3 → **LORA WIN** (p=0.031) | 0.889/0.889 | 0.903/0.715 |
| tech | base1/lora4/tie4 → tie (lora-lean) | 0.833/**0.889** | 0.733/0.733 |
| OVERALL | no sig loss; 1 win 2 tie | 0.907/**0.852 (−0.056)** | 0.845/0.761 |

**Advisory pairwise progression: committee_w1 base7/0 → committee_blend base7/0 → committee_w2 TIE.**
On-policy (with advisory ACTUALLY trained) pulled advisory from "significant loss" to "tie", recovered
the gov win, tech auto now beats base (0.889>0.833), best overall auto yet (−0.074→−0.056).

**Still not a net win:** advisory automated stuck at 0.778 — pure COVERAGE gap (w2 advisory len 11291 <
base 20K). Committee (quality) already calls advisory a tie; automated (coverage) still dings the
shorter report. Root limit: on-policy advisory rollouts capped at ~15K (committee_blend's generation) →
training can't exceed what it samples → can't reach base's 20K coverage.

**Next lever (committee_w3):** raise blend AUTO_W 0.5→0.7 (automated rewards coverage) to push advisory
completeness. Cheap test = re-inject existing on-policy rollouts at AUTO_W=0.7 + retrain (no new rollout).

**Lesson (process):** check `timed_out` + automated-on-written-reports IMMEDIATELY after rollout; a
written report with auto=0 is almost always a harness (timeout/sync/path) artifact, not a model failure.

## What each step isolates

- **round-1e → committee_w1**: change reward (flash absolute → relative committee). → relative committee FIXES the tech verbosity hack (auto de-inflated + pairwise quality flip).
- **committee_w1 → committee_blend**: add automated-blend + inject llm_rubric (same reused data, same cold-base init). → advisory hybrid quality +0.183, but automated COVERAGE not recovered (still reused all-complete base rollouts → no short-vs-complete contrast → advisory length stays truncated at 13856 due to cross-task "concise" bleed).
- **committee_blend → committee_w2**: fresh on-policy rollouts (temp=1.0 length diversity) + base-ref anchor + continue-train from committee_blend. The key jump designed to FIX advisory (finally has self-sampled diverse rollouts to contrast).

## ⚠️ Attribution caveat

committee_w2 changes THREE variables at once (data source / base-ref / init). If it wins, we
CANNOT attribute to a single factor. A clean ablation would vary one at a time
(e.g., on-policy-only vs on-policy+base-ref). Merged here for speed; split later if attribution matters.

## Through-line conclusions

1. automated-only reward is gameable by verbosity (round-1e).
2. relative committee fixes tech, but the single global "concise/grounded" gradient bleeds across
   tasks and shortens advisory (which needs comprehensiveness).
3. blend(+automated) + llm_rubric raises quality judgement but cannot fix advisory LENGTH/coverage
   when training on reused all-complete rollouts (no contrast).
4. advisory length/coverage can only be taught via on-policy diverse samples + a reference anchor
   (committee_w2 — under evaluation).

## Reward design decisions (settled)

- blend = `0.5·automated + 0.5·committee` (ADDITIVE, not multiplicative — multiplicative is too harsh,
  non-standard, couples component weights; rejected after user pushback).
- committee uses the task's hand-written `llm_rubric` when available, else a generic RULER fallback;
  on top of either, a thin anti-hallucination + anti-duplicate overlay (the "concise>verbose" clause
  lives ONLY in the generic fallback, so it never fights a completeness rubric like advisory's).
- base reference = 放法B: injected into the judge prompt as a CALIBRATION anchor (not scored, not in
  GRPO advantage normalization); the K policy rollouts are the only ranked/normalized set.
- on-policy: rollout from current policy + continue-train from it; logprobs computed WITH that adapter.
