# Findings — Committee-Reward Agentic RL (qwen3.5-4b, meeting analysis)

## Research question
Can a stable heterogeneous LLM **committee** reward (RULER-style relative scoring) train
qwen3.5-4b to beat its own base on agentic meeting-analysis (Val3: advisory / gov / tech),
when the automated rule-based grader is a weak, gameable proxy? What AUTO_W is right, and
does judge **deliberation** help?

## Current understanding
- **The automated grader is gameable.** Round-1e (old absolute flash reward) inflated tech's
  automated score to 0.944 via verbosity/hallucination while a pairwise committee judged base
  the clear winner. Automated measures presence/coverage (length-friendly, blind to padding);
  the committee measures grounding/quality (penalizes padding). They **agree at the floor
  (did it write anything) but conflict at the top (max coverage ≠ max quality)**.
- **Relative committee > absolute scoring.** Switching reward from absolute-flash to a
  relative committee fixed tech's reward-hack (automated de-inflated to 0.833, pairwise quality
  flipped). This is the core methodological win.
- **advisory completeness needs on-policy diverse samples + a reference anchor.** A global
  "concise" signal bled across tasks and shortened advisory under naive committee scoring;
  blend+rubric raised quality but didn't fix length; on-policy (temp=1.0) rollouts + a base
  reference anchor (放法B, calibration-only) is the lever that teaches advisory length/coverage.

## COMPLETE temp=0.3 ablation (all vs canonical base@0.3, 9-pair committee + auto/hyb)

| var | AUTO_W | delib | advisory committee [auto/hyb] | gov committee [auto/hyb] | tech committee [auto/hyb] | MEET% |
|---|---|---|---|---|---|---|
| base | — | — | anchor [.96/.93] | anchor [.85/.74] | anchor [.80/.72] | 79.5 |
| w5(flake) | 0.0 | no | tie~base 5:3 [.67/.60]* | WIN 8:1 p=.039 [.85/.74] | tie~base 5:2 [.83/.70] | 67.9 |
| **w5-clean** | 0.0 | no | tie 5:4 [.92/.84] | tie 6:1 p=.125 [.85/.85] | tie 4:3 [.83/.71] | 80.1 |
| w6 | 0.0 | yes | tie~lora 6:2 [.96/.90] | WIN 7:1 p=.070 [.89/.74] | tie~base 6:3 [.87/.75] | 79.7 |
| w4 | 0.2 | no | tie~base 5:3 [.94/.94] | WIN 7:1 p=.070 [.83/.70] | tie~base 5:4 [.87/.73] | 79.3 |
| w2 | 0.5 | no | tie~lora 5:2 [1.0/.87] | tie 4:4 [.93/.86] | tie~base 5:3 [.83/.71] | **81.3** |
| w3 | 0.7 | no | tie~lora 6:2 [.81/.79] | **WIN 9:0 p=.004** [.87/.73] | tie 4:4 [.83/.72] | 74.7 |

*w5(flake) advisory had 1 policy-failure run (read-loop→timeout→empty). w5-clean = clean rerun
(no flake); USE w5-clean for all deliberation comparisons.

## Patterns and insights (full @0.3 ablation, CORRECTED after w5 clean rerun)
- **CORE RESULT (robust) — committee-reward beats base on gov, INVISIBLY to automated.** Loras win
  (or favor) gov on committee while gov auto/hyb stays FLAT (~.85/.74 = base). The automated grader
  cannot see the gov quality gain the committee detects. This is the central proof-of-thesis.
  CAVEAT: only LARGE margins are trustworthy — **the same w5 adapter judged twice gave gov 8:1
  (p=.039, sig) then 6:1 (p=.125, n.s.)**. Verdicts wobble across reruns at 6:1–8:1; only w3 9:0
  (p=.004) is solid. Trust margin, not the p<.10 boundary.
- **w2 paradox (robust).** w2 has the HIGHEST automated (MEETING 81.3%, only variant beating base)
  but is the WEAKEST on committee (gov tie 4:4). w3 has near-lowest automated (74.7%) but strongest
  committee (gov 9:0). Optimizing automated produces reports that look best to rules, not to the
  committee. Chasing automated is counterproductive for real quality.
- **AUTO_W (H1) NOT the lever.** gov favored across AUTO_W 0.0→0.7. (Earlier temp=0 "w3 worst" was a
  base-nondeterminism confound — validates the canonical-base protocol.)
- **Deliberation (H2) — RETRACTED. No robust effect.** Earlier "w6 > w5" was driven by the w5
  FLAKE. After the clean rerun: w5-clean vs w6 are statistically indistinguishable — advisory
  5:4 vs 6:2 (both ties), tech auto IDENTICAL (owners 0.67, attribution 0.50 both; the "w5
  attribution crashed to 0.17" was the flake run), MEETING 80.1 vs 79.7. So NOT a proven quality
  lever. DECISION: deliberation OFF by default — the only evidence is "no quality effect", so don't
  carry an unproven knob. Its effect on training reward-VARIANCE was never measured; re-enable
  (DELIBERATE=1) only if a cheap variance test (re-score same rollouts w/ and w/o delib) shows it helps.
- **advisory & tech = genuine ties.** A human practitioner confirms they can spot OBVIOUS
  differences but cannot adjudicate these — when both human and committee can't tell two reports
  apart, tie IS the answer, not a judge failure. The marginal 6:2 "lora-favored" reads are noise in
  a region of genuine indistinguishability; do not chase them.

## Best model & answer (corrected)
- **No single config is robustly "best."** w5-clean / w4 / w6 are all ≈ equivalent: gov favored,
  advisory/tech tie. w3 has the single most solid result (gov 9:0) but lowest automated. For the
  next round, continue from any committee-trained adapter (w6 by default — cheap, deliberation
  harmless); the choice is not load-bearing.
- **Research question — answered YES.** Committee-reward RL trains qwen3.5-4b to beat its base on
  agentic meeting analysis (significant gov win, advisory non-worse, tech tie), and the gain is
  invisible to (even inverted by) the automated grader — validating the committee as the
  meaningful objective. The decisive lever is judge deliberation, not the AUTO_W blend weight.

## Evaluation methodology (hard-won)
- **temp=0 greedy is deterministic WITHIN one eval invocation** (RUNS=3 → md5-identical) but
  **differs ACROSS invocations** (agentic multi-turn: session ids/timestamps/tool timing/fp).
  → base is not unique across runs; this is a real confound. **Fix: anchor ONE canonical base
  and judge every lora against it.**
- **temp=0.3 is for eval validity, not quality.** It yields 3 genuinely different reports per
  model (md5-distinct) so the 9-pair committee is real and we see the quality *distribution*,
  not just the mode. Best-of-3 can exceed greedy; mean ≈ greedy.
- **9-pair committee** (3×3) with order-consistency + heterogeneous panel + deliberation +
  sign-test is the trustworthy verdict. Null calibration (base-vs-base ≈ all ties) validates no bias.

## Lessons and constraints
- **NEVER pkill a running eval from outside.** Self-inflicted: my external `pkill benchmark.py`
  killed a live base eval (rc=137) mid-run. The eval script self-cleans at start; launch once, then hands off.
- **/tmp eval script patches don't persist / can desync from the repo copy.** The temp-propagation
  patch (SHIM_DEFAULT_TEMP/PINCHBENCH_MODEL_TEMPERATURE = ${EVAL_TEMP}) was missing from /tmp and
  the eval ran greedy. Always re-verify the live /tmp script before a run.
- **RunPod ssh is unstable.** Never hold long ssh / never drive a pipeline over ssh. Launch
  pod-self-contained scripts (exec >log 2>&1), monitor ARTIFACT files via short independent ssh.
- **vLLM 0.22 online LoRA is a no-op on qwen3.5** → both train & infer use transformers+PEFT shim.

## Probe (2026-06-16): "reasoning-hard" ≠ base-failure — base is more capable than the eval implied
Built a SHORT synthetic tech transcript with deliberately hard features (implicit owners via
"I'll take that"/"our designer", a RETRACTED item "scratch dark mode", a REVISED deadline
Friday→Wednesday, distractor near-names Jon/John + Sara/Sarah). Ran base@temp0 single-turn.
**base got 6/6 action items + owners + revised deadline RIGHT and correctly EXCLUDED the
retracted item.** (My auto-grader reported 1/6 — that was a multi-line-format bug, not base failing.)
- **Implication:** base's reasoning on clean short input is strong. The real-eval failures
  (tech owners_identified 0.17, the read-loop→timeout empty report) come from the LONG-DOC +
  AGENTIC regime (retrieval/coverage collapse over 71K chars), NOT from reasoning difficulty.
- **So "make tasks harder" via reasoning tricks on short transcripts does NOT create a base gap.**
  The only demonstrated base-failure lever is length/agentic-retrieval — which confounds with the
  read-loop/harness-length problem (the messy region).
- **Also reframes the advisory/tech TIE:** base reasons well, so the tie may be a genuine
  quality ceiling reached by both, not base weakness. Whether "harder tasks" is even the right
  direction is now an open strategic question (see below).

## Open questions
- Which AUTO_W is best @0.3 once w5/w4/w3/w2 are judged? (expect low / committee-dominant)
- Does deliberation (w6) beat its no-deliberation twin (w5, same AUTO_W=0) net across tasks?
- Can we net-beat base on advisory at significance (currently 6:2 favored but p=0.289)?
- Does temp=0.3 sampling change base *quality* vs greedy? (head-to-head not yet run)
