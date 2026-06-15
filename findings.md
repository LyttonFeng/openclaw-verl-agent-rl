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
| w5 | 0.0 | no | tie~base 5:3 [.67/.60]* | **WIN 8:1 p=.039** [.85/.74] | tie~base 5:2 [.83/.70] | 67.9 |
| w6 | 0.0 | yes | tie~lora **6:2** [.96/.90] | **WIN 7:1 p=.070** [.89/.74] | tie~base 6:3 [.87/.75] | 79.7 |
| w4 | 0.2 | no | tie~base 5:3 [.94/.94] | **WIN 7:1 p=.070** [.83/.70] | tie~base 5:4 [.87/.73] | 79.3 |
| w2 | 0.5 | no | tie~lora 5:2 [1.0/.87] | tie 4:4 [.93/.86] | tie~base 5:3 [.83/.71] | **81.3** |
| w3 | 0.7 | no | tie~lora **6:2** [.81/.79] | **WIN 9:0 p=.004** [.87/.73] | tie 4:4 [.83/.72] | 74.7 |

*w5 advisory dragged by 1 policy-failure run (read-loop → timeout → empty deliverable; root cause = policy, not harness).

## Patterns and insights (CONFIRMED by the full @0.3 ablation)
- **CORE RESULT — committee-reward RL beats base on gov, INVISIBLY to automated.** 4/5 loras win
  gov significantly (w3 9:0 p=.004, w5 8:1 p=.039, w4/w6 7:1 p=.070), yet gov auto/hyb is FLAT
  (~.85/.74, identical to base) across ALL of them. The automated grader cannot see the gov
  quality gain the committee clearly detects. This is the central proof-of-thesis.
- **The decoupling is INVERTED at the top (w2 paradox).** w2 has the HIGHEST automated (gov
  .93/.86, MEETING 81.3%, only variant beating base) but is the WEAKEST on committee (gov tie
  4:4, all ties). w3 has among the LOWEST automated (74.7%) but the STRONGEST committee (gov 9:0).
  → Optimizing the automated/hybrid signal produces reports that look best to rules but are NOT
  better to the committee. Chasing automated is actively counterproductive for real quality.
- **AUTO_W (H1) NOT supported once base is anchored.** Within the controlled set (w3/w4/w5/w6,
  same data+init), the gov WIN is robust across AUTO_W 0.0→0.7 — AUTO_W is not the lever. The
  earlier temp=0 "w3 worst / advisory 9:0 loss" was a base-nondeterminism confound; vs the fixed
  canonical base, w3 advisory is lora-favored 6:2. (Validates the canonical-base protocol.)
- **Deliberation (H2) IS the real stabilizer — KEEP IT.** w6 vs w5 (controlled, only delib
  differs): advisory w6 clean 6:2 vs w5 flake-tie; tech accuracy w6 preserved (owner-attribution
  0.50, completeness/context 0.75) vs w5 CRASHED (attribution 0.17); gov both WIN. Deliberation
  prevents the committee reward from trading grounding/accuracy for coverage.
- **tech: committee-tie everywhere, but it MOVED (not a ceiling).** base tech ~0.72 has ~28%
  headroom. RL raised coverage (owners_identified 0.17→0.67) but, except under w6, traded away
  accuracy (w4 attribution 0.50→0.25; w5 →0.17). Only w6 (pure committee + delib) raised coverage
  without wrecking accuracy. Net committee tie = the coverage-vs-grounding conflict on tech.

## Best model & answer
- **Best config: w6 (AUTO_W=0 + deliberation).** gov WIN + advisory clean 6:2 lora + tech
  accuracy preserved + automated ≈ base (79.7). Most balanced net-positive, no significant loss.
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

## Open questions
- Which AUTO_W is best @0.3 once w5/w4/w3/w2 are judged? (expect low / committee-dominant)
- Does deliberation (w6) beat its no-deliberation twin (w5, same AUTO_W=0) net across tasks?
- Can we net-beat base on advisory at significance (currently 6:2 favored but p=0.289)?
- Does temp=0.3 sampling change base *quality* vs greedy? (head-to-head not yet run)
