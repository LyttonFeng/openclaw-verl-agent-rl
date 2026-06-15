# Research Log — Committee-Reward Agentic RL

## 2026-06-15 — Autoresearch framing + temp=0.3 table
- Adopted autoresearch orchestration over the ongoing committee-reward RL work. Set up 20-min
  heartbeat (cron 7eb32d96). Scaffolded research-state.yaml / findings.md / this log.
- **Decision (eval protocol):** lock a canonical base@0.3 anchor; judge every lora vs it with a
  9-pair (3×3) heterogeneous committee + order-consistency + deliberation + sign-test. Reason:
  temp=0 differs across invocations → base not unique → confound.
- **base@0.3 DONE:** MEETING 79.5%; advisory 0.93 / gov 0.738 / tech 0.716. 3 advisory runs
  md5-distinct (12938/11952/12775) → temp=0.3 sampling confirmed real.
- **w6@0.3 (AUTO_W=0 + deliberation) DONE & judged vs base@0.3:** gov WIN (7:1, p=0.070);
  advisory lora-favored (6:2, p=0.289); tech base-lean (6:3, n.s.). Deliberation looks positive,
  esp. recovering advisory.
- Updated training base anchor → base_ref_temp03.jsonl (pod /workspace/saved_adapters/).
- **Incident:** external `pkill benchmark.py` killed a live base eval (rc=137); /tmp eval script
  was also missing the temp patch (ran greedy). Both fixed; logged in findings Lessons.
- **In progress:** w5@0.3 eval running (poller bl9uctv2i). Then w4/w3/w2 → full table → synthesis.

## 2026-06-15 (later) — Full @0.3 ablation complete + outer-loop synthesis
- All variants judged vs canonical base@0.3 (9-pair committee). Verdicts: verdict_w{2,3,4,5,6}.log.
- **Result:** committee-reward beats base on gov significantly (4/5 loras), INVISIBLE to automated
  (gov auto/hyb flat ~.85/.74). w2 paradox: highest automated (81.3%) = weakest committee.
- **H1 (AUTO_W) overturned** — earlier temp=0 "w3 worst" was a base-nondeterminism confound;
  vs anchored base, gov WIN robust across AUTO_W 0.0-0.7. AUTO_W is not the lever.
- **H2 (deliberation) confirmed** — w6 vs w5: preserves advisory + tech accuracy. KEEP. Best = w6.
- **tech diagnosed:** not a ceiling (~28% headroom); RL trades accuracy for coverage except under w6.
- Empty-report root cause = policy read-loop→timeout, NOT harness.
- Synthesis → findings.md updated; to_human/committee_reward_ablation_20260615.html built + opened.
- Open: clean-rerun w5 advisory; continue on-policy from w6 to push advisory to significance.

## 2026-06-16 — Deliberation OFF; tech-hard probe = informative negative
- Decision churn resolved: deliberation DEFAULT OFF (only evidence is "no quality effect";
  re-enable only if a reward-variance test shows it helps). All artifacts + GitHub naive_ppo
  branch corrected.
- Consolidated next-round recipe + 3 guardrails to GitHub (run_next_round.sh, next_round_recipe.md).
- **tech-hard probe (negative, informative):** short synthetic transcript with implicit owners /
  retraction / revised deadline / distractor names → base@0 got 6/6 RIGHT + excluded the retracted
  item. Reasoning tricks on short input do NOT create a base gap. base's real failures are
  long-doc/agentic-retrieval. STRATEGIC FORK (awaiting user): (1) build long-doc variant to hit the
  real failure region (confounds with read-loop), or (2) reconsider whether "harder tasks" is the
  right direction given base reasons well (the advisory/tech tie may be a real shared ceiling).
