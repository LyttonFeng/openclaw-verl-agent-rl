# H-E — Finding RL's space: instance-specific reasoning that a generic mem0 hint can't carry

**Locked: 2026-06-17 (before running). User directive: "寻找 RL 的空间哈，我相信 mem 提带不了它."**

## Claim under test
On a mem0 *harness* (hints at both train and eval), RL beats the base+mem0 baseline ONLY in a
regime where the correct behavior is **instance-specific** (cannot be expressed as a one-line generic
how-to hint) AND rollouts still **complete** (so within-group reward variance / GRPO gradient exists).
Generic mem0 hints help on average but cannot resolve instance-specific decisions; RL with a
correctness reward can. Predict: **RL-from-base+mem0 > base+mem0 head-to-head on H-E tasks.**

## Why the current Val3 tasks don't show it
- advisory: training instance uncompleteable (1/8) + variance 0.055 → no gradient. Dead for RL.
- gov: both RL and mem0 win → redundant, not RL's exclusive space.
- tech: highest variance (0.160) and the only unique RL+mem0 win (9:0 vs base), BUT head-to-head
  w7+mem0 vs base+mem0 is a tie → RL hasn't escaped on the *existing* tech instance.
- Root cause: the existing tasks are either (i) too long → context wall (no completion, no gradient),
  or (ii) answerable by a generic hint (mem0 already caps them). Neither leaves room for RL.

## The design sweet spot (the whole trick)
Build transcripts that are **completable-length (~3–6K chars, NOT 71K)** but **dense with
instance-specific traps** a generic hint cannot resolve:
- **Implicit owners**: "I'll take that" / "our designer will handle it" → owner must be inferred from
  the specific dialogue, not from a hint that says "state the owner."
- **Retracted item**: "actually, scratch the dark-mode task" → must be EXCLUDED; a generic
  "list all action items" hint pushes the wrong way.
- **Revised deadline**: Friday → later corrected to Wednesday → must reconcile to the latest.
- **Multi-turn synthesis**: one action item split across 3 non-adjacent turns → must be merged.
- **Distractor near-names**: Jon/John, Sara/Sarah → correct attribution is instance-specific.
Difficulty must be tuned so **base+mem0 makes real errors** (coverage/attribution/retraction) while
the doc still completes. (The earlier tech-hard probe was SHORT → base aced it 6/6 → no gap. The
lever here is density+length-just-below-the-wall, not reasoning trickiness alone.)

## Procedure
1. Author 6–10 such tasks (3 tech-style action-items + 3 gov/advisory-style), each with a
   deterministic answer key (owners, included/excluded items, reconciled deadlines).
2. **Sanity gates BEFORE training:** (a) base+mem0 @0.3 must score clearly < 1.0 (real errors to fix);
   (b) rollouts must COMPLETE (≥3/4 non-empty per group) so a GRPO gradient exists;
   (c) within-group committee std must be meaningfully > 0.06 (the flat-gradient threshold).
   If any gate fails, re-tune difficulty/length. (No gate → no valid test.)
3. Train RL from **base+mem0** (hints in rollouts) with committee reward, K=4–8, batched-2.
4. Eval @0.3 WITH hints; committee head-to-head vs **base+mem0** on these tasks.
5. Held-out judge check (exclude one training judge) to guard committee-overfit.

## Predictions (falsifiable)
- **Support:** RL-from-base+mem0 > base+mem0 on H-E (sign-test sig on ≥1 dim). → RL has a real,
  non-redundant space: instance-specific correctness that generic memory cannot encode.
- **Refute:** tie/loss everywhere → even with instance-specific difficulty, on-policy committee-RL
  cannot exceed the mem0 harness here → strong "memory subsumes RL" boundary claim.

## Dependencies
- Gated on H-A (#31, base+mem0 rollout variance): if base+mem0 has higher variance than w7+mem0,
  start RL from base+mem0 (more gradient). If not, H-E's instance-specific design must itself
  manufacture the variance (its whole point) — proceed regardless, but expect to rely on (2c).
