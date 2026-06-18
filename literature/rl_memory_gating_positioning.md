# Literature positioning — RL for adaptive memory gating (relevance to our gated-RL finding)

Saved 2026-06-18. Context: our breakthrough = on a mem0 *harness* (generic how-to hints injected
at inference), static FORCED injection ≈ RL ("memory subsumes RL"); but making memory OPTIONAL
(policy-gated: "you may apply/ignore — judge relevance") + cold-start-from-base lets RL net-beat
base+mem on tech (7:1). The lever is the GATING mechanism (learn WHEN to use memory), not raw gradient.

## Closest prior work
- **Memory-R1 (arXiv 2508.19828, 2025)** — RL to teach LLM agents to MANAGE + USE retrieved memories
  (add/update/delete/retain + how to use retrieved memories for reasoning) via outcome-based reward.
  KEY QUOTE matching our mechanism: "Retrieved memories are typically passed to the LLM without
  meaningful filtering, forcing the model to reason over both relevant and irrelevant content, whereas
  humans retrieve broadly but then filter SELECTIVELY." → exactly our policy-gated insight.
- **DeltaMem (arXiv 2604.01560)** — agentic memory management via RL (memory ops as atomic actions).
- **MemAgent / AtomMem** — RL to learn optimal memory policies, memory ops as atomic actions, long-term reward.
- **Retrieval-Augmented LLM Agents: Learning to Learn from Experience (2603.18272)**; **Beyond Experience
  Retrieval: utility-optimized structured experience for FROZEN LLMs (2602.02556)** — the training-free /
  frozen-LLM experience-injection side (≈ our base+mem0 baseline).
- **SSGM (2603.11768)** — governing evolving memory in LLM agents (risks/stability) — relevant to our red line.

## Our differentiation / contribution vs this trend
1. We isolate **forced vs policy-gated memory** as the lever: forced injection (more rollout variance, 0.133)
   ties the static-memory baseline everywhere; gated (model decides when to use the hint) is what beats it.
   → "gating mechanism > gradient magnitude." Most memory-RL work jumps to gating without the controlled
   forced-vs-gated ablation against a strong training-free baseline.
2. **Automated-blindness**: the gated tech win is INVISIBLE to the rule/hybrid grader (78.7 vs 80.0) — the
   committee sees it, automated doesn't. Ties our committee-reward + automated-blindness thread.
3. **mem0-as-harness baseline is strong**: training-free hint injection ≈ RL on most dims; RL only carves
   space where the correct behavior is instance-specific (not encodable in a generic hint). Frames WHEN
   RL is worth it over cheap experiential memory.
4. **Saturation insight (H-A)**: continue-training from a converged adapter kills the gradient (var 0.13→0.06);
   cold-start-from-base is required. A practical recipe note under-discussed in the memory-RL papers.

## TODO for paper
- Pull full Memory-R1 + DeltaMem methods; compare their reward/gating formulation to our committee + optional-hint framing.
- Note our held-out-judge gap (same 3 judges train+eval) as a limitation vs their outcome-based rewards.
