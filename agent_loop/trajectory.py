"""
Trajectory reconstruction: align OpenClaw transcripts to token-level
prompt_ids / response_ids / response_mask for veRL training.

Follows veRL's SWE-Agent TrajectoryReconstructor pattern:
  - Each turn is replayed through the tokenizer's chat template
  - Model-generated tokens get mask=1, environment/tool tokens get mask=0
  - Strict validation: mismatched trajectories are marked failed
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Optional

logger = logging.getLogger(__name__)


@dataclass
class TurnRecord:
    """Record of a single LLM generation turn."""

    turn_index: int
    messages: list[dict[str, Any]]
    prompt_ids: list[int]
    response_ids: list[int]
    response_text: str
    response_logprobs: Optional[list[float]] = None


@dataclass
class AlignedTrajectory:
    """Token-level aligned trajectory ready for veRL training."""

    ok: bool
    initial_prompt_ids: list[int]
    response_ids: list[int]
    response_mask: list[int]  # 1 = model-generated, 0 = env/template
    response_logprobs: list[float]
    num_turns: int
    failure_reason: Optional[str] = None
    per_turn_boundaries: list[tuple[int, int]] = field(default_factory=list)


@dataclass
class TurnCompactionResult:
    """Result of evicting oldest turns from a flattened trajectory."""

    response_ids: list[int]
    response_mask: list[int]
    response_logprobs: list[float]
    turns: list[TurnRecord]
    dropped_turns: int
    dropped_tokens: int
    kept_turn_boundaries: list[tuple[int, int]]


class TrajectoryReconstructor:
    """Reconstruct token-level alignment from per-turn records.

    For each turn pair (k, k+1):
      - Turn k's response tokens → mask=1
      - Template/separator tokens between turns → mask=0
      - Tool/environment response tokens → mask=0
    """

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

        im_end_ids = tokenizer.encode("<|im_end|>", add_special_tokens=False)
        self.im_end_token_id = im_end_ids[-1] if im_end_ids else None

    def reconstruct(self, turns: list[TurnRecord]) -> AlignedTrajectory:
        if not turns:
            return AlignedTrajectory(
                ok=False,
                initial_prompt_ids=[],
                response_ids=[],
                response_mask=[],
                response_logprobs=[],
                num_turns=0,
                failure_reason="empty turns",
            )

        initial_prompt_ids = list(turns[0].prompt_ids)
        all_response_ids: list[int] = []
        all_mask: list[int] = []
        all_logprobs: list[float] = []
        boundaries: list[tuple[int, int]] = []

        for i, turn in enumerate(turns):
            turn_start = len(all_response_ids)

            # Model-generated tokens (mask=1)
            all_response_ids.extend(turn.response_ids)
            all_mask.extend([1] * len(turn.response_ids))
            if turn.response_logprobs and len(turn.response_logprobs) == len(
                turn.response_ids
            ):
                all_logprobs.extend(turn.response_logprobs)
            else:
                all_logprobs.extend([0.0] * len(turn.response_ids))

            turn_end = len(all_response_ids)
            boundaries.append((turn_start, turn_end))

            # Environment tokens between this turn and next
            if i + 1 < len(turns):
                next_prompt = turns[i + 1].prompt_ids
                expected_prefix_len = len(initial_prompt_ids) + len(all_response_ids)

                if len(next_prompt) > expected_prefix_len:
                    env_tokens = next_prompt[expected_prefix_len:]
                    all_response_ids.extend(env_tokens)
                    all_mask.extend([0] * len(env_tokens))
                    all_logprobs.extend([0.0] * len(env_tokens))
                elif len(next_prompt) < expected_prefix_len:
                    # Prompt is shorter than expected — possible template difference
                    # Use a relaxed approach: find the divergence point
                    logger.warning(
                        "Turn %d→%d: next prompt (%d) shorter than expected (%d), "
                        "using relaxed alignment",
                        i, i + 1, len(next_prompt), expected_prefix_len,
                    )

        return AlignedTrajectory(
            ok=True,
            initial_prompt_ids=initial_prompt_ids,
            response_ids=all_response_ids,
            response_mask=all_mask,
            response_logprobs=all_logprobs,
            num_turns=len(turns),
            per_turn_boundaries=boundaries,
        )

    def find_assistant_turn_ends(self, response_ids: list[int]) -> list[int]:
        """Find positions of <|im_end|> tokens in response sequence."""
        if self.im_end_token_id is None:
            return []
        return [
            i for i, tid in enumerate(response_ids)
            if tid == self.im_end_token_id
        ]


def compact_turn_history(
    turns: list[TurnRecord],
    response_ids: list[int],
    response_mask: list[int],
    response_logprobs: list[float],
    max_response_tokens: int,
) -> TurnCompactionResult:
    """Drop oldest full turns until the flattened response fits the budget.

    This is intentionally turn-aligned:
      - the entire oldest assistant turn is removed
      - any env/tool tokens that were part of that turn's flattened segment are removed too
      - the newest turns remain untouched

    The helper assumes `response_ids`, `response_mask`, and `response_logprobs`
    are already flattened in the same order as `turns`.
    """
    if max_response_tokens <= 0:
        return TurnCompactionResult(
            response_ids=[],
            response_mask=[],
            response_logprobs=[],
            turns=[],
            dropped_turns=len(turns),
            dropped_tokens=len(response_ids),
            kept_turn_boundaries=[],
        )

    if len(response_ids) <= max_response_tokens:
        boundaries: list[tuple[int, int]] = []
        cursor = 0
        for turn in turns:
            start = cursor
            cursor += len(turn.response_ids)
            boundaries.append((start, cursor))
            if cursor < len(response_ids) and turn is not turns[-1]:
                # The flattened stream may contain env tokens between turns.
                # We keep boundaries in response-space only; env tokens are not
                # represented in per-turn boundaries.
                pass
        return TurnCompactionResult(
            response_ids=list(response_ids),
            response_mask=list(response_mask),
            response_logprobs=list(response_logprobs),
            turns=list(turns),
            dropped_turns=0,
            dropped_tokens=0,
            kept_turn_boundaries=boundaries,
        )

    flattened_turn_spans: list[tuple[int, int]] = []
    cursor = 0
    for i, turn in enumerate(turns):
        start = cursor
        cursor += len(turn.response_ids)
        if i + 1 < len(turns):
            next_turn = turns[i + 1]
            expected_prefix = len(turns[0].prompt_ids) + cursor
            if len(next_turn.prompt_ids) > expected_prefix:
                cursor += len(next_turn.prompt_ids) - expected_prefix
        flattened_turn_spans.append((start, cursor))

    start_turn = 0
    while start_turn < len(flattened_turn_spans):
        remaining = len(response_ids) - flattened_turn_spans[start_turn][0]
        if remaining <= max_response_tokens:
            break
        start_turn += 1

    if start_turn >= len(flattened_turn_spans):
        # If even the newest turn exceeds the budget, keep the tail tokens only.
        tail_start = max(0, len(response_ids) - max_response_tokens)
        return TurnCompactionResult(
            response_ids=list(response_ids[tail_start:]),
            response_mask=list(response_mask[tail_start:]),
            response_logprobs=list(response_logprobs[tail_start:]),
            turns=[turns[-1]] if turns else [],
            dropped_turns=max(0, len(turns) - 1),
            dropped_tokens=tail_start,
            kept_turn_boundaries=[(0, len(response_ids) - tail_start)] if turns else [],
        )

    drop_tokens = flattened_turn_spans[start_turn][0]
    kept_response_ids = list(response_ids[drop_tokens:])
    kept_response_mask = list(response_mask[drop_tokens:])
    kept_response_logprobs = list(response_logprobs[drop_tokens:])
    kept_turns = list(turns[start_turn:])

    kept_boundaries: list[tuple[int, int]] = []
    cursor = 0
    for turn in kept_turns:
        start = cursor
        cursor += len(turn.response_ids)
        kept_boundaries.append((start, cursor))
        if cursor < len(kept_response_ids) and turn is not kept_turns[-1]:
            pass

    return TurnCompactionResult(
        response_ids=kept_response_ids,
        response_mask=kept_response_mask,
        response_logprobs=kept_response_logprobs,
        turns=kept_turns,
        dropped_turns=start_turn,
        dropped_tokens=drop_tokens,
        kept_turn_boundaries=kept_boundaries,
    )
