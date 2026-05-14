"""Unit tests for JiuwenClawAgentLoop response_mask construction.

No vLLM, no veRL runtime — just verifies the pure-Python re-tokenization
of jiuwenclaw history.json against a real Qwen3 tokenizer.

Run from repo root:
    python3 -m pytest experiments/verl_port_poc/tests/ -v
or directly:
    python3 experiments/verl_port_poc/tests/test_jiuwenclaw_agent_loop.py
"""
from __future__ import annotations

import json
import os
import sys
import unittest
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jiuwenclaw_agent_loop import JiuwenClawAgentLoop  # noqa: E402


def _apply_chat_template_ids(tok, msgs, add_gen):
    """Normalize apply_chat_template(tokenize=True) → list[int] across transformers versions."""
    out = tok.apply_chat_template(msgs, add_generation_prompt=add_gen, tokenize=True)
    if isinstance(out, dict) or hasattr(out, "input_ids"):
        out = out["input_ids"]
    return list(out)

FIXTURE_DIR = Path(__file__).parent / "fixtures"
SENTIMENT_HISTORY = FIXTURE_DIR / "jiuwen_history_sentiment.json"


def _load_tokenizer():
    try:
        from transformers import AutoTokenizer
    except ImportError:
        return None
    try:
        return AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    except Exception:
        return None


TOKENIZER = _load_tokenizer()


def _make_loop():
    """Bypass AgentLoopBase.__init__ (which wants veRL runtime); inject tokenizer."""
    loop = JiuwenClawAgentLoop.__new__(JiuwenClawAgentLoop)
    loop.tokenizer = TOKENIZER
    return loop


class TestHistoryToMessages(unittest.TestCase):
    def test_synthetic_simple_turn(self):
        history = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "event_type": "chat.delta", "content": "hi there"},
        ]
        msgs = JiuwenClawAgentLoop._history_to_messages(history)
        self.assertEqual(msgs, [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ])

    def test_synthetic_tool_call(self):
        history = [
            {"role": "user", "content": "read file foo.md"},
            {"role": "assistant", "event_type": "chat.delta", "content": "reading"},
            {"role": "assistant", "event_type": "chat.tool_call",
             "tool_call": {"name": "read_file",
                           "arguments": '{"path": "foo.md"}',
                           "tool_call_id": "call_1"}},
            {"role": "assistant", "event_type": "chat.tool_result",
             "result": "file content here", "tool_call_id": "call_1"},
            {"role": "assistant", "event_type": "chat.delta", "content": "done"},
        ]
        msgs = JiuwenClawAgentLoop._history_to_messages(history)
        self.assertEqual(len(msgs), 4)
        self.assertEqual(msgs[0]["role"], "user")
        self.assertEqual(msgs[1]["role"], "assistant")
        self.assertEqual(msgs[1]["content"], "reading")
        self.assertEqual(len(msgs[1]["tool_calls"]), 1)
        self.assertEqual(msgs[1]["tool_calls"][0]["function"]["name"], "read_file")
        self.assertEqual(msgs[1]["tool_calls"][0]["id"], "call_1")
        self.assertEqual(msgs[2]["role"], "tool")
        self.assertEqual(msgs[2]["tool_call_id"], "call_1")
        self.assertEqual(msgs[3], {"role": "assistant", "content": "done"})

    def test_delta_overwrite_keeps_last(self):
        history = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "event_type": "chat.delta", "content": "partial"},
            {"role": "assistant", "event_type": "chat.delta", "content": "partial more"},
            {"role": "assistant", "event_type": "chat.final", "content": "FINAL"},
        ]
        msgs = JiuwenClawAgentLoop._history_to_messages(history)
        self.assertEqual(msgs[-1]["content"], "FINAL")

    def test_skip_usage_metadata(self):
        history = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "event_type": "chat.usage_metadata", "content": ""},
            {"role": "assistant", "event_type": "chat.tool_update", "content": ""},
            {"role": "assistant", "event_type": "chat.delta", "content": "answer"},
        ]
        msgs = JiuwenClawAgentLoop._history_to_messages(history)
        self.assertEqual(len(msgs), 2)
        self.assertEqual(msgs[1]["content"], "answer")

    def test_dict_args_serialized(self):
        history = [
            {"role": "user", "content": "q"},
            {"role": "assistant", "event_type": "chat.tool_call",
             "tool_call": {"name": "f", "arguments": {"k": "v"}, "tool_call_id": "c1"}},
            {"role": "assistant", "event_type": "chat.tool_result",
             "result": "ok", "tool_call_id": "c1"},
        ]
        msgs = JiuwenClawAgentLoop._history_to_messages(history)
        args = msgs[1]["tool_calls"][0]["function"]["arguments"]
        self.assertIsInstance(args, str)
        self.assertEqual(json.loads(args), {"k": "v"})

    def test_real_fixture_shape(self):
        if not SENTIMENT_HISTORY.exists():
            self.skipTest(f"fixture missing: {SENTIMENT_HISTORY}")
        history = json.loads(SENTIMENT_HISTORY.read_text())
        msgs = JiuwenClawAgentLoop._history_to_messages(history)
        # 1 user + N (assistant, tool) pairs
        self.assertGreaterEqual(len(msgs), 3)
        self.assertEqual(msgs[0]["role"], "user")
        roles = [m["role"] for m in msgs]
        # No two consecutive tool messages (each tool follows an assistant)
        for a, b in zip(roles, roles[1:]):
            if b == "tool":
                self.assertEqual(a, "assistant",
                                 f"tool message must follow assistant, got {a!r}")
        # Last must be assistant (final answer flushed at end-of-stream)
        self.assertEqual(roles[-1], "assistant")
        # Tool calls reference valid IDs
        for i, m in enumerate(msgs):
            if m["role"] == "tool":
                self.assertTrue(m.get("tool_call_id"))


@unittest.skipIf(TOKENIZER is None, "Qwen3-4B tokenizer not available")
class TestBuildResponseFromHistory(unittest.TestCase):
    def test_mask_shape_matches_response(self):
        if not SENTIMENT_HISTORY.exists():
            self.skipTest(f"fixture missing: {SENTIMENT_HISTORY}")
        history = json.loads(SENTIMENT_HISTORY.read_text())
        loop = _make_loop()
        msgs = loop._history_to_messages(history)
        prompt_ids = _apply_chat_template_ids(TOKENIZER, msgs[:1], add_gen=True)
        resp_ids, resp_mask, num_turns = loop._build_response_from_history(history, prompt_ids)
        self.assertEqual(len(resp_ids), len(resp_mask))
        self.assertGreater(len(resp_ids), 0)
        # At least one assistant turn → at least one mask=1 token
        self.assertGreaterEqual(num_turns, 1)
        self.assertIn(1, resp_mask)
        # Tool results present → at least one mask=0 token
        self.assertIn(0, resp_mask)

    def test_concat_reconstructs_full_transcript(self):
        """prompt_ids + response_ids should equal apply_chat_template(all_msgs)."""
        history = [
            {"role": "user", "content": "what's 2+2?"},
            {"role": "assistant", "event_type": "chat.delta",
             "content": "Let me think. <answer>4</answer>"},
        ]
        loop = _make_loop()
        msgs = loop._history_to_messages(history)
        prompt_ids = _apply_chat_template_ids(TOKENIZER, msgs[:1], add_gen=True)
        resp_ids, resp_mask, num_turns = loop._build_response_from_history(history, prompt_ids)
        full = _apply_chat_template_ids(TOKENIZER, msgs, add_gen=False)
        self.assertEqual(prompt_ids + resp_ids, full)
        self.assertEqual(num_turns, 1)
        # Entire response is one assistant turn → all mask=1
        self.assertTrue(all(m == 1 for m in resp_mask))

    def test_mask_segregates_assistant_from_tool(self):
        """Assistant token regions get mask=1, tool result regions get mask=0."""
        history = [
            {"role": "user", "content": "read foo.md"},
            {"role": "assistant", "event_type": "chat.delta", "content": "ok"},
            {"role": "assistant", "event_type": "chat.tool_call",
             "tool_call": {"name": "read_file",
                           "arguments": '{"path": "foo.md"}',
                           "tool_call_id": "c1"}},
            {"role": "assistant", "event_type": "chat.tool_result",
             "result": "FILE_CONTENT_TOKEN_XYZ", "tool_call_id": "c1"},
            {"role": "assistant", "event_type": "chat.delta", "content": "the file said xyz"},
        ]
        loop = _make_loop()
        msgs = loop._history_to_messages(history)
        prompt_ids = _apply_chat_template_ids(TOKENIZER, msgs[:1], add_gen=True)
        resp_ids, resp_mask, num_turns = loop._build_response_from_history(history, prompt_ids)
        self.assertEqual(num_turns, 2)
        # The synthetic tool result content "FILE_CONTENT_TOKEN_XYZ" should
        # tokenize to tokens whose mask is 0.
        marker_ids = TOKENIZER.encode("FILE_CONTENT_TOKEN_XYZ", add_special_tokens=False)
        # find marker span in resp_ids
        found = False
        for s in range(len(resp_ids) - len(marker_ids) + 1):
            if resp_ids[s:s + len(marker_ids)] == marker_ids:
                found = True
                self.assertTrue(all(m == 0 for m in resp_mask[s:s + len(marker_ids)]),
                                "tool result tokens must have mask=0")
                break
        self.assertTrue(found, "tool result marker tokens not found in response")
        # The final assistant text "the file said xyz" should have mask=1.
        asst_marker = TOKENIZER.encode("the file said xyz", add_special_tokens=False)
        # find span
        for s in range(len(resp_ids) - len(asst_marker) + 1):
            if resp_ids[s:s + len(asst_marker)] == asst_marker:
                self.assertTrue(all(m == 1 for m in resp_mask[s:s + len(asst_marker)]),
                                "final assistant tokens must have mask=1")
                break

    def test_no_history_returns_empty(self):
        loop = _make_loop()
        resp_ids, resp_mask, num_turns = loop._build_response_from_history([], [1, 2, 3])
        self.assertEqual(resp_ids, [])
        self.assertEqual(resp_mask, [])
        self.assertEqual(num_turns, 0)


class TestRailV1Builder(unittest.TestCase):
    """Verify rail-v1 sample loader + builder (the new Path A that replaces
    history.json reverse-engineering when USE_RL_ONLINE_RAIL=1)."""

    def setUp(self):
        # Lazy import so this test file still runs without veRL
        from jiuwenclaw_agent_loop import (  # noqa: E402
            _load_rail_v1_samples,
            _build_response_from_rail_samples,
        )
        self._load = _load_rail_v1_samples
        self._build = _build_response_from_rail_samples

    def _write_samples(self, rail_dir, samples_by_file):
        for fname, samples in samples_by_file.items():
            (rail_dir / fname).write_text(
                "\n".join(json.dumps(s) for s in samples), encoding="utf-8"
            )

    def test_two_turn_with_tool_gap(self):
        """Tool-result tokens between turns → mask=0, logprobs=0.0 for the gap."""
        samples = [
            {"session_id": "S", "step_index": 0,
             "prompt_ids": [1, 2, 3, 4, 5],
             "response_tokens": [100, 101, 102],
             "logprobs": [-0.1, -0.2, -0.3]},
            {"session_id": "S", "step_index": 1,
             # prompt[i+1] = (prompt[i] + response[i]) + [200, 201] tool result
             "prompt_ids": [1, 2, 3, 4, 5, 100, 101, 102, 200, 201],
             "response_tokens": [110, 111],
             "logprobs": [-0.4, -0.5]},
        ]
        prompt, resp, mask, lp, n_turns = self._build(samples, eos_token_id=0)
        self.assertEqual(prompt, [1, 2, 3, 4, 5])
        self.assertEqual(resp, [100, 101, 102, 200, 201, 110, 111])
        self.assertEqual(mask, [1, 1, 1, 0, 0, 1, 1])
        self.assertEqual(lp, [-0.1, -0.2, -0.3, 0.0, 0.0, -0.4, -0.5])
        self.assertEqual(n_turns, 2)

    def test_empty_samples_returns_eos_placeholder(self):
        """No samples → EOS-as-placeholder so veRL postprocess sees a tensor."""
        prompt, resp, mask, lp, n_turns = self._build([], eos_token_id=7)
        self.assertEqual(prompt, [7])
        self.assertEqual(resp, [7])
        self.assertEqual(mask, [0])
        self.assertEqual(lp, [0.0])
        self.assertEqual(n_turns, 0)

    def test_missing_logprobs_padded_to_response_length(self):
        """Rail may omit logprobs for some turns — pad with 0.0 to keep alignment."""
        samples = [
            {"session_id": "X", "step_index": 0,
             "prompt_ids": [], "response_tokens": [9], "logprobs": [-0.9]},
            {"session_id": "X", "step_index": 1,
             "prompt_ids": [9], "response_tokens": [10], "logprobs": []},
        ]
        _, resp, _, lp, _ = self._build(samples, 0)
        self.assertEqual(resp, [9, 10])
        self.assertEqual(lp, [-0.9, 0.0])

    def test_loader_filters_by_session_id(self):
        """Loader scans all JSONL files, returns only matching session_id."""
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            rail_dir = Path(tmp)
            self._write_samples(rail_dir, {
                "a.jsonl": [
                    {"session_id": "WANT", "step_index": 1, "prompt_ids": [], "response_tokens": [], "logprobs": []},
                    {"session_id": "OTHER", "step_index": 0, "prompt_ids": [], "response_tokens": [], "logprobs": []},
                ],
                "b.jsonl": [
                    {"session_id": "WANT", "step_index": 0, "prompt_ids": [], "response_tokens": [], "logprobs": []},
                ],
            })
            loaded = self._load(rail_dir, "WANT", wait_seconds=1)
            self.assertEqual(len(loaded), 2)
            # Sorted by step_index
            self.assertEqual([s["step_index"] for s in loaded], [0, 1])

    def test_loader_returns_empty_on_miss(self):
        import tempfile
        with tempfile.TemporaryDirectory() as tmp:
            rail_dir = Path(tmp)
            self._write_samples(rail_dir, {
                "a.jsonl": [
                    {"session_id": "OTHER", "step_index": 0, "prompt_ids": [], "response_tokens": [], "logprobs": []}
                ],
            })
            self.assertEqual(self._load(rail_dir, "MISS", wait_seconds=1), [])

    def test_loader_handles_nonexistent_dir(self):
        self.assertEqual(self._load(Path("/nonexistent/jw_rail_dir"), "X", wait_seconds=0.1), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
