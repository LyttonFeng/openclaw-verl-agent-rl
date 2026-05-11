"""Unit tests for jiuwen_lora_sync — no network, no GPU."""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import call

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from jiuwen_lora_sync import sync_lora_to_jiuwen  # noqa: E402


def _make_lora_dir(tmp: Path) -> Path:
    d = tmp / "lora_adapter"
    d.mkdir()
    (d / "adapter_config.json").write_text(json.dumps({"base_model_name_or_path": "x"}))
    (d / "adapter_model.safetensors").write_bytes(b"\0\0\0\0")
    return d


class FakeHttp:
    """Records calls and returns programmable responses."""
    def __init__(self, post_responses, get_responses):
        self.post_responses = list(post_responses)
        self.get_responses = list(get_responses)
        self.post_calls = []
        self.get_calls = []

    def post(self, url, body, timeout):
        self.post_calls.append((url, body))
        return self.post_responses.pop(0)

    def get(self, url, timeout):
        self.get_calls.append(url)
        return self.get_responses.pop(0)


class TestSyncLoraToJiuwen(unittest.TestCase):
    def test_happy_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            lora = _make_lora_dir(Path(tmp))
            http = FakeHttp(
                post_responses=[
                    (200, '{"ok":true}'),   # unload (no-op)
                    (200, '{"status":"Success: LoRA loaded"}'),  # load
                ],
                get_responses=[
                    (200, json.dumps({"data": [{"id": "my-step3"}, {"id": "base"}]})),
                ],
            )
            res = sync_lora_to_jiuwen(
                lora_path=lora, lora_name="my-step3",
                vllm_base="http://x:1", post_json=http.post, get_json=http.get,
            )
            self.assertEqual(res["status"], "ok")
            self.assertEqual(res["lora_name"], "my-step3")
            self.assertIn("my-step3", res["models"])
            self.assertIn("loaded_at", res)
            # Verify call order: unload, then load, then GET /v1/models
            self.assertEqual(len(http.post_calls), 2)
            self.assertTrue(http.post_calls[0][0].endswith("/v1/unload_lora_adapter"))
            self.assertTrue(http.post_calls[1][0].endswith("/v1/load_lora_adapter"))
            self.assertEqual(http.post_calls[1][1]["lora_name"], "my-step3")
            self.assertEqual(http.post_calls[1][1]["lora_path"], str(lora.resolve()))

    def test_load_http_failure_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            lora = _make_lora_dir(Path(tmp))
            http = FakeHttp(
                post_responses=[
                    (200, '{}'),  # unload
                    (500, '{"error":"vllm exploded"}'),  # load fails
                ],
                get_responses=[],
            )
            res = sync_lora_to_jiuwen(
                lora_path=lora, lora_name="x",
                vllm_base="http://x:1", post_json=http.post, get_json=http.get,
            )
            self.assertEqual(res["status"], "error")
            self.assertIn("500", res["error"])
            self.assertIn("vllm exploded", res["error"])

    def test_models_missing_lora_returns_error(self):
        with tempfile.TemporaryDirectory() as tmp:
            lora = _make_lora_dir(Path(tmp))
            http = FakeHttp(
                post_responses=[
                    (200, '{}'),
                    (200, '{"status":"ok"}'),
                ],
                get_responses=[
                    (200, json.dumps({"data": [{"id": "other-lora"}]})),
                ],
            )
            res = sync_lora_to_jiuwen(
                lora_path=lora, lora_name="missing",
                vllm_base="http://x:1", post_json=http.post, get_json=http.get,
            )
            self.assertEqual(res["status"], "error")
            self.assertIn("absent from /v1/models", res["error"])

    def test_skip_unload(self):
        with tempfile.TemporaryDirectory() as tmp:
            lora = _make_lora_dir(Path(tmp))
            http = FakeHttp(
                post_responses=[(200, '{}')],  # only load (no unload)
                get_responses=[(200, json.dumps({"data": [{"id": "v42"}]}))],
            )
            res = sync_lora_to_jiuwen(
                lora_path=lora, lora_name="v42",
                vllm_base="http://x:1", unload_existing=False,
                post_json=http.post, get_json=http.get,
            )
            self.assertEqual(res["status"], "ok")
            self.assertEqual(len(http.post_calls), 1)
            self.assertTrue(http.post_calls[0][0].endswith("/v1/load_lora_adapter"))

    def test_missing_adapter_config_raises(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty"
            empty.mkdir()
            with self.assertRaises(FileNotFoundError):
                sync_lora_to_jiuwen(
                    lora_path=empty, lora_name="x", vllm_base="http://x:1",
                    post_json=lambda *a, **k: (200, "{}"),
                    get_json=lambda *a, **k: (200, "{}"),
                )

    def test_models_endpoint_failure(self):
        with tempfile.TemporaryDirectory() as tmp:
            lora = _make_lora_dir(Path(tmp))
            http = FakeHttp(
                post_responses=[(200, '{}'), (200, '{}')],
                get_responses=[(503, "service unavailable")],
            )
            res = sync_lora_to_jiuwen(
                lora_path=lora, lora_name="x",
                vllm_base="http://x:1", post_json=http.post, get_json=http.get,
            )
            self.assertEqual(res["status"], "error")
            self.assertIn("503", res["error"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
