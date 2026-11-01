import asyncio
import json
import os
import sys
import tempfile
import types
import unittest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# The real `groq` package is only needed for live calls; tests use a minimal stand-in
# when it is not installed.
try:
    import groq  # noqa: F401
except ImportError:
    fake = types.ModuleType("groq")

    class _Err(Exception):
        pass

    for name in ("BadRequestError", "NotFoundError", "RateLimitError", "AuthenticationError",
                 "APITimeoutError", "APIConnectionError"):
        setattr(fake, name, type(name, (_Err,), {}))
    fake.AsyncGroq = object
    sys.modules["groq"] = fake
    import groq  # noqa: F401

import config as cfgmod
import vision


class TestModelSelection(unittest.TestCase):
    LIVE = ["llama-3.3-70b-versatile", "openai/gpt-oss-120b", "qwen/qwen3.6-27b",
            "qwen/qwen3.8-27b", "whisper-large-v3", "meta-llama/llama-prompt-guard-2-86m"]

    def test_configured_model_wins(self):
        self.assertEqual(vision.select_best_vision_model(self.LIVE, "qwen/qwen3.6-27b"), "qwen/qwen3.6-27b")

    def test_retired_scout_falls_back_to_newest_qwen(self):
        got = vision.select_best_vision_model(self.LIVE, "meta-llama/llama-4-scout-17b-16e-instruct")
        self.assertEqual(got, "qwen/qwen3.8-27b")

    def test_only_older_qwen_available(self):
        got = vision.select_best_vision_model(["qwen/qwen3.6-27b", "llama-3.3-70b-versatile"], "x")
        self.assertEqual(got, "qwen/qwen3.6-27b")

    def test_text_only_account_returns_none(self):
        self.assertIsNone(vision.select_best_vision_model(["llama-3.3-70b-versatile", "openai/gpt-oss-120b"], "x"))

    def test_name_heuristic(self):
        self.assertTrue(vision.is_vision_model_name("qwen/qwen3.8-27b"))
        self.assertFalse(vision.is_vision_model_name("qwen/qwen3-32b"))  # text-only, retired
        self.assertFalse(vision.is_vision_model_name("whisper-large-v3"))
        self.assertFalse(vision.is_vision_model_name("llama-3.2-11b-vision-preview"))


class TestConfigMigration(unittest.TestCase):
    def test_retired_model_is_migrated(self):
        cfg = cfgmod._validate_and_merge({"model": "meta-llama/llama-4-scout-17b-16e-instruct"})
        self.assertEqual(cfg["model"], "qwen/qwen3.8-27b")

    def test_custom_model_kept_and_defaults(self):
        cfg = cfgmod._validate_and_merge({"model": "qwen/qwen3.6-27b", "reasoning_effort": "low"})
        self.assertEqual(cfg["model"], "qwen/qwen3.6-27b")
        self.assertEqual(cfg["reasoning_effort"], "low")
        self.assertEqual(cfgmod._validate_and_merge({})["reasoning_effort"], "none")

    def test_invalid_reasoning_effort_ignored(self):
        self.assertEqual(cfgmod._validate_and_merge({"reasoning_effort": "turbo"})["reasoning_effort"], "none")


class _Resp:
    def __init__(self, text):
        self.choices = [types.SimpleNamespace(message=types.SimpleNamespace(content=text))]


class _FakeClient:
    """Records each request; can reject chosen parameters like the real API would."""

    def __init__(self, reject=()):
        self.calls = []
        self.reject = set(reject)
        self.chat = types.SimpleNamespace(completions=types.SimpleNamespace(create=self._create))

    async def _create(self, **kw):
        self.calls.append(dict(kw))
        if "reasoning" in self.reject and "extra_body" in kw:
            raise groq.BadRequestError("unsupported parameter: reasoning_effort")
        if "json" in self.reject and "response_format" in kw:
            raise groq.BadRequestError("response_format json_object not supported")
        return _Resp('{"found": true}')


class TestRequestBuilding(unittest.TestCase):
    def _run(self, client, effort="none", model="qwen/qwen3.8-27b"):
        eng = vision.VisionEngine()
        eng._reasoning_effort = effort
        out = asyncio.run(eng._call_chat_completion(client, model, [], 0.0, 100, None, 2.0))
        return eng, out

    def test_reasoning_none_sent_for_qwen(self):
        c = _FakeClient()
        self._run(c)
        self.assertEqual(c.calls[0]["extra_body"], {"reasoning_effort": "none"})
        self.assertEqual(c.calls[0]["max_tokens"], 100)

    def test_higher_effort_hides_reasoning_and_adds_token_headroom(self):
        c = _FakeClient()
        self._run(c, effort="medium")
        self.assertEqual(c.calls[0]["extra_body"]["reasoning_format"], "hidden")
        self.assertEqual(c.calls[0]["max_tokens"], 100 + 1024)

    def test_non_qwen_model_gets_no_reasoning_param(self):
        c = _FakeClient()
        self._run(c, model="llama-3.3-70b-versatile")
        self.assertNotIn("extra_body", c.calls[0])

    def test_rejected_reasoning_param_is_dropped_and_remembered(self):
        c = _FakeClient(reject={"reasoning"})
        eng, out = self._run(c)
        self.assertEqual(out, '{"found": true}')
        self.assertEqual(len(c.calls), 2)
        self.assertNotIn("extra_body", c.calls[1])
        self.assertFalse(eng._reasoning_supported["qwen/qwen3.8-27b"])

    def test_both_params_rejected_still_succeeds(self):
        c = _FakeClient(reject={"reasoning", "json"})
        _, out = self._run(c)
        self.assertEqual(out, '{"found": true}')
        self.assertEqual(len(c.calls), 3)


if __name__ == "__main__":
    unittest.main()
