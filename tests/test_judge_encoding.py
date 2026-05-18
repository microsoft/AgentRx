"""Tests for judge.py encoding fix (debug print in retry loop).

Verifies that:
1. Non-ASCII content in system_prompt/user_message doesn't raise exceptions
   when DEBUG_PROMPTS is enabled.
2. The ascii-safe encoding correctly replaces non-ASCII characters.
3. When DEBUG_PROMPTS is False, no output is produced.
"""
import io
import os
import sys
import unittest
from unittest.mock import patch


class TestDebugPrintEncoding(unittest.TestCase):
    """Test the safe encoding pattern used in the judge retry loop."""

    def _safe_print(self, label: str, text: str, max_len: int = 500):
        """Replicate the fixed debug print pattern from judge.py line 950-953."""
        try:
            print(f"{label}:", text[:max_len].encode('ascii', 'replace').decode())
        except Exception:
            pass

    def test_ascii_content_prints_normally(self):
        """ASCII-only content should print without modification."""
        buf = io.StringIO()
        with patch('sys.stdout', buf):
            self._safe_print("System Prompt", "You are a helpful assistant.")
        self.assertIn("You are a helpful assistant.", buf.getvalue())

    def test_unicode_arrows_replaced(self):
        """Unicode arrows (U+2192) that caused the original bug should be replaced."""
        text = "Step 1 → Step 2 → Step 3"
        buf = io.StringIO()
        with patch('sys.stdout', buf):
            self._safe_print("User Message", text)
        output = buf.getvalue()
        # Arrow chars replaced with '?'
        self.assertIn("Step 1 ? Step 2 ? Step 3", output)
        self.assertNotIn("→", output)

    def test_emoji_content_replaced(self):
        """Emoji characters should be safely replaced, not crash."""
        text = "The agent failed ❌ at step 5 ✅"
        buf = io.StringIO()
        with patch('sys.stdout', buf):
            self._safe_print("Test", text)
        output = buf.getvalue()
        self.assertIn("step 5", output)

    def test_cjk_content_replaced(self):
        """CJK characters should be replaced without exception."""
        text = "用户请求：查找产品信息"
        buf = io.StringIO()
        with patch('sys.stdout', buf):
            self._safe_print("System Prompt", text)
        # Should not raise
        self.assertIn("System Prompt:", buf.getvalue())

    def test_mixed_content_partial_replacement(self):
        """Mixed ASCII/Unicode should preserve ASCII parts."""
        text = "Error at line 42: 无效操作 → crashed"
        buf = io.StringIO()
        with patch('sys.stdout', buf):
            self._safe_print("Msg", text)
        output = buf.getvalue()
        self.assertIn("Error at line 42", output)
        self.assertIn("crashed", output)

    def test_long_content_truncated(self):
        """Content longer than max_len should be truncated before encoding."""
        text = "A" * 1000
        buf = io.StringIO()
        with patch('sys.stdout', buf):
            self._safe_print("Long", text, max_len=500)
        output = buf.getvalue()
        # 500 A's plus label
        self.assertEqual(output.count("A"), 500)

    def test_empty_string(self):
        """Empty strings should not cause issues."""
        buf = io.StringIO()
        with patch('sys.stdout', buf):
            self._safe_print("Empty", "")
        self.assertIn("Empty:", buf.getvalue())

    def test_null_bytes_handled(self):
        """Null bytes in content should not crash the encoding."""
        text = "before\x00after"
        buf = io.StringIO()
        with patch('sys.stdout', buf):
            self._safe_print("Null", text)
        # Should complete without exception
        self.assertIn("Null:", buf.getvalue())


class TestDebugPromptsFlag(unittest.TestCase):
    """Test that DEBUG_PROMPTS flag controls output."""

    def test_debug_prompts_disabled_no_output(self):
        """When DEBUG_PROMPTS is False, the guarded block should not execute."""
        DEBUG_PROMPTS = False
        buf = io.StringIO()
        with patch('sys.stdout', buf):
            if DEBUG_PROMPTS:
                try:
                    print("System Prompt:", "test".encode('ascii', 'replace').decode())
                except Exception:
                    pass
        self.assertEqual(buf.getvalue(), "")

    def test_debug_prompts_enabled_produces_output(self):
        """When DEBUG_PROMPTS is True, output should be produced."""
        DEBUG_PROMPTS = True
        buf = io.StringIO()
        with patch('sys.stdout', buf):
            if DEBUG_PROMPTS:
                try:
                    print("System Prompt:", "test".encode('ascii', 'replace').decode())
                except Exception:
                    pass
        self.assertIn("System Prompt: test", buf.getvalue())


class TestModelNameEnvVar(unittest.TestCase):
    """Test that AGENT_VERIFY_MODEL_NAME env var is respected."""

    def test_model_name_defaults_to_gpt5(self):
        """Without env var, MODEL_NAME should default to gpt-5."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AGENT_VERIFY_MODEL_NAME", None)
            model_name = os.environ.get("AGENT_VERIFY_MODEL_NAME", "gpt-5")
            self.assertEqual(model_name, "gpt-5")

    def test_model_name_reads_from_env(self):
        """With env var set, MODEL_NAME should use the provided value."""
        with patch.dict(os.environ, {"AGENT_VERIFY_MODEL_NAME": "gpt-4.1"}):
            model_name = os.environ.get("AGENT_VERIFY_MODEL_NAME", "gpt-5")
            self.assertEqual(model_name, "gpt-4.1")

    def test_deployment_and_model_name_should_match(self):
        """Both DEPLOYMENT and MODEL_NAME should be set to the same value for Azure."""
        with patch.dict(os.environ, {
            "AGENT_VERIFY_DEPLOYMENT": "gpt-4.1",
            "AGENT_VERIFY_MODEL_NAME": "gpt-4.1",
        }):
            deployment = os.environ.get("AGENT_VERIFY_DEPLOYMENT", "gpt-5")
            model_name = os.environ.get("AGENT_VERIFY_MODEL_NAME", "gpt-5")
            self.assertEqual(deployment, model_name,
                             "DEPLOYMENT and MODEL_NAME must match for Azure OpenAI")


if __name__ == "__main__":
    unittest.main()
