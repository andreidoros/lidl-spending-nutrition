"""Temporary-file tests for Streamlit AI settings persistence."""

from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from ai_eval.providers import GROQ_DEFAULT_MODEL, parse_env_file
from app_settings import load_public_ai_settings, save_ai_settings


class AppSettingsTests(unittest.TestCase):
    def test_settings_update_preserves_other_values_and_never_returns_key(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "# keep this comment\n"
                "LIDL_REFRESH_TOKEN=synthetic-lidl-token\n"
                "LIDL_COUNTRY=ro\n"
                "CUSTOM_SETTING=leave-me\n",
                encoding="utf-8",
            )
            environment: dict[str, str] = {}

            saved = save_ai_settings(
                mode="categories_and_nutrition",
                model=GROQ_DEFAULT_MODEL,
                new_api_key="synthetic-groq-secret",
                env_path=path,
                environ=environment,
            )

            stored = parse_env_file(path)
            self.assertEqual(stored["LIDL_REFRESH_TOKEN"], "synthetic-lidl-token")
            self.assertEqual(stored["LIDL_COUNTRY"], "ro")
            self.assertEqual(stored["CUSTOM_SETTING"], "leave-me")
            self.assertEqual(stored["GROQ_API_KEY"], "synthetic-groq-secret")
            self.assertEqual(environment["GROQ_API_KEY"], "synthetic-groq-secret")
            self.assertTrue(saved.api_key_configured)
            self.assertNotIn("synthetic-groq-secret", repr(saved))
            self.assertIn("# keep this comment", path.read_text(encoding="utf-8"))

    def test_blank_key_keeps_existing_key_and_clear_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("GROQ_API_KEY=old-synthetic-key\n", encoding="utf-8")
            environment: dict[str, str] = {}

            saved = save_ai_settings(
                mode="categories",
                model=GROQ_DEFAULT_MODEL,
                env_path=path,
                environ=environment,
            )
            self.assertEqual(parse_env_file(path)["GROQ_API_KEY"], "old-synthetic-key")
            self.assertTrue(saved.api_key_configured)

            cleared = save_ai_settings(
                mode="off",
                model=GROQ_DEFAULT_MODEL,
                clear_api_key=True,
                env_path=path,
                environ=environment,
            )
            self.assertNotIn("GROQ_API_KEY", parse_env_file(path))
            self.assertFalse(cleared.api_key_configured)

    def test_public_settings_mask_presence_and_detect_first_run(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text("GROQ_API_KEY=never-display-this\n", encoding="utf-8")
            public = load_public_ai_settings(path, environ={})
            self.assertTrue(public.api_key_configured)
            self.assertFalse(public.settings_configured)
            self.assertNotIn("never-display-this", repr(public))

    def test_settings_append_preserves_final_env_value_without_trailing_newline(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text(
                "LIDL_REFRESH_TOKEN=synthetic-token\nLIDL_COUNTRY=ro",
                encoding="utf-8",
            )

            save_ai_settings(
                mode="categories",
                model=GROQ_DEFAULT_MODEL,
                env_path=path,
                environ={},
            )

            values = parse_env_file(path)
            self.assertEqual(values["LIDL_REFRESH_TOKEN"], "synthetic-token")
            self.assertEqual(values["LIDL_COUNTRY"], "ro")
            self.assertEqual(values["AI_ENRICHMENT_MODE"], "categories")
            self.assertEqual(values["GROQ_MODEL"], GROQ_DEFAULT_MODEL)

    def test_invalid_mode_model_and_multiline_keys_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            for kwargs in (
                {"mode": "unknown", "model": GROQ_DEFAULT_MODEL},
                {"mode": "categories", "model": "paid-model"},
                {"mode": "categories", "model": GROQ_DEFAULT_MODEL, "new_api_key": "line1\nline2"},
            ):
                with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                    save_ai_settings(env_path=path, **kwargs)
            self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main()
