"""Offline contract checks for the Windows Streamlit launcher."""

import re
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class WindowsLauncherContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.launcher = (PROJECT_ROOT / "run.bat").read_text(encoding="utf-8")
        cls.readme = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")

    @unittest.skipUnless(os.name == "nt", "Requires Windows batch execution")
    def test_python_version_check_executes_through_cmd(self):
        version_line = next(
            line for line in self.launcher.splitlines() if line.startswith("python -c ")
        )
        # Select the test interpreter while preserving cmd.exe's batch parsing.
        version_line = version_line.replace("python", f'"{sys.executable}"', 1)
        with tempfile.TemporaryDirectory(prefix="launcher version ") as directory:
            batch = Path(directory) / "version_probe.bat"
            batch.write_text(
                "@echo off\nsetlocal enabledelayedexpansion\n" + version_line + "\n",
                encoding="utf-8",
            )
            result = subprocess.run(
                ["cmd.exe", "/d", "/c", str(batch)],
                capture_output=True,
                text=True,
                timeout=15,
                check=False,
            )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn(f"Detected Python {sys.version.split()[0]}", result.stdout)
        self.assertNotIn("SyntaxError", result.stderr)

    def test_launcher_starts_streamlit_on_loopback_and_fixed_port(self):
        self.assertIn(
            r"-m streamlit run app.py --server.address 127.0.0.1 --server.port 8501",
            self.launcher,
        )
        port_check = self.launcher.index("connect_ex(('127.0.0.1', 8501))")
        streamlit_start = self.launcher.index("-m streamlit run app.py")
        self.assertLess(port_check, streamlit_start)
        self.assertIn("Port 8501 is already in use", self.launcher)

    def test_launcher_never_runs_sync_automatically(self):
        self.assertNotIn("sync_incremental.py", self.launcher)
        self.assertIn("No synchronization starts automatically", self.launcher)

    def test_launcher_disables_streamlit_email_prompt(self):
        start_line = next(
            line for line in self.launcher.splitlines() if "-m streamlit run app.py" in line
        )
        self.assertIn("--server.showEmailPrompt false", start_line)

    def test_missing_token_runs_authentication_and_rechecks_token(self):
        first_check = self.launcher.index("call :CHECK_LIDL_TOKEN")
        auth_call = self.launcher.index("python.exe auth_login.py")
        second_check = self.launcher.index("call :CHECK_LIDL_TOKEN", first_check + 1)
        streamlit_start = self.launcher.index("-m streamlit run app.py")

        self.assertLess(first_check, auth_call)
        self.assertLess(auth_call, second_check)
        self.assertLess(second_check, streamlit_start)
        self.assertIn("LIDL_REFRESH_TOKEN=", self.launcher)
        self.assertIn("Authentication finished without saving a Lidl refresh token", self.launcher)

    def test_token_check_matches_loader_decoding_and_duplicate_semantics(self):
        token_check_lines = [line for line in self.launcher.splitlines() if "p=Path('.env')" in line]
        self.assertEqual(len(token_check_lines), 1)
        match = re.search(r'-c "([^"]+)"', token_check_lines[0])
        self.assertIsNotNone(match)
        token_check = match.group(1)

        cases = (
            ("LIDL_REFRESH_TOKEN=valid-token\n", 0),
            ("LIDL_REFRESH_TOKEN=valid-token\nLIDL_REFRESH_TOKEN=   \n", 1),
            ("\ufeffLIDL_REFRESH_TOKEN=valid-token\n", 1),
            ("LIDL_REFRESH_TOKEN=\nLIDL_REFRESH_TOKEN=valid-token\n", 0),
            ("OTHER=value\u2028LIDL_REFRESH_TOKEN=valid-token\n", 1),
        )
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            for contents, expected_returncode in cases:
                with self.subTest(contents=contents):
                    env_file.write_text(contents, encoding="utf-8")
                    result = subprocess.run(
                        [sys.executable, "-c", token_check],
                        cwd=directory,
                        capture_output=True,
                        text=True,
                        check=False,
                    )
                    self.assertEqual(result.returncode, expected_returncode, result.stderr)

    def test_dependencies_and_python_minimum_are_checked(self):
        self.assertIn("(3, 10)", self.launcher)
        self.assertIn("import duckdb, requests, lidlplus_api, zstandard, playwright, httpx, pandas, streamlit", self.launcher)

    def test_readme_documents_local_only_explicit_sync_startup(self):
        self.assertIn("http://127.0.0.1:8501", self.readme)
        self.assertIn("Does **not** start synchronization automatically", self.readme)
        self.assertIn("Sync is an explicit button in the app", self.readme)


if __name__ == "__main__":
    unittest.main()
