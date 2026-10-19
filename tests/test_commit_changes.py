"""Exercise compatibility commits against a local Git remote."""

import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class CommitChangesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.remote = self.root / "remote.git"
        self.writer = self.root / "writer"
        self.runner = self.root / "runner"
        self.git(self.root, "init", "--bare", str(self.remote))
        self.git(self.root, "clone", str(self.remote), str(self.writer))
        self.configure(self.writer)
        self.git(self.writer, "checkout", "-b", "main")
        (self.writer / "compatibility.json").write_text("{}\n")
        self.git(self.writer, "add", ".")
        self.git(self.writer, "commit", "-m", "Initial data")
        self.git(self.writer, "push", "origin", "main")
        self.git(
            self.root, "clone", "--branch", "main", str(self.remote), str(self.runner)
        )
        self.configure(self.runner)

    def git(self, cwd: Path, *args: str) -> str:
        return subprocess.run(
            ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
        ).stdout.strip()

    def configure(self, cwd: Path) -> None:
        self.git(cwd, "config", "user.name", "Test")
        self.git(cwd, "config", "user.email", "test@example.invalid")
        self.git(cwd, "config", "pull.rebase", "false")
        self.git(cwd, "config", "commit.gpgsign", "false")

    def run_script(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "-m", "scripts.commit-changes", "compatibility.json"],
            cwd=self.runner,
            env={**os.environ, "PYTHONPATH": str(ROOT)},
            capture_output=True,
            text=True,
        )

    def test_remote_already_has_generated_update(self) -> None:
        update = '{"example": {"failed_at": "2026-10-18"}}\n'
        (self.runner / "compatibility.json").write_text(update)
        (self.writer / "compatibility.json").write_text(update)
        self.git(self.writer, "add", ".")
        self.git(self.writer, "commit", "-m", "Concurrent update")
        self.git(self.writer, "push", "origin", "main")
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.git(self.runner, "status", "--porcelain"), "")
        self.assertEqual(
            self.git(self.runner, "rev-parse", "HEAD"),
            self.git(self.writer, "rev-parse", "HEAD"),
        )

    def test_new_update_is_committed_and_pushed(self) -> None:
        (self.runner / "compatibility.json").write_text('{"new": {}}\n')
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.git(self.runner, "status", "--porcelain"), "")
        self.assertEqual(
            self.git(self.runner, "rev-parse", "HEAD"),
            self.git(self.root, "--git-dir", str(self.remote), "rev-parse", "main"),
        )

    def test_unchanged_data_is_a_successful_noop(self) -> None:
        before = self.git(self.runner, "rev-parse", "HEAD")
        result = self.run_script()
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
        self.assertEqual(self.git(self.runner, "rev-parse", "HEAD"), before)


if __name__ == "__main__":
    unittest.main()
