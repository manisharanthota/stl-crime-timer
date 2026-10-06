"""Secrets, local databases, and logs must never be committed."""

import fnmatch
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent

pytestmark = pytest.mark.skipif(
    shutil.which("git") is None or not (ROOT / ".git").exists(),
    reason="needs a git checkout",
)

FORBIDDEN = [".env", ".env.*", "*.db", "*.db.*", "*.db-*", "*.sqlite*", "logs/*", "*.log"]
ALLOWED = {".env.example"}


def git(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True)


def test_no_forbidden_files_tracked():
    tracked = git("ls-files").stdout.splitlines()
    bad = [
        path for path in tracked
        if path not in ALLOWED
        and any(fnmatch.fnmatch(Path(path).name, p) or fnmatch.fnmatch(path, p)
                for p in FORBIDDEN)
    ]
    assert bad == []


@pytest.mark.parametrize("path", [
    ".env", ".env.production", "stl_crime.db", "stl_crime.db.bak-20261004",
    "stl_crime.before-alerts.db", "stl_crime.db-journal", "logs/pipeline.log",
    "logs/pipeline.log.1", "debug.log",
])
def test_ignored(path):
    assert git("check-ignore", "-q", "--no-index", path).returncode == 0, path


@pytest.mark.parametrize("path", [".env.example", "alembic/env.py", "db.py"])
def test_not_ignored(path):
    assert git("check-ignore", "-q", "--no-index", path).returncode == 1, path
