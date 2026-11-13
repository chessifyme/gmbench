"""Tests that read the repository's git state (the commit a run records, the tracked files and history) skip outside a git
checkout, e.g. in an unpacked archive of the public release."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
IN_GIT_CHECKOUT = subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=ROOT, capture_output=True, check=False).returncode == 0
requires_git = pytest.mark.skipif(not IN_GIT_CHECKOUT, reason="needs a git checkout of the repository")
