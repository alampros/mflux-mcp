"""Pytest configuration — ensure repo root is importable."""

import sys
from pathlib import Path

# Add the repo root to sys.path so that modules like mflux_cache and server
# can be imported directly in tests.
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)
