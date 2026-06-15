"""Pytest configuration — ensure repo root is importable."""

import sys
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

# Add the repo root to sys.path so that modules like mflux_cache and server
# can be imported directly in tests.
_repo_root = str(Path(__file__).resolve().parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)


@pytest.fixture
def mock_queue():
    """Return a MagicMock suitable for patching server_module._queue."""
    return MagicMock()


@pytest.fixture
def mock_cache():
    """Return a MagicMock suitable for patching server_module._cache."""
    return MagicMock()


@pytest.fixture
def mock_worker_manager():
    """Return an AsyncMock suitable for patching server_module._worker_manager."""
    return AsyncMock()
