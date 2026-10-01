"""Shared pytest fixtures."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Make `src/` importable when running `pytest` from the project root without install.
ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


@pytest.fixture(autouse=True)
def _isolated_env(tmp_path, monkeypatch):
    """Each test gets a clean ULTRADEBATE_BASE inside tmp_path."""
    monkeypatch.setenv("ULTRADEBATE_BASE", str(tmp_path / ".ultradebate"))
    monkeypatch.delenv("ULTRADEBATE_MOCK", raising=False)
    yield


@pytest.fixture
def session_dir(tmp_path):
    d = tmp_path / "session_dir"
    d.mkdir()
    return d
