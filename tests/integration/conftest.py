"""
Integration test configuration.

These tests require external dependencies (python-dotenv, Ollama, etc.)
and real data. Guard collection so missing deps don't break the full suite.
"""
import importlib

import pytest

# ── Collection guard: skip entire directory if python-dotenv is missing ──
try:
    importlib.import_module("dotenv")
except ModuleNotFoundError:
    # Emit a clear skip reason instead of a noisy ImportError
    collect_ignore_glob = ["test_*.py"]


def pytest_collect_file(parent, file_path):
    """Skip integration test files when python-dotenv is not installed."""
    try:
        importlib.import_module("dotenv")
    except ModuleNotFoundError:
        if file_path.suffix == ".py" and file_path.name.startswith("test_"):
            return None  # suppress collection
    return None  # let default collector handle it
