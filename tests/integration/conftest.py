"""
Integration test configuration.

These tests require external dependencies (python-dotenv, Ollama, etc.)
and real data. Guard collection so missing deps don't break the full suite.

Standing on Giants: Dijkstra (testing discipline, 1970)
"""

import importlib
import os

# -- Collection guard: skip entire directory if python-dotenv is missing --
try:
    importlib.import_module("dotenv")
except ModuleNotFoundError:
    collect_ignore_glob = ["test_*.py"]


# -- Torch/pandas collection guard --
# test_live_pipeline.py and test_one_human.py import bizra_orchestrator at
# module-level, which triggers torch._dynamo during full-suite collection.
_HEAVY_ORCHESTRATOR_TESTS = {"test_live_pipeline.py", "test_one_human.py"}

collect_ignore = [
    f for f in _HEAVY_ORCHESTRATOR_TESTS if not os.environ.get("BIZRA_COLLECT_HEAVY")
]


def pytest_collect_file(parent, file_path):
    """Skip integration test files when python-dotenv is not installed."""
    try:
        importlib.import_module("dotenv")
    except ModuleNotFoundError:
        if file_path.suffix == ".py" and file_path.name.startswith("test_"):
            return None  # suppress collection
    return None  # let default collector handle it
