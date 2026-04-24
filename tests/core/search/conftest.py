"""Conftest for search tests — ensures sys.modules cleanup after mock injection."""

from __future__ import annotations

import sys
from importlib.machinery import ModuleSpec
from types import ModuleType
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# faiss/pandas mocks used by vector-search tests.
#
# Do not inject these at conftest import time: pytest may collect this
# conftest before unrelated modules run, and global pandas/faiss pollution
# makes real-data tests see MagicMock objects.
# ---------------------------------------------------------------------------

_SENTINEL = object()

# Save originals
_prev_faiss = sys.modules.get("faiss", _SENTINEL)
_prev_pandas = sys.modules.get("pandas", _SENTINEL)

# Create mocks
mock_faiss = ModuleType("faiss")
mock_faiss.__spec__ = ModuleSpec("faiss", loader=None)
mock_faiss.METRIC_L2 = 1  # type: ignore[attr-defined]
mock_faiss.METRIC_INNER_PRODUCT = 0  # type: ignore[attr-defined]
mock_faiss.read_index = MagicMock()  # type: ignore[attr-defined]

mock_pd_module = ModuleType("pandas")
mock_pd_module.__spec__ = ModuleSpec("pandas", loader=None)
mock_pd_module.read_parquet = MagicMock()  # type: ignore[attr-defined]
mock_pd_module.DataFrame = MagicMock  # type: ignore[attr-defined]


def _restore_real_modules() -> None:
    """Restore original faiss/pandas in sys.modules."""
    if _prev_faiss is _SENTINEL:
        sys.modules.pop("faiss", None)
    else:
        sys.modules["faiss"] = _prev_faiss
    if _prev_pandas is _SENTINEL:
        sys.modules.pop("pandas", None)
    else:
        sys.modules["pandas"] = _prev_pandas


def pytest_unconfigure() -> None:
    """Restore real modules when pytest finishes with this conftest's scope."""
    _restore_real_modules()


@pytest.fixture(autouse=True, scope="module")
def _search_module_cleanup():
    """Restore sys.modules after all tests in each search test module finish."""
    yield
    _restore_real_modules()
