"""[ENFORCEMENT: WIRED] Conftest for search tests.

This conftest used to construct module-level ``faiss``/``pandas`` mocks and
inject them into ``sys.modules`` at import time. That pattern leaked into
unrelated tests collected later by pytest (e.g. real-data integration suites
saw ``MagicMock`` instead of real ``pandas``).

The vector-search tests now own their own per-test mocks via ``monkeypatch``
(see ``test_vector_search.py``), so the conftest's only remaining
responsibility is to ensure ``sys.modules`` is restored to whatever it was
before the search tests ran. We snapshot ``faiss``/``pandas`` once here and
restore them after this conftest's scope finishes.
"""

from __future__ import annotations

import sys

import pytest

_SENTINEL = object()

_prev_faiss = sys.modules.get("faiss", _SENTINEL)
_prev_pandas = sys.modules.get("pandas", _SENTINEL)


def _restore_real_modules() -> None:
    """Restore the original ``faiss``/``pandas`` entries in ``sys.modules``."""
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
    """Restore ``sys.modules`` after each search test module finishes."""
    yield
    _restore_real_modules()
