# BIZRA Test Configuration
# Pytest fixtures and configuration

import os
import sys
from pathlib import Path

import pytest

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


@pytest.fixture(scope="session")
def bizra_root():
    """Return BIZRA root directory"""
    return Path(__file__).parent.parent


@pytest.fixture(scope="session")
def sample_documents():
    """Sample documents for testing"""
    return [
        {
            "id": "doc1",
            "title": "Test Document 1",
            "content": "This is sample content for testing purposes.",
            "source": "test",
        },
        {
            "id": "doc2",
            "title": "Test Document 2",
            "content": "Another test document with different content.",
            "source": "test",
        },
    ]


@pytest.fixture(scope="session")
def sample_chunks():
    """Sample chunks for testing"""
    return [
        {
            "chunk_id": "chunk1",
            "doc_id": "doc1",
            "text": "Sample chunk text for testing.",
            "position": 0,
        },
        {
            "chunk_id": "chunk2",
            "doc_id": "doc1",
            "text": "Another chunk from the same document.",
            "position": 1,
        },
    ]


def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line(
        "markers", "requires_ollama: marks tests that require Ollama running"
    )
    config.addinivalue_line("markers", "requires_gpu: marks tests that require GPU")
    config.addinivalue_line(
        "markers", "requires_real_data: marks tests that need real parquet/index data"
    )


@pytest.fixture(autouse=True)
def _isolate_receipt_key_env(monkeypatch):
    """Prevent key env pollution between tests that mutate receipt signing env."""
    key = "BIZRA_RECEIPT_PUBLIC_KEY_HEX"
    original = os.environ.get(key)
    yield
    if original is None:
        monkeypatch.delenv(key, raising=False)
    else:
        monkeypatch.setenv(key, original)


@pytest.fixture(autouse=True)
def _detect_magicmock_file_leaks(request):
    """Fail tests that leak MagicMock objects to the filesystem.

    Root cause: unittest.mock.MagicMock.__truediv__() creates files at CWD when
    the result is used as a path (e.g., open(mock / 'file')). These garbage files
    poison os.walk() performance and mask real failures. See workspace_surgery.md.

    Checks repo root AND sovereign_state/ (the fallback path when isinstance
    guards reject MagicMock paths).
    """
    repo_root = Path(__file__).parent.parent
    before = set(repo_root.glob("<MagicMock*"))
    before |= set(repo_root.glob("*/<MagicMock*"))
    yield
    after = set(repo_root.glob("<MagicMock*"))
    after |= set(repo_root.glob("*/<MagicMock*"))
    leaked = after - before
    if leaked:
        for f in leaked:
            f.unlink(missing_ok=True)
        pytest.fail(
            f"Test leaked {len(leaked)} MagicMock file(s) to filesystem. "
            f"Fix: use tmp_path or SimpleNamespace(state_dir=tmp_path) "
            f"instead of bare MagicMock() for config paths. "
            f"Files: {[str(f.relative_to(repo_root))[:60] for f in leaked]}"
        )
