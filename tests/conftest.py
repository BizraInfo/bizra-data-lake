# BIZRA Test Configuration
# Pytest fixtures and configuration

import pytest
import sys
from pathlib import Path

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
            "source": "test"
        },
        {
            "id": "doc2",
            "title": "Test Document 2",
            "content": "Another test document with different content.",
            "source": "test"
        }
    ]


@pytest.fixture(scope="session")
def sample_chunks():
    """Sample chunks for testing"""
    return [
        {
            "chunk_id": "chunk1",
            "doc_id": "doc1",
            "text": "Sample chunk text for testing.",
            "position": 0
        },
        {
            "chunk_id": "chunk2",
            "doc_id": "doc1",
            "text": "Another chunk from the same document.",
            "position": 1
        }
    ]


def pytest_configure(config):
    """Configure pytest markers"""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "requires_ollama: marks tests that require Ollama running"
    )
    config.addinivalue_line(
        "markers", "requires_gpu: marks tests that require GPU"
    )
