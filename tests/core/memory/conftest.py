"""Shared fixtures for memory tests."""

from __future__ import annotations

import numpy as np
import pytest

from core.memory.config import HNSWConfig, MemoryConfig
from core.memory.types import MemoryKind, MemoryRecord, RecordState


@pytest.fixture
def tmp_memory_dir(tmp_path):
    """Provide a temporary directory for memory storage."""
    d = tmp_path / "agent_db"
    d.mkdir()
    return d


@pytest.fixture
def memory_config(tmp_memory_dir):
    """MemoryConfig pointing to a temp directory."""
    return MemoryConfig(
        data_dir=tmp_memory_dir,
        hnsw=HNSWConfig(dimensions=8, max_elements=1000),
    )


@pytest.fixture
def small_hnsw_config():
    """Small HNSW config for tests (dim=8, small capacity)."""
    return HNSWConfig(dimensions=8, max_elements=100)


def make_record(
    record_id: str = "test_001",
    content: str = "The Earth orbits the Sun",
    kind: MemoryKind = MemoryKind.SEMANTIC,
    embedding: list | None = None,
    importance: float = 0.5,
    source: str = "test",
) -> MemoryRecord:
    """Factory for test MemoryRecords."""
    return MemoryRecord(
        id=record_id,
        content=content,
        kind=kind,
        embedding=embedding,
        importance=importance,
        source=source,
    )


def random_embedding(dim: int = 8) -> list[float]:
    """Generate a random unit-length embedding vector."""
    vec = np.random.randn(dim).astype(np.float32)
    vec = vec / (np.linalg.norm(vec) + 1e-10)
    return vec.tolist()
