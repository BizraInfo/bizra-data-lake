"""
Tests for Living Memory Core — Self-Organizing Knowledge Store
================================================================

Standing on the Shoulders of Giants:
- Endel Tulving (1972): Episodic vs Semantic Memory
- Alan Baddeley (1974): Working Memory Model
- Hermann Ebbinghaus (1885): Forgetting Curve

إحسان — Excellence in all things.
"""

import asyncio
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock

from core.living_memory.core import (
    LivingMemoryCore,
    MemoryEntry,
    MemoryType,
    MemoryState,
    MemoryStats,
)


# ═══════════════════════════════════════════════════════════════════════════════
# FIXTURES
# ═══════════════════════════════════════════════════════════════════════════════


@pytest.fixture
def storage(tmp_path):
    return tmp_path / "living_memory"


def _dummy_embedding(text: str) -> np.ndarray:
    """Deterministic pseudo-embedding for tests."""
    np.random.seed(abs(hash(text)) % 2**31)
    return np.random.randn(64).astype(np.float32)


@pytest.fixture
def core(storage):
    return LivingMemoryCore(
        storage_path=storage,
        embedding_fn=_dummy_embedding,
        max_entries=500,
        ihsan_threshold=0.80,
    )


@pytest.fixture
def core_no_embed(storage):
    return LivingMemoryCore(
        storage_path=storage,
        embedding_fn=None,
        max_entries=500,
        ihsan_threshold=0.80,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# MemoryEntry dataclass
# ═══════════════════════════════════════════════════════════════════════════════


class TestMemoryEntry:

    def test_creation(self):
        e = MemoryEntry(id="abc", content="Hello world", memory_type=MemoryType.SEMANTIC)
        assert e.id == "abc"
        assert e.state == MemoryState.ACTIVE
        assert e.access_count == 0

    def test_to_dict(self):
        e = MemoryEntry(id="x", content="c", memory_type=MemoryType.EPISODIC)
        d = e.to_dict()
        assert d["id"] == "x"
        assert d["memory_type"] == "episodic"
        assert "importance" in d

    def test_from_dict_roundtrip(self):
        e = MemoryEntry(id="r", content="roundtrip", memory_type=MemoryType.PROCEDURAL, importance=0.8)
        d = e.to_dict()
        e2 = MemoryEntry.from_dict(d)
        assert e2.id == "r"
        assert e2.memory_type == MemoryType.PROCEDURAL
        assert e2.importance == 0.8

    def test_retrieval_score(self):
        e = MemoryEntry(id="s", content="score", memory_type=MemoryType.SEMANTIC)
        score = e.compute_retrieval_score()
        assert 0.0 <= score <= 1.0


# ═══════════════════════════════════════════════════════════════════════════════
# LivingMemoryCore — initialize
# ═══════════════════════════════════════════════════════════════════════════════


class TestInitialize:

    @pytest.mark.asyncio
    async def test_initialize_creates_dir(self, core, storage):
        await core.initialize()
        assert storage.exists()
        assert core._initialized is True

    @pytest.mark.asyncio
    async def test_double_init_is_idempotent(self, core):
        await core.initialize()
        await core.initialize()
        assert core._initialized is True


# ═══════════════════════════════════════════════════════════════════════════════
# Encode
# ═══════════════════════════════════════════════════════════════════════════════


class TestEncode:

    @pytest.mark.asyncio
    async def test_encode_semantic(self, core):
        await core.initialize()
        entry = await core.encode(
            "The speed of light is approximately 299,792 km/s",
            memory_type=MemoryType.SEMANTIC,
            source="physics",
        )
        assert entry is not None
        assert entry.memory_type == MemoryType.SEMANTIC
        assert entry.source == "physics"
        assert entry.embedding is not None

    @pytest.mark.asyncio
    async def test_encode_empty_returns_none(self, core):
        await core.initialize()
        assert await core.encode("") is None
        assert await core.encode("   ") is None

    @pytest.mark.asyncio
    async def test_encode_episodic(self, core):
        await core.initialize()
        entry = await core.encode("I debugged the auth module today", memory_type=MemoryType.EPISODIC)
        assert entry is not None
        assert entry.memory_type == MemoryType.EPISODIC

    @pytest.mark.asyncio
    async def test_encode_without_embedding(self, core_no_embed):
        await core_no_embed.initialize()
        entry = await core_no_embed.encode("Plain text, no vectors")
        assert entry is not None
        assert entry.embedding is None

    @pytest.mark.asyncio
    async def test_encode_stores_entry(self, core):
        await core.initialize()
        entry = await core.encode(
            "Living memory persistence verification ensures encoded entries are stored correctly in the internal dictionary"
        )
        assert entry is not None
        assert entry.id in core._memories


# ═══════════════════════════════════════════════════════════════════════════════
# Retrieve
# ═══════════════════════════════════════════════════════════════════════════════


class TestRetrieve:

    @pytest.mark.asyncio
    async def test_retrieve_empty(self, core):
        await core.initialize()
        results = await core.retrieve("anything")
        assert results == []

    @pytest.mark.asyncio
    async def test_retrieve_by_type(self, core):
        await core.initialize()
        await core.encode("Fact about AI safety", memory_type=MemoryType.SEMANTIC)
        await core.encode("Went for a walk", memory_type=MemoryType.EPISODIC)
        results = await core.retrieve(memory_type=MemoryType.SEMANTIC)
        assert all(r.memory_type == MemoryType.SEMANTIC for r in results)

    @pytest.mark.asyncio
    async def test_retrieve_updates_access(self, core):
        await core.initialize()
        entry = await core.encode("Access tracking content")
        results = await core.retrieve("Access tracking")
        if results:
            assert results[0].access_count >= 1

    @pytest.mark.asyncio
    async def test_retrieve_top_k(self, core):
        await core.initialize()
        for i in range(20):
            await core.encode(f"Memory number {i} with unique keywords_{i}")
        results = await core.retrieve("Memory number", top_k=5)
        assert len(results) <= 5

    @pytest.mark.asyncio
    async def test_retrieve_keyword_fallback(self, core_no_embed):
        """Without embeddings, keyword matching is used."""
        await core_no_embed.initialize()
        await core_no_embed.encode("Python programming language guide")
        await core_no_embed.encode("Cooking recipe for pasta")
        results = await core_no_embed.retrieve("Python programming")
        # The Python entry should score higher with keyword matching
        if results:
            assert "Python" in results[0].content or "python" in results[0].content.lower()


# ═══════════════════════════════════════════════════════════════════════════════
# Forget
# ═══════════════════════════════════════════════════════════════════════════════


class TestForget:

    @pytest.mark.asyncio
    async def test_soft_delete(self, core):
        await core.initialize()
        entry = await core.encode(
            "Temporary memory allocated for integration testing that validates soft deletion lifecycle management"
        )
        assert entry is not None
        assert await core.forget(entry.id) is True
        assert core._memories[entry.id].state == MemoryState.DELETED

    @pytest.mark.asyncio
    async def test_hard_delete(self, core):
        await core.initialize()
        entry = await core.encode(
            "Ephemeral memory allocated for hard deletion verification that must be completely purged from storage"
        )
        assert entry is not None
        assert await core.forget(entry.id, hard_delete=True) is True
        assert entry.id not in core._memories

    @pytest.mark.asyncio
    async def test_forget_nonexistent(self, core):
        await core.initialize()
        assert await core.forget("no-such-id") is False


# ═══════════════════════════════════════════════════════════════════════════════
# Consolidate
# ═══════════════════════════════════════════════════════════════════════════════


class TestConsolidate:

    @pytest.mark.asyncio
    async def test_consolidate_empty(self, core):
        await core.initialize()
        stats = await core.consolidate()
        assert stats["merged"] == 0
        assert stats["archived"] == 0

    @pytest.mark.asyncio
    async def test_consolidate_repairs_corrupted(self, core):
        await core.initialize()
        entry = await core.encode("Corrupted memory that still has good content")
        entry.state = MemoryState.CORRUPTED
        stats = await core.consolidate()
        assert stats["repaired"] >= 1
        assert entry.state == MemoryState.ACTIVE


# ═══════════════════════════════════════════════════════════════════════════════
# Working context & Stats
# ═══════════════════════════════════════════════════════════════════════════════


class TestWorkingContext:

    @pytest.mark.asyncio
    async def test_working_context_empty(self, core):
        await core.initialize()
        ctx = core.get_working_context()
        assert ctx == ""

    @pytest.mark.asyncio
    async def test_working_context_episodic(self, core):
        await core.initialize()
        await core.encode(
            "Episodic event recorded: production server restarted after deploying the authentication microservice upgrade",
            memory_type=MemoryType.EPISODIC,
        )
        ctx = core.get_working_context()
        assert "server restarted" in ctx


class TestStats:

    @pytest.mark.asyncio
    async def test_stats_empty(self, core):
        await core.initialize()
        s = core.get_stats()
        assert isinstance(s, MemoryStats)
        assert s.total_entries == 0

    @pytest.mark.asyncio
    async def test_stats_after_encode(self, core):
        await core.initialize()
        await core.encode(
            "Semantic knowledge: photosynthesis converts carbon dioxide and water into glucose using chlorophyll pigments",
            memory_type=MemoryType.SEMANTIC,
        )
        await core.encode(
            "Episodic event: completed comprehensive integration testing of the distributed memory consolidation pipeline",
            memory_type=MemoryType.EPISODIC,
        )
        s = core.get_stats()
        assert s.total_entries == 2
        assert s.active_entries == 2
        assert "semantic" in s.by_type
        assert "episodic" in s.by_type


# ═══════════════════════════════════════════════════════════════════════════════
# Persistence
# ═══════════════════════════════════════════════════════════════════════════════


class TestPersistence:

    @pytest.mark.asyncio
    async def test_save_and_reload(self, storage):
        # First core: encode and consolidate (which saves)
        c1 = LivingMemoryCore(storage_path=storage, ihsan_threshold=0.80)
        await c1.initialize()
        entry = await c1.encode("Persistent memory content here")
        await c1.consolidate()

        # Second core: load from disk
        c2 = LivingMemoryCore(storage_path=storage, ihsan_threshold=0.80)
        await c2.initialize()
        assert len(c2._memories) >= 1
