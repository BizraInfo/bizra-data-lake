"""Tests for AgentDB Memory Patterns: SessionMemory, FactStore,
HierarchicalMemory, MemoryConsolidator, ContextSynthesizer."""


import pytest

from core.memory.agent_db import AgentDB
from core.memory.config import HNSWConfig, MemoryConfig
from core.memory.memory_patterns import (
    ConsolidationResult,
    ContextSynthesizer,
    FactStore,
    HierarchicalMemory,
    MemoryConsolidator,
    MemoryTier,
    SessionMemory,
    SynthesizedContext,
)
from core.memory.types import MemoryKind


@pytest.fixture
def tmp_db(tmp_path):
    """Provide an initialized AgentDB in a temp directory."""
    config = MemoryConfig(
        data_dir=tmp_path / "agent_db",
        hnsw=HNSWConfig(dimensions=3, max_elements=100),
        auto_embed=False,
    )
    db = AgentDB(config)
    db.initialize()
    yield db
    db.close()


# ── Session Memory Tests ─────────────────────────────────────────────


class TestSessionMemory:
    def test_store_turn(self, tmp_db):
        sm = SessionMemory(tmp_db, "sess-001")
        rec = sm.store_turn("user", "Hello world")
        assert rec.kind == MemoryKind.EPISODIC
        assert "session" in rec.tags
        assert "role:user" in rec.tags
        assert rec.metadata["session_id"] == "sess-001"
        assert rec.metadata["role"] == "user"
        assert rec.metadata["turn_index"] == 1

    def test_store_multiple_turns(self, tmp_db):
        sm = SessionMemory(tmp_db, "sess-002")
        sm.store_turn("user", "What is BIZRA?")
        sm.store_turn("assistant", "BIZRA is a sovereign agentic system.")
        sm.store_turn("user", "Tell me more.")
        history = sm.get_history(limit=10)
        assert len(history) == 3
        # Should be in chronological order
        assert history[0].metadata["turn_index"] == 1
        assert history[2].metadata["turn_index"] == 3

    def test_get_recent_context(self, tmp_db):
        sm = SessionMemory(tmp_db, "sess-003")
        sm.store_turn("user", "Hello")
        sm.store_turn("assistant", "Hi there")
        sm.store_turn("user", "How are you?")
        ctx = sm.get_recent_context(limit=2)
        assert "[assistant]: Hi there" in ctx
        assert "[user]: How are you?" in ctx

    def test_clear_session(self, tmp_db):
        sm = SessionMemory(tmp_db, "sess-004")
        sm.store_turn("user", "Test message 1")
        sm.store_turn("assistant", "Test reply 1")
        cleared = sm.clear_session(archive=True)
        assert cleared == 2
        # After clear, history should be empty
        history = sm.get_history()
        assert len(history) == 0

    def test_isolated_sessions(self, tmp_db):
        sm1 = SessionMemory(tmp_db, "sess-A")
        sm2 = SessionMemory(tmp_db, "sess-B")
        sm1.store_turn("user", "Session A content")
        sm2.store_turn("user", "Session B content")
        h1 = sm1.get_history()
        h2 = sm2.get_history()
        assert len(h1) == 1
        assert len(h2) == 1
        assert "Session A" in h1[0].content
        assert "Session B" in h2[0].content


# ── Fact Store Tests ─────────────────────────────────────────────────


class TestFactStore:
    def test_store_and_retrieve_fact(self, tmp_db):
        fs = FactStore(tmp_db)
        fs.store_fact("user_pref", "language", "English", confidence=1.0)
        rec = fs.get_fact("user_pref", "language")
        assert rec is not None
        assert rec.content == "English"
        assert rec.metadata["confidence"] == 1.0

    def test_get_facts_by_category(self, tmp_db):
        fs = FactStore(tmp_db)
        fs.store_fact("config", "theme", "dark")
        fs.store_fact("config", "timezone", "UTC")
        fs.store_fact("other", "foo", "bar")
        facts = fs.get_facts("config")
        assert len(facts) == 2
        contents = {f.content for f in facts}
        assert "dark" in contents
        assert "UTC" in contents

    def test_min_confidence_filter(self, tmp_db):
        fs = FactStore(tmp_db)
        fs.store_fact("scores", "high", "good", confidence=0.9)
        fs.store_fact("scores", "low", "poor", confidence=0.3)
        facts = fs.get_facts("scores", min_confidence=0.5)
        assert len(facts) == 1
        assert facts[0].content == "good"

    def test_update_confidence(self, tmp_db):
        fs = FactStore(tmp_db)
        fs.store_fact("data", "key1", "value1", confidence=0.5)
        success = fs.update_confidence("data", "key1", 0.95)
        assert success is True
        rec = fs.get_fact("data", "key1")
        assert rec is not None
        assert rec.metadata["confidence"] == 0.95

    def test_forget_fact(self, tmp_db):
        fs = FactStore(tmp_db)
        fs.store_fact("temp", "k", "v")
        assert fs.get_fact("temp", "k") is not None
        result = fs.forget_fact("temp", "k")
        assert result is True

    def test_forget_nonexistent(self, tmp_db):
        fs = FactStore(tmp_db)
        result = fs.forget_fact("ghost", "missing")
        assert result is False

    def test_update_nonexistent(self, tmp_db):
        fs = FactStore(tmp_db)
        result = fs.update_confidence("ghost", "missing", 0.99)
        assert result is False


# ── Hierarchical Memory Tests ────────────────────────────────────────


class TestHierarchicalMemory:
    def test_store_immediate(self, tmp_db):
        hm = HierarchicalMemory(tmp_db)
        rec = hm.store("Quick thought", tier=MemoryTier.IMMEDIATE)
        assert rec.kind == MemoryKind.WORKING
        assert "tier:immediate" in rec.tags

    def test_store_short_term(self, tmp_db):
        hm = HierarchicalMemory(tmp_db)
        rec = hm.store("Session note", tier=MemoryTier.SHORT_TERM)
        assert rec.kind == MemoryKind.EPISODIC
        assert "tier:short_term" in rec.tags

    def test_store_long_term(self, tmp_db):
        hm = HierarchicalMemory(tmp_db)
        rec = hm.store("Important knowledge", tier=MemoryTier.LONG_TERM)
        assert rec.kind == MemoryKind.SEMANTIC
        assert "tier:long_term" in rec.tags

    def test_retrieve_by_tier(self, tmp_db):
        hm = HierarchicalMemory(tmp_db)
        hm.store("Immediate data", tier=MemoryTier.IMMEDIATE)
        hm.store("Long-term knowledge", tier=MemoryTier.LONG_TERM)
        immediate = hm.retrieve(tier=MemoryTier.IMMEDIATE)
        long_term = hm.retrieve(tier=MemoryTier.LONG_TERM)
        assert len(immediate) == 1
        assert len(long_term) == 1

    def test_promote(self, tmp_db):
        hm = HierarchicalMemory(tmp_db)
        rec = hm.store("Promote me", tier=MemoryTier.SHORT_TERM)
        success = hm.promote(rec.id, to_tier=MemoryTier.LONG_TERM)
        assert success is True
        # Verify it's now in long_term tier
        updated = tmp_db.retrieve(rec.id)
        assert updated is not None
        assert "tier:long_term" in updated.tags
        assert updated.metadata["memory_tier"] == MemoryTier.LONG_TERM
        assert updated.kind == MemoryKind.SEMANTIC

    def test_promote_nonexistent(self, tmp_db):
        hm = HierarchicalMemory(tmp_db)
        assert hm.promote("nonexistent-id", MemoryTier.LONG_TERM) is False

    def test_tier_stats(self, tmp_db):
        hm = HierarchicalMemory(tmp_db)
        hm.store("A", tier=MemoryTier.IMMEDIATE)
        hm.store("B", tier=MemoryTier.SHORT_TERM)
        hm.store("C", tier=MemoryTier.LONG_TERM)
        stats = hm.tier_stats()
        assert stats["immediate"] == 1
        assert stats["short_term"] == 1
        assert stats["long_term"] == 1


# ── Memory Consolidator Tests ────────────────────────────────────────


class TestMemoryConsolidator:
    def test_prune_low_importance(self, tmp_db):
        mc = MemoryConsolidator(tmp_db)
        tmp_db.store("Important", importance=0.9)
        tmp_db.store("Junk item with unique text", importance=0.01)
        result = mc.consolidate(min_importance=0.05)
        assert result.pruned >= 1
        assert result.total_after < result.total_before

    def test_consolidate_result_structure(self, tmp_db):
        mc = MemoryConsolidator(tmp_db)
        result = mc.consolidate()
        assert isinstance(result, ConsolidationResult)
        d = result.to_dict()
        assert "pruned" in d
        assert "total_before" in d
        assert "duration_ms" in d

    def test_deduplicate(self, tmp_db):
        mc = MemoryConsolidator(tmp_db)
        # Store exact duplicates (different sources to get different IDs)
        tmp_db.store("Duplicate content here", source="src1", importance=0.5)
        tmp_db.store("Duplicate content here", source="src2", importance=0.9)
        deduped = mc.deduplicate()
        assert deduped == 1

    def test_compact(self, tmp_db):
        mc = MemoryConsolidator(tmp_db)
        tmp_db.store("Some data for compaction")
        result = mc.compact()
        assert "rebuild" in result
        assert "stats" in result
        assert result["stats"]["index_health"]["status"] == "healthy"


# ── Context Synthesizer Tests ────────────────────────────────────────


class TestContextSynthesizer:
    def test_synthesize_empty(self, tmp_db):
        cs = ContextSynthesizer(tmp_db)
        result = cs.synthesize("random query with no matches")
        assert isinstance(result, SynthesizedContext)
        assert result.context == ""
        assert len(result.sources) == 0

    def test_synthesize_with_facts(self, tmp_db):
        fs = FactStore(tmp_db)
        fs.store_fact("knowledge", "capital", "Paris is the capital of France")
        cs = ContextSynthesizer(tmp_db, fact_store=fs)
        result = cs.synthesize(
            "capital of France",
            include_facts=True,
            fact_categories=["knowledge"],
        )
        assert result.fact_count >= 1
        assert "Paris" in result.context

    def test_synthesize_mixed_kinds(self, tmp_db):
        # Store different kinds
        tmp_db.store("Earth orbits the Sun", kind=MemoryKind.SEMANTIC, importance=0.9)
        tmp_db.store(
            "Observed solar eclipse yesterday",
            kind=MemoryKind.EPISODIC,
            importance=0.7,
        )
        tmp_db.store(
            "Procedure for telescope alignment",
            kind=MemoryKind.PROCEDURAL,
            importance=0.8,
        )
        cs = ContextSynthesizer(tmp_db)
        result = cs.synthesize("solar system observations")
        # Context should have organized sections
        assert isinstance(result.context, str)
        d = result.to_dict()
        assert "source_count" in d
        assert "total_score" in d

    def test_build_prompt_context_truncation(self, tmp_db):
        tmp_db.store("A" * 500, kind=MemoryKind.SEMANTIC, importance=0.9)
        cs = ContextSynthesizer(tmp_db)
        ctx = cs.build_prompt_context("test", max_chars=100)
        assert len(ctx) <= 100

    def test_synthesized_context_to_dict(self, tmp_db):
        sc = SynthesizedContext(
            context="test",
            sources=[],
            fact_count=2,
            episodic_count=1,
            procedural_count=0,
            total_score=0.85,
        )
        d = sc.to_dict()
        assert d["fact_count"] == 2
        assert d["total_score"] == 0.85
