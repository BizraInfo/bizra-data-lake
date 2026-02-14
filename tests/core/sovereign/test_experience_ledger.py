"""
Tests for the Sovereign Experience Ledger (SEL).

Covers: episode hashing, chain integrity, RIR retrieval,
distillation, and the SovereignExperienceLedger class.
"""

import math
import os
import tempfile
import time

import pytest

from core.sovereign.experience_ledger import (
    SELIntegrityError,
    Episode,
    EpisodeAction,
    EpisodeImpact,
    SovereignExperienceLedger,
    _compute_chain_hash,
    _compute_efficiency_score,
    _compute_episode_hash,
    _cosine_similarity,
    _integer_log2,
    _keyword_similarity,
)


# ═══════════════════════════════════════════════════════════════════════════════
# Episode Hash Determinism
# ═══════════════════════════════════════════════════════════════════════════════


class TestEpisodeHashing:
    """Test content-addressing and hash determinism."""

    def test_hash_determinism(self):
        """Same inputs must produce the same hash."""
        actions = [EpisodeAction("inference", "LLM call", True, 1000)]
        impact = EpisodeImpact(0.95, 0.96, True)

        h1 = _compute_episode_hash(0, 1000, "test", "abc123", 5, actions, impact)
        h2 = _compute_episode_hash(0, 1000, "test", "abc123", 5, actions, impact)

        assert h1 == h2

    def test_hash_changes_with_context(self):
        """Different context should produce different hash."""
        actions = [EpisodeAction("inference", "call", True, 1000)]
        impact = EpisodeImpact(0.95, 0.96, True)

        h1 = _compute_episode_hash(0, 1000, "query A", "g1", 3, actions, impact)
        h2 = _compute_episode_hash(0, 1000, "query B", "g1", 3, actions, impact)

        assert h1 != h2

    def test_hash_changes_with_snr(self):
        """Different SNR should produce different hash."""
        actions = [EpisodeAction("inference", "call", True, 1000)]
        i1 = EpisodeImpact(0.50, 0.96, False)
        i2 = EpisodeImpact(0.95, 0.96, True)

        h1 = _compute_episode_hash(0, 1000, "query", "g1", 3, actions, i1)
        h2 = _compute_episode_hash(0, 1000, "query", "g1", 3, actions, i2)

        assert h1 != h2

    def test_hash_is_hex_string(self):
        """Hash should be a hex string (SHA-256 or BLAKE3 = 64 hex chars)."""
        actions = [EpisodeAction("inference", "call", True, 1000)]
        impact = EpisodeImpact(0.90, 0.92, True)

        h = _compute_episode_hash(0, 1000, "test", "g1", 3, actions, impact)
        assert len(h) == 64  # SHA-256 hex
        assert all(c in "0123456789abcdef" for c in h)


# ═══════════════════════════════════════════════════════════════════════════════
# Chain Integrity
# ═══════════════════════════════════════════════════════════════════════════════


class TestChainIntegrity:
    """Test hash-chain tamper detection."""

    def test_chain_integrity(self):
        """Chain of 10 episodes should verify."""
        sel = SovereignExperienceLedger()
        for i in range(10):
            sel.commit(
                context=f"query {i}",
                graph_hash=f"g{i}",
                graph_node_count=3,
                actions=[("inference", "call", True, 1000)],
                snr_score=0.90,
                ihsan_score=0.92,
                snr_ok=True,
            )

        assert len(sel) == 10
        assert sel.verify_chain_integrity()

    def test_tamper_detection(self):
        """Modifying an episode should break chain verification."""
        sel = SovereignExperienceLedger()
        for i in range(5):
            sel.commit(
                context=f"query {i}",
                graph_hash=f"g{i}",
                graph_node_count=3,
                actions=[("inference", "call", True, 1000)],
                snr_score=0.90,
                ihsan_score=0.92,
                snr_ok=True,
            )

        assert sel.verify_chain_integrity()

        # Tamper with an episode
        sel._episodes[2].context = "TAMPERED"

        assert not sel.verify_chain_integrity()

    def test_empty_chain_is_valid(self):
        """Empty ledger should verify."""
        sel = SovereignExperienceLedger()
        assert sel.verify_chain_integrity()
        assert sel.chain_head == "genesis"

    def test_chain_head_changes(self):
        """Each commit should update the chain head."""
        sel = SovereignExperienceLedger()

        head1 = sel.chain_head
        sel.commit("q1", "g1", 3, [("inference", "call", True, 1000)], 0.90, 0.92, True)
        head2 = sel.chain_head
        sel.commit("q2", "g2", 3, [("inference", "call", True, 1000)], 0.90, 0.92, True)
        head3 = sel.chain_head

        assert head1 == "genesis"
        assert head2 != head1
        assert head3 != head2


# ═══════════════════════════════════════════════════════════════════════════════
# RIR Retrieval
# ═══════════════════════════════════════════════════════════════════════════════


class TestRIRRetrieval:
    """Test Recency-Importance-Relevance retrieval algorithm."""

    def test_retrieve_by_importance(self):
        """Higher importance episodes should rank first when recency=0."""
        sel = SovereignExperienceLedger(
            weight_recency=0.0, weight_importance=1.0, weight_relevance=0.0
        )

        sel.commit("low quality", "g1", 2, [("inference", "call", True, 1000)],
                    snr_score=0.50, ihsan_score=0.50, snr_ok=False)
        sel.commit("high quality", "g2", 5, [("inference", "call", True, 1000)],
                    snr_score=0.99, ihsan_score=0.98, snr_ok=True)

        results = sel.retrieve("anything", top_k=2)
        assert len(results) == 2
        assert results[0].context == "high quality"

    def test_retrieve_by_relevance_keywords(self):
        """Keyword overlap should drive relevance when other weights=0."""
        sel = SovereignExperienceLedger(
            weight_recency=0.0, weight_importance=0.0, weight_relevance=1.0
        )

        sel.commit("chocolate cake recipe baking", "g1", 2,
                    [("inference", "call", True, 1000)], 0.90, 0.90, True)
        sel.commit("neural network machine learning training", "g2", 3,
                    [("inference", "call", True, 1000)], 0.90, 0.90, True)

        results = sel.retrieve("neural network architecture", top_k=2)
        assert len(results) == 2
        assert results[0].context == "neural network machine learning training"

    def test_retrieve_empty_ledger(self):
        """Empty ledger returns empty results."""
        sel = SovereignExperienceLedger()
        results = sel.retrieve("anything", top_k=5)
        assert results == []

    def test_retrieve_top_k_limit(self):
        """Results should not exceed top_k."""
        sel = SovereignExperienceLedger()
        for i in range(20):
            sel.commit(f"query {i}", f"g{i}", 3,
                       [("inference", "call", True, 1000)], 0.90, 0.92, True)

        results = sel.retrieve("query", top_k=5)
        assert len(results) == 5

    def test_retrieve_with_cosine_similarity(self):
        """Cosine similarity should work with embeddings."""
        sel = SovereignExperienceLedger(
            weight_recency=0.0, weight_importance=0.0, weight_relevance=1.0
        )

        sel.commit("episode one", "g1", 2, [("inference", "call", True, 1000)],
                    0.90, 0.90, True, context_embedding=[1.0, 0.0, 0.0])
        sel.commit("episode two", "g2", 2, [("inference", "call", True, 1000)],
                    0.90, 0.90, True, context_embedding=[0.0, 1.0, 0.0])

        results = sel.retrieve("anything", top_k=2, query_embedding=[0.9, 0.1, 0.0])
        assert results[0].context == "episode one"


# ═══════════════════════════════════════════════════════════════════════════════
# Impact Measurement
# ═══════════════════════════════════════════════════════════════════════════════


class TestEpisodeImpact:
    """Test impact scoring."""

    def test_importance_calculation(self):
        impact = EpisodeImpact(0.95, 0.98, True)
        assert abs(impact.importance() - 0.95 * 0.98) < 1e-10

    def test_importance_zero_snr(self):
        impact = EpisodeImpact(0.0, 0.98, False)
        assert impact.importance() == 0.0


# ═══════════════════════════════════════════════════════════════════════════════
# Similarity Functions
# ═══════════════════════════════════════════════════════════════════════════════


class TestSimilarityFunctions:
    """Test cosine and keyword similarity."""

    def test_cosine_identical(self):
        sim = _cosine_similarity([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
        assert abs(sim - 1.0) < 1e-6

    def test_cosine_orthogonal(self):
        sim = _cosine_similarity([1.0, 0.0], [0.0, 1.0])
        assert abs(sim) < 1e-6

    def test_cosine_empty(self):
        assert _cosine_similarity([], []) == 0.0

    def test_cosine_different_lengths(self):
        assert _cosine_similarity([1.0, 2.0], [1.0, 2.0, 3.0]) == 0.0

    def test_keyword_overlap(self):
        sim = _keyword_similarity("neural network training", "neural network inference")
        assert sim > 0.0
        assert sim < 1.0

    def test_keyword_no_overlap(self):
        sim = _keyword_similarity("chocolate cake", "quantum physics")
        assert sim < 0.01


# ═══════════════════════════════════════════════════════════════════════════════
# Ledger Operations
# ═══════════════════════════════════════════════════════════════════════════════


class TestLedgerOperations:
    """Test commit, lookup, sequence tracking."""

    def test_sequence_increments(self):
        sel = SovereignExperienceLedger()
        assert sel.sequence == 0

        sel.commit("q1", "g1", 3, [("inference", "call", True, 1000)], 0.90, 0.92, True)
        assert sel.sequence == 1

        sel.commit("q2", "g2", 3, [("inference", "call", True, 1000)], 0.90, 0.92, True)
        assert sel.sequence == 2

    def test_get_by_hash(self):
        sel = SovereignExperienceLedger()
        h = sel.commit("findme", "g1", 3, [("inference", "call", True, 1000)], 0.95, 0.96, True)

        ep = sel.get_by_hash(h)
        assert ep is not None
        assert ep.context == "findme"

        assert sel.get_by_hash("nonexistent") is None

    def test_get_by_sequence(self):
        sel = SovereignExperienceLedger()
        sel.commit("q0", "g0", 3, [("inference", "call", True, 1000)], 0.90, 0.92, True)
        sel.commit("q1", "g1", 3, [("inference", "call", True, 1000)], 0.90, 0.92, True)

        ep = sel.get_by_sequence(1)
        assert ep is not None
        assert ep.context == "q1"

        assert sel.get_by_sequence(99) is None

    def test_episode_to_dict(self):
        sel = SovereignExperienceLedger()
        sel.commit("test query", "graph_hash", 5, [("inference", "call", True, 1000)],
                    0.95, 0.96, True, response_summary="test response")

        ep = sel.get_by_sequence(0)
        d = ep.to_dict()

        assert d["context"] == "test query"
        assert d["snr_score"] == 0.95
        assert d["ihsan_score"] == 0.96
        assert d["response_summary"] == "test response"
        assert "episode_hash" in d
        assert "chain_hash" in d

    def test_distillation_triggered(self):
        """Committing beyond max_episodes should trigger distillation."""
        sel = SovereignExperienceLedger(max_episodes=10)

        for i in range(11):
            snr = 0.50 if i < 5 else 0.95
            sel.commit(f"ep{i}", f"g{i}", 3,
                       [("inference", "call", True, 1000)], snr, 0.90, snr >= 0.85)

        assert sel.distillation_count > 0
        assert len(sel) <= 11

    def test_multiple_actions(self):
        """Episode with multiple actions should commit correctly."""
        sel = SovereignExperienceLedger()
        actions = [
            ("inference", "deepseek-r1 call", True, 54_600_000),
            ("snr_gate", "SNR=0.950", True, 100),
            ("guardian", "5-dimension eval", True, 500),
        ]
        h = sel.commit("complex query", "g1", 8, actions, 0.95, 0.96, True)

        ep = sel.get_by_hash(h)
        assert ep is not None
        assert len(ep.actions) == 3
        assert ep.actions[0].action_type == "inference"
        assert ep.actions[1].action_type == "snr_gate"
        assert ep.actions[2].action_type == "guardian"

    def test_episode_verify_hash(self):
        """Episode hash should verify after commit."""
        sel = SovereignExperienceLedger()
        sel.commit("test", "g1", 3, [("inference", "call", True, 1000)], 0.95, 0.96, True)

        ep = sel.get_by_sequence(0)
        assert ep.verify_hash()


# ═══════════════════════════════════════════════════════════════════════════════
# HashMap Index Tests (O(1) Lookups)
# ═══════════════════════════════════════════════════════════════════════════════


class TestHashMapIndex:
    """Test O(1) hash and sequence lookups via internal indexes."""

    def test_hash_index_100_episodes(self):
        """All 100 hashes should be retrievable via O(1) index."""
        sel = SovereignExperienceLedger()
        hashes = []
        for i in range(100):
            h = sel.commit(f"query {i}", f"g{i}", 3,
                           [("inference", "call", True, 1000)], 0.90, 0.92, True)
            hashes.append(h)

        for i, h in enumerate(hashes):
            ep = sel.get_by_hash(h)
            assert ep is not None, f"hash lookup failed for episode {i}"
            assert ep.sequence == i

    def test_seq_index_50_episodes(self):
        """All 50 sequences should be retrievable via O(1) index."""
        sel = SovereignExperienceLedger()
        for i in range(50):
            sel.commit(f"query {i}", f"g{i}", 3,
                       [("inference", "call", True, 1000)], 0.90, 0.92, True)

        for seq in range(50):
            ep = sel.get_by_sequence(seq)
            assert ep is not None, f"seq lookup failed for {seq}"
            assert ep.context == f"query {seq}"
        assert sel.get_by_sequence(999) is None

    def test_index_survives_distillation(self):
        """Indexes should be valid after distillation removes episodes."""
        sel = SovereignExperienceLedger(max_episodes=10)

        for i in range(11):
            snr = 0.50 if i < 5 else 0.95
            sel.commit(f"ep{i}", f"g{i}", 3,
                       [("inference", "call", True, 1000)], snr, 0.90, snr >= 0.85)

        assert sel.distillation_count > 0

        # All surviving episodes should be findable
        for ep in sel._episodes:
            assert sel.get_by_hash(ep.episode_hash) is not None
            assert sel.get_by_sequence(ep.sequence) is not None

    def test_index_consistent_after_multiple_distillations(self):
        """Indexes should remain consistent across multiple distillation cycles."""
        sel = SovereignExperienceLedger(max_episodes=5)

        for i in range(20):
            snr = 0.50 + (i % 10) * 0.05
            sel.commit(f"ep{i}", f"g{i}", 3,
                       [("inference", "call", True, 1000)], snr, 0.90, snr >= 0.85)

        assert sel.distillation_count > 1

        for ep in sel._episodes:
            found = sel.get_by_hash(ep.episode_hash)
            assert found is not None
            assert found.sequence == ep.sequence

    def test_nonexistent_hash_returns_none(self):
        """Nonexistent hash should return None."""
        sel = SovereignExperienceLedger()
        sel.commit("q1", "g1", 3, [("inference", "call", True, 1000)], 0.90, 0.92, True)
        assert sel.get_by_hash("0" * 64) is None

    def test_nonexistent_sequence_returns_none(self):
        """Nonexistent sequence should return None."""
        sel = SovereignExperienceLedger()
        sel.commit("q1", "g1", 3, [("inference", "call", True, 1000)], 0.90, 0.92, True)
        assert sel.get_by_sequence(42) is None


# ═══════════════════════════════════════════════════════════════════════════════
# Efficiency_k Tests (Deterministic Integer Log2)
# ═══════════════════════════════════════════════════════════════════════════════


class TestIntegerLog2:
    """Test the integer log2 function."""

    def test_edge_cases(self):
        assert _integer_log2(0) == 0
        assert _integer_log2(1) == 0

    def test_powers_of_two(self):
        assert _integer_log2(2) == 1
        assert _integer_log2(4) == 2
        assert _integer_log2(8) == 3
        assert _integer_log2(16) == 4
        assert _integer_log2(1024) == 10

    def test_non_powers(self):
        assert _integer_log2(3) == 1  # floor(log2(3)) = 1
        assert _integer_log2(7) == 2  # floor(log2(7)) = 2
        assert _integer_log2(1023) == 9  # floor(log2(1023)) = 9

    def test_large_values(self):
        assert _integer_log2(1_000_000) == 19
        assert _integer_log2(1_000_000_000) == 29


class TestEfficiencyScore:
    """Test the Efficiency_k computation."""

    def test_deterministic(self):
        """Same inputs must produce same output."""
        e1 = _compute_efficiency_score(0.95, 0.98, 1000)
        e2 = _compute_efficiency_score(0.95, 0.98, 1000)
        assert e1 == e2

    def test_decreases_with_tokens(self):
        """More tokens should lower efficiency (cost penalty)."""
        e_small = _compute_efficiency_score(0.95, 0.98, 100)
        e_large = _compute_efficiency_score(0.95, 0.98, 1_000_000)
        assert e_small > e_large

    def test_zero_snr_gives_zero(self):
        """Zero SNR should give zero efficiency."""
        e = _compute_efficiency_score(0.0, 0.98, 1000)
        assert e == 0.0

    def test_positive_for_good_scores(self):
        """High quality + tokens should give positive efficiency."""
        e = _compute_efficiency_score(0.95, 0.98, 1000)
        assert e > 0.0

    def test_formula_manual_check(self):
        """Verify formula: (quantize(0.95) * quantize(0.98)) // max(1, log2(1002))."""
        snr_q = int(0.95 * 1_000_000)   # 950000
        ihsan_q = int(0.98 * 1_000_000)  # 980000
        numerator = snr_q * ihsan_q       # 931000000000
        log_val = max(1, _integer_log2(1000 + 2))  # log2(1002) = 9
        expected_fp = numerator // log_val
        expected = expected_fp / (1_000_000 * 1_000_000)

        actual = _compute_efficiency_score(0.95, 0.98, 1000)
        assert actual == expected


class TestImportanceWithEfficiency:
    """Test the updated importance() method with Efficiency_k."""

    def test_backward_compatible_no_tokens(self):
        """Without tokens_used, importance = SNR * Ihsan (backward compatible)."""
        impact = EpisodeImpact(0.95, 0.98, True)
        assert abs(impact.importance() - 0.95 * 0.98) < 1e-10

    def test_importance_with_efficiency(self):
        """With tokens, importance = SNR * Ihsan * Efficiency."""
        impact = EpisodeImpact(0.95, 0.98, True, tokens_used=1000,
                                efficiency_score=_compute_efficiency_score(0.95, 0.98, 1000))
        base = 0.95 * 0.98
        assert impact.importance() < base  # Efficiency < 1.0
        assert impact.importance() > 0.0

    def test_commit_with_tokens_used(self):
        """Committing with tokens_used should set efficiency_score."""
        sel = SovereignExperienceLedger()
        h = sel.commit("test", "g1", 3, [("inference", "call", True, 1000)],
                        0.95, 0.96, True, tokens_used=5000)

        ep = sel.get_by_hash(h)
        assert ep is not None
        assert ep.impact.tokens_used == 5000
        assert ep.impact.efficiency_score > 0.0

    def test_commit_without_tokens_used_backward_compat(self):
        """Committing without tokens_used should have zero efficiency."""
        sel = SovereignExperienceLedger()
        h = sel.commit("test", "g1", 3, [("inference", "call", True, 1000)],
                        0.95, 0.96, True)

        ep = sel.get_by_hash(h)
        assert ep is not None
        assert ep.impact.tokens_used == 0
        assert ep.impact.efficiency_score == 0.0

    def test_episode_hash_includes_efficiency(self):
        """Different efficiency scores should produce different hashes."""
        sel = SovereignExperienceLedger()
        h1 = sel.commit("test", "g1", 3, [("inference", "call", True, 1000)],
                         0.95, 0.96, True, tokens_used=100)
        h2 = sel.commit("test", "g1", 3, [("inference", "call", True, 1000)],
                         0.95, 0.96, True, tokens_used=100000)
        # Different tokens_used -> different efficiency -> different hash
        # (Note: timestamps also differ, so hashes differ for that reason too)
        assert h1 != h2

    def test_chain_integrity_with_efficiency(self):
        """Chain should verify with efficiency scores present."""
        sel = SovereignExperienceLedger()
        for i in range(5):
            sel.commit(f"q{i}", f"g{i}", 3,
                       [("inference", "call", True, 1000)],
                       0.90, 0.92, True, tokens_used=1000 * (i + 1))
        assert sel.verify_chain_integrity()

    def test_to_dict_includes_efficiency(self):
        """to_dict should include efficiency when tokens_used > 0."""
        sel = SovereignExperienceLedger()
        sel.commit("test", "g1", 3, [("inference", "call", True, 1000)],
                    0.95, 0.96, True, tokens_used=5000)

        ep = sel.get_by_sequence(0)
        d = ep.to_dict()
        assert "tokens_used" in d
        assert "efficiency_score" in d
        assert d["tokens_used"] == 5000
        assert d["efficiency_score"] > 0.0

    def test_to_dict_excludes_efficiency_when_no_tokens(self):
        """to_dict should not include efficiency when tokens_used = 0."""
        sel = SovereignExperienceLedger()
        sel.commit("test", "g1", 3, [("inference", "call", True, 1000)],
                    0.95, 0.96, True)

        ep = sel.get_by_sequence(0)
        d = ep.to_dict()
        assert "tokens_used" not in d
        assert "efficiency_score" not in d


# ═══════════════════════════════════════════════════════════════════════════════
# Serialization Tests (JSONL Export/Import + Chain Verification on Load)
# ═══════════════════════════════════════════════════════════════════════════════


class TestSELSerialization:
    """Test JSONL export/import with chain verification on load."""

    def _make_populated_sel(self, n: int = 5) -> SovereignExperienceLedger:
        sel = SovereignExperienceLedger()
        for i in range(n):
            sel.commit(
                context=f"query {i}",
                graph_hash=f"g{i}",
                graph_node_count=3 + i,
                actions=[("inference", f"call {i}", True, 1000 * (i + 1))],
                snr_score=0.85 + i * 0.03,
                ihsan_score=0.90 + i * 0.02,
                snr_ok=True,
                tokens_used=500 * (i + 1),
            )
        return sel

    def test_export_creates_file(self):
        """export_jsonl should create a file with correct line count."""
        sel = self._make_populated_sel(5)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            count = sel.export_jsonl(path)
            assert count == 5
            with open(path, "r") as f:
                lines = [l for l in f if l.strip()]
            assert len(lines) == 5
        finally:
            os.unlink(path)

    def test_roundtrip_preserves_chain(self):
        """Export + import should produce identical chain state."""
        sel = self._make_populated_sel(10)
        original_head = sel.chain_head
        original_len = len(sel)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            sel.export_jsonl(path)
            loaded = SovereignExperienceLedger.import_jsonl(path)

            assert len(loaded) == original_len
            assert loaded.chain_head == original_head
            assert loaded.verify_chain_integrity()
        finally:
            os.unlink(path)

    def test_roundtrip_preserves_episodes(self):
        """Imported episodes should match originals."""
        sel = self._make_populated_sel(3)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            sel.export_jsonl(path)
            loaded = SovereignExperienceLedger.import_jsonl(path)

            for i in range(3):
                orig = sel.get_by_sequence(i)
                imp = loaded.get_by_sequence(i)
                assert imp is not None
                assert imp.context == orig.context
                assert imp.episode_hash == orig.episode_hash
                assert imp.chain_hash == orig.chain_hash
                assert imp.impact.tokens_used == orig.impact.tokens_used
                assert imp.impact.efficiency_score == orig.impact.efficiency_score
        finally:
            os.unlink(path)

    def test_import_detects_tampered_file(self):
        """Importing a tampered JSONL should raise SELIntegrityError."""
        sel = self._make_populated_sel(5)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            sel.export_jsonl(path)

            # Tamper with the 3rd line
            with open(path, "r") as f:
                lines = f.readlines()
            import json as _json
            record = _json.loads(lines[2])
            record["context"] = "TAMPERED"
            lines[2] = _json.dumps(record) + "\n"
            with open(path, "w") as f:
                f.writelines(lines)

            with pytest.raises(SELIntegrityError):
                SovereignExperienceLedger.import_jsonl(path, verify=True)
        finally:
            os.unlink(path)

    def test_import_skip_verification(self):
        """verify=False should skip chain check even on tampered data."""
        sel = self._make_populated_sel(3)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            sel.export_jsonl(path)

            # Tamper
            with open(path, "r") as f:
                lines = f.readlines()
            import json as _json
            record = _json.loads(lines[1])
            record["context"] = "TAMPERED"
            lines[1] = _json.dumps(record) + "\n"
            with open(path, "w") as f:
                f.writelines(lines)

            # Should not raise
            loaded = SovereignExperienceLedger.import_jsonl(path, verify=False)
            assert len(loaded) == 3
        finally:
            os.unlink(path)

    def test_import_empty_file(self):
        """Importing an empty file should return empty SEL."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            loaded = SovereignExperienceLedger.import_jsonl(path)
            assert len(loaded) == 0
            assert loaded.chain_head == "genesis"
            assert loaded.verify_chain_integrity()
        finally:
            os.unlink(path)

    def test_indexes_populated_on_import(self):
        """Hash and sequence indexes should work after import."""
        sel = self._make_populated_sel(5)
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            sel.export_jsonl(path)
            loaded = SovereignExperienceLedger.import_jsonl(path)

            for ep in loaded._episodes:
                assert loaded.get_by_hash(ep.episode_hash) is not None
                assert loaded.get_by_sequence(ep.sequence) is not None
        finally:
            os.unlink(path)

    def test_roundtrip_with_embeddings(self):
        """Episodes with context embeddings should survive roundtrip."""
        sel = SovereignExperienceLedger()
        sel.commit("q1", "g1", 3, [("inference", "call", True, 1000)],
                    0.95, 0.96, True, context_embedding=[0.1, 0.2, 0.3])
        sel.commit("q2", "g2", 3, [("inference", "call", True, 1000)],
                    0.90, 0.92, True)

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            path = f.name
        try:
            sel.export_jsonl(path)
            loaded = SovereignExperienceLedger.import_jsonl(path)

            ep0 = loaded.get_by_sequence(0)
            assert ep0.context_embedding == [0.1, 0.2, 0.3]
            ep1 = loaded.get_by_sequence(1)
            assert ep1.context_embedding is None
        finally:
            os.unlink(path)
