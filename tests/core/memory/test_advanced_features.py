"""Tests for AgentDB advanced features: MMR, distance metrics, metadata filters."""

from core.memory.config import HNSWConfig
from core.memory.hnsw_index import HNSWIndex
from core.memory.hybrid_query import (
    _match_metadata_filters,
    _mmr_rerank,
)
from core.memory.types import (
    MemoryRecord,
    QueryOptions,
    SearchResult,
)

# ── Metadata Filter Tests ──────────────────────────────────────────────


class TestMetadataFilters:
    """Tests for _match_metadata_filters."""

    def test_exact_match_pass(self):
        assert _match_metadata_filters({"color": "red"}, {"color": "red"})

    def test_exact_match_fail(self):
        assert not _match_metadata_filters({"color": "red"}, {"color": "blue"})

    def test_missing_key_fails(self):
        assert not _match_metadata_filters({}, {"color": "red"})

    def test_gte_pass(self):
        assert _match_metadata_filters({"year": 2025}, {"year": {"$gte": 2023}})

    def test_gte_fail(self):
        assert not _match_metadata_filters({"year": 2020}, {"year": {"$gte": 2023}})

    def test_lte_pass(self):
        assert _match_metadata_filters({"price": 50}, {"price": {"$lte": 100}})

    def test_lte_fail(self):
        assert not _match_metadata_filters({"price": 150}, {"price": {"$lte": 100}})

    def test_range_combined(self):
        assert _match_metadata_filters(
            {"score": 0.95}, {"score": {"$gte": 0.90, "$lte": 1.0}}
        )

    def test_in_pass(self):
        assert _match_metadata_filters(
            {"category": "ml"}, {"category": {"$in": ["ml", "nlp"]}}
        )

    def test_in_fail(self):
        assert not _match_metadata_filters(
            {"category": "cv"}, {"category": {"$in": ["ml", "nlp"]}}
        )

    def test_contains_list(self):
        assert _match_metadata_filters(
            {"tags": ["python", "rust"]}, {"tags": {"$contains": "python"}}
        )

    def test_contains_string(self):
        assert _match_metadata_filters(
            {"desc": "hello world"}, {"desc": {"$contains": "world"}}
        )

    def test_contains_fail(self):
        assert not _match_metadata_filters(
            {"tags": ["python"]}, {"tags": {"$contains": "rust"}}
        )

    def test_multi_filter_all_pass(self):
        assert _match_metadata_filters(
            {"year": 2025, "lang": "python", "score": 0.95},
            {"year": {"$gte": 2020}, "lang": "python"},
        )

    def test_multi_filter_one_fails(self):
        assert not _match_metadata_filters(
            {"year": 2025, "lang": "rust"},
            {"year": {"$gte": 2020}, "lang": "python"},
        )


# ── MMR Re-ranking Tests ──────────────────────────────────────────────


def _make_result(rid: str, embedding: list, score: float = 0.8) -> SearchResult:
    """Helper: create a SearchResult with embedding."""
    return SearchResult(
        record=MemoryRecord(
            id=rid,
            content=f"Record {rid}",
            embedding=embedding,
        ),
        score=score,
        vector_score=score,
    )


class TestMMR:
    """Tests for Maximal Marginal Relevance re-ranking."""

    def test_mmr_empty(self):
        result = _mmr_rerank([], query_embedding=[1, 0], lam=0.5, top_k=5)
        assert result == []

    def test_mmr_single(self):
        r = _make_result("a", [1.0, 0.0])
        result = _mmr_rerank([r], query_embedding=[1, 0], lam=0.5, top_k=5)
        assert len(result) == 1
        assert result[0].record.id == "a"

    def test_mmr_no_query(self):
        """Without query embedding, MMR returns results as-is."""
        r = _make_result("a", [1.0, 0.0])
        result = _mmr_rerank([r], query_embedding=None, lam=0.5, top_k=5)
        assert len(result) == 1

    def test_mmr_diversifies(self):
        """Similar vectors should be de-prioritized in favor of diverse ones."""
        # Query is [1, 0]
        # Two near-identical results + one diverse
        r1 = _make_result("similar1", [0.99, 0.01], score=0.9)
        r2 = _make_result("similar2", [0.98, 0.02], score=0.89)
        r3 = _make_result("diverse", [0.0, 1.0], score=0.7)

        # With high diversity (lambda=0.2), diverse should rank higher
        result = _mmr_rerank([r1, r2, r3], query_embedding=[1.0, 0.0], lam=0.2, top_k=3)
        ids = [r.record.id for r in result]
        # First pick is most relevant, second should favor diversity
        assert ids[0] == "similar1"
        assert "diverse" in ids[:2]  # Diverse should be picked early

    def test_mmr_lambda_1_preserves_order(self):
        """Lambda=1.0 (pure relevance) preserves original ranking."""
        r1 = _make_result("a", [1.0, 0.0], score=0.9)
        r2 = _make_result("b", [0.5, 0.5], score=0.7)
        result = _mmr_rerank([r1, r2], query_embedding=[1.0, 0.0], lam=1.0, top_k=2)
        assert result[0].record.id == "a"

    def test_mmr_respects_top_k(self):
        results = [_make_result(f"r{i}", [float(i), 0.0]) for i in range(10)]
        result = _mmr_rerank(results, query_embedding=[1.0, 0.0], lam=0.5, top_k=3)
        assert len(result) == 3

    def test_mmr_skips_no_embedding(self):
        """Records without embeddings don't crash MMR; only embeddable ones are re-ranked."""
        r_with = _make_result("a", [1.0, 0.0])
        r_without = SearchResult(
            record=MemoryRecord(id="b", content="no vec", embedding=None),
            score=0.5,
        )
        # MMR filters to embeddable candidates only for diversity calc
        result = _mmr_rerank(
            [r_with, r_without], query_embedding=[1.0, 0.0], lam=0.5, top_k=5
        )
        # Only the embeddable record survives MMR re-ranking
        embeddable = [r for r in result if r.record.embedding is not None]
        assert len(embeddable) >= 1
        assert embeddable[0].record.id == "a"


# ── Distance Metric Tests ─────────────────────────────────────────────


class TestDistanceMetrics:
    """Tests for multi-metric numpy fallback in HNSWIndex."""

    def _make_index(self, space: str) -> HNSWIndex:
        config = HNSWConfig(dimensions=3, space=space, max_elements=100)
        idx = HNSWIndex(config)
        # Force numpy fallback
        idx._use_hnswlib = False
        idx._initialized = True
        return idx

    def test_cosine_finds_nearest(self):
        idx = self._make_index("cosine")
        idx.add("a", [1.0, 0.0, 0.0])
        idx.add("b", [0.0, 1.0, 0.0])
        results = idx.search([0.9, 0.1, 0.0], top_k=2)
        assert results[0][0] == "a"  # Closest to query

    def test_l2_finds_nearest(self):
        idx = self._make_index("l2")
        idx.add("close", [1.0, 0.0, 0.0])
        idx.add("far", [10.0, 10.0, 10.0])
        results = idx.search([1.1, 0.0, 0.0], top_k=2)
        assert results[0][0] == "close"

    def test_ip_finds_highest_dot(self):
        idx = self._make_index("ip")
        idx.add("high", [1.0, 1.0, 1.0])
        idx.add("low", [0.1, 0.1, 0.1])
        results = idx.search([1.0, 1.0, 1.0], top_k=2)
        assert results[0][0] == "high"  # Highest inner product

    def test_l2_distance_values(self):
        idx = self._make_index("l2")
        idx.add("origin", [0.0, 0.0, 0.0])
        results = idx.search([3.0, 4.0, 0.0], top_k=1)
        # L2 distance squared = 9 + 16 = 25
        assert abs(results[0][1] - 25.0) < 0.01


# ── QueryOptions Integration Tests ────────────────────────────────────


class TestQueryOptionsAdvanced:
    """Test that new QueryOptions fields have correct defaults."""

    def test_default_mmr_off(self):
        opts = QueryOptions()
        assert opts.use_mmr is False
        assert opts.mmr_lambda == 0.5

    def test_default_metadata_filters_none(self):
        opts = QueryOptions()
        assert opts.metadata_filters is None

    def test_mmr_enabled(self):
        opts = QueryOptions(use_mmr=True, mmr_lambda=0.3)
        assert opts.use_mmr is True
        assert opts.mmr_lambda == 0.3

    def test_metadata_filters_set(self):
        opts = QueryOptions(metadata_filters={"year": {"$gte": 2024}})
        assert opts.metadata_filters == {"year": {"$gte": 2024}}
