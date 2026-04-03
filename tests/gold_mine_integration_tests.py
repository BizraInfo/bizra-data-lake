"""
tests/gold_mine_integration_tests.py
=====================================

Integration tests for Gold Mine graph connector and multi-hop reasoning.

Tests:
1. Gold Mine connector initialization
2. Entity-based queries
3. Multi-hop graph traversal
4. Hop orchestrator planning
5. Hop orchestrator execution
6. Evidence receipt generation

Run with:
    pytest tests/gold_mine_integration_tests.py -v
"""

import asyncio
import pytest
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================================
# FIXTURES
# ============================================================================

@pytest.fixture
def gold_mine_connector():
    """Create a Gold Mine connector instance."""
    from bizra_kernel.gold_mine_connector import GoldMineConnector
    return GoldMineConnector()


@pytest.fixture
def hop_orchestrator():
    """Create a Hop orchestrator instance without dependencies."""
    from bizra_kernel.hop_orchestrator import HopOrchestrator
    return HopOrchestrator(
        session_id="test-session",
        gold_mine=None,
        got_orchestrator=None,
        validation_threshold=0.80,  # Lower threshold for testing
    )


# ============================================================================
# GOLD MINE CONNECTOR TESTS
# ============================================================================

class TestGoldMineConnector:
    """Tests for GoldMineConnector."""

    def test_connector_creation(self, gold_mine_connector):
        """Test connector can be instantiated."""
        assert gold_mine_connector is not None
        assert not gold_mine_connector._initialized

    def test_is_available_checks_path(self, gold_mine_connector):
        """Test is_available checks for graph data."""
        # This will be True only if Gold Mine data exists
        available = gold_mine_connector.is_available()
        assert isinstance(available, bool)

    @pytest.mark.asyncio
    async def test_initialization(self, gold_mine_connector):
        """Test connector initialization."""
        if not gold_mine_connector.is_available():
            pytest.skip("Gold Mine data not available")

        await gold_mine_connector.initialize()
        assert gold_mine_connector._initialized

        stats = gold_mine_connector.get_statistics()
        assert stats["nodes"] > 0
        assert stats["edges"] > 0
        assert stats["initialized"]

    @pytest.mark.asyncio
    async def test_query_by_entity(self, gold_mine_connector):
        """Test entity-based query."""
        if not gold_mine_connector.is_available():
            pytest.skip("Gold Mine data not available")

        await gold_mine_connector.initialize()

        # Query for BIZRA entity
        nodes = gold_mine_connector.query_by_entity("bizra", limit=10)

        # Should find at least some nodes
        assert isinstance(nodes, list)
        # Note: might be empty if no BIZRA nodes exist
        for node in nodes:
            assert hasattr(node, 'id')
            assert hasattr(node, 'label')
            assert hasattr(node, 'kind')

    @pytest.mark.asyncio
    async def test_query_by_entity_with_kind_filter(self, gold_mine_connector):
        """Test entity query with kind filter."""
        if not gold_mine_connector.is_available():
            pytest.skip("Gold Mine data not available")

        await gold_mine_connector.initialize()

        # Query for documents only
        nodes = gold_mine_connector.query_by_entity("pdf", limit=5, kinds=["Document"])

        for node in nodes:
            assert node.kind == "Document"

    @pytest.mark.asyncio
    async def test_multi_hop_expand(self, gold_mine_connector):
        """Test multi-hop graph traversal."""
        if not gold_mine_connector.is_available():
            pytest.skip("Gold Mine data not available")

        await gold_mine_connector.initialize()

        # Find seed nodes
        seed_nodes = gold_mine_connector.query_by_entity("gpu", limit=2)
        if not seed_nodes:
            pytest.skip("No seed nodes found")

        seed_ids = [n.id for n in seed_nodes]

        # Expand
        result = gold_mine_connector.multi_hop_expand(
            seed_ids=seed_ids,
            max_hops=2,
            max_nodes_per_hop=10,
        )

        assert result.seed_ids == seed_ids
        assert len(result.reached_nodes) >= len(seed_ids)  # At least seeds
        assert isinstance(result.paths, list)

    @pytest.mark.asyncio
    async def test_get_neighbors(self, gold_mine_connector):
        """Test get neighbors functionality."""
        if not gold_mine_connector.is_available():
            pytest.skip("Gold Mine data not available")

        await gold_mine_connector.initialize()

        # Find a node with edges
        nodes = gold_mine_connector.query_by_entity("gpu", limit=1)
        if not nodes:
            pytest.skip("No nodes found")

        neighbors = gold_mine_connector.get_neighbors(nodes[0].id)
        assert isinstance(neighbors, list)

    def test_get_statistics_before_init(self, gold_mine_connector):
        """Test statistics before initialization."""
        stats = gold_mine_connector.get_statistics()

        assert not stats["initialized"]
        assert stats["nodes"] == 0
        assert stats["edges"] == 0

    def test_receipt_generation(self, gold_mine_connector):
        """Test receipt generation."""
        receipt = gold_mine_connector.generate_receipt(
            operation="test_query",
            query="BIZRA",
            limit=10,
        )

        assert "receipt_id" in receipt
        assert "timestamp" in receipt
        assert receipt["operation"] == "test_query"
        assert receipt["params"]["query"] == "BIZRA"


# ============================================================================
# HOP ORCHESTRATOR TESTS
# ============================================================================

class TestHopOrchestrator:
    """Tests for HopOrchestrator."""

    def test_orchestrator_creation(self, hop_orchestrator):
        """Test orchestrator can be instantiated."""
        assert hop_orchestrator is not None
        assert hop_orchestrator.session_id == "test-session"
        assert hop_orchestrator.validation_threshold == 0.80

    @pytest.mark.asyncio
    async def test_plan_hops_basic(self, hop_orchestrator):
        """Test basic hop planning."""
        plan = await hop_orchestrator.plan_hops(
            query="How does BIZRA SAPE work?",
            max_hops=3,
        )

        assert plan.original_query == "How does BIZRA SAPE work?"
        assert len(plan.planned_hops) <= 3
        assert plan.max_hops == 3
        assert "BIZRA" in plan.entities_detected or "SAPE" in plan.entities_detected

    @pytest.mark.asyncio
    async def test_plan_hops_entity_extraction(self, hop_orchestrator):
        """Test entity extraction during planning."""
        plan = await hop_orchestrator.plan_hops(
            query="Explain Ihsan threshold validation in FATE",
            max_hops=2,
        )

        # Should detect BIZRA-specific entities
        entities_lower = [e.lower() for e in plan.entities_detected]
        assert any(term in entities_lower for term in ["ihsan", "fate"])

    @pytest.mark.asyncio
    async def test_execute_single_hop(self, hop_orchestrator):
        """Test single hop execution."""
        hop = await hop_orchestrator.execute_hop(
            hop_query="What is SAPE?",
            hop_number=1,
            context="",
        )

        assert hop.hop_number == 1
        assert hop.query == "What is SAPE?"
        assert hop.ihsan_score > 0
        assert hop.latency_ms > 0
        # With mock validator, should pass
        from bizra_kernel.hop_orchestrator import HopStatus
        assert hop.status == HopStatus.VALIDATED

    @pytest.mark.asyncio
    async def test_execute_hop_with_context(self, hop_orchestrator):
        """Test hop execution with prior context."""
        hop = await hop_orchestrator.execute_hop(
            hop_query="How does it relate to validation?",
            hop_number=2,
            context="SAPE is a Symbolic-Abstraction Probe Elevation system.",
        )

        assert hop.hop_number == 2
        assert "SAPE" in hop.accumulated_context

    @pytest.mark.asyncio
    async def test_execute_plan_full(self, hop_orchestrator):
        """Test full plan execution."""
        plan = await hop_orchestrator.plan_hops(
            query="What is BIZRA?",
            max_hops=2,
        )

        hops, answer = await hop_orchestrator.execute_plan(plan)

        assert len(hops) == len(plan.planned_hops)
        assert isinstance(answer, str)
        assert len(answer) > 0

    @pytest.mark.asyncio
    async def test_validation_failure(self):
        """Test hop validation failure handling."""
        from bizra_kernel.hop_orchestrator import (
            HopOrchestrator,
            HopValidationError,
            HopStatus,
        )

        # Create orchestrator with high threshold
        orch = HopOrchestrator(
            session_id="test-fail",
            gold_mine=None,
            validation_threshold=0.99,  # Very high, will fail
        )

        # Mock a validator that returns low scores
        async def low_score_validator(content):
            return {"groundedness": 0.5, "safety": 0.5}, 0.5

        orch.sape_validator = low_score_validator

        with pytest.raises(HopValidationError) as exc_info:
            await orch.execute_hop("Test query", 1, "")

        assert exc_info.value.hop_number == 1
        assert exc_info.value.ihsan_score == 0.5
        assert exc_info.value.threshold == 0.99

    @pytest.mark.asyncio
    async def test_evidence_receipt(self, hop_orchestrator):
        """Test evidence receipt generation."""
        plan = await hop_orchestrator.plan_hops("Test query", max_hops=1)
        await hop_orchestrator.execute_plan(plan)

        receipt = hop_orchestrator.get_evidence_receipt()

        assert "receipt_id" in receipt
        assert "timestamp" in receipt
        assert receipt["session_id"] == "test-session"
        assert "hops" in receipt
        assert "validated_hops" in receipt
        assert "total_latency_ms" in receipt


# ============================================================================
# INTEGRATION TESTS (Gold Mine + Hop Orchestrator)
# ============================================================================

class TestGoldMineHopIntegration:
    """Integration tests combining Gold Mine and Hop Orchestrator."""

    @pytest.mark.asyncio
    async def test_full_integration(self):
        """Test full integration with Gold Mine and Hop Orchestrator."""
        from bizra_kernel.gold_mine_connector import GoldMineConnector
        from bizra_kernel.hop_orchestrator import HopOrchestrator

        connector = GoldMineConnector()

        if not connector.is_available():
            pytest.skip("Gold Mine data not available")

        await connector.initialize()

        orch = HopOrchestrator(
            session_id="integration-test",
            gold_mine=connector,
            validation_threshold=0.80,
        )

        plan = await orch.plan_hops(
            query="What is BIZRA architecture?",
            max_hops=2,
        )

        hops, answer = await orch.execute_plan(plan)

        # Should complete at least one hop
        assert len(hops) > 0

        # Check hops have retrieved nodes
        total_retrieved = sum(len(h.retrieved_nodes) for h in hops)
        # Might be 0 if no matching entities exist
        print(f"Total retrieved nodes: {total_retrieved}")

        receipt = orch.get_evidence_receipt()
        assert receipt["gold_mine_available"]

    @pytest.mark.asyncio
    async def test_retrieval_backend_gold_mine(self):
        """Test Gold Mine integration with retrieval backend."""
        from bizra_kernel.retrieval_backend import (
            RetrievalBackend,
            RetrievalSource,
        )

        backend = RetrievalBackend()

        # Check Gold Mine is available in sources
        assert hasattr(backend, 'gold_mine')

        # Query with Gold Mine source
        if backend.gold_mine.is_available():
            results = await backend.retrieve(
                query="BIZRA architecture",
                top_k=5,
                sources=[RetrievalSource.GOLD_MINE],
            )

            assert RetrievalSource.GOLD_MINE in results.sources_queried
        else:
            pytest.skip("Gold Mine not available")


# ============================================================================
# SAPE EVIDENCE TESTS
# ============================================================================

class TestSapeEvidence:
    """Tests for SAPE Gold Mine evidence integration."""

    @pytest.mark.asyncio
    async def test_retrieve_gold_mine_evidence(self):
        """Test retrieving Gold Mine evidence for SAPE."""
        from core.sape import retrieve_gold_mine_evidence

        evidence = await retrieve_gold_mine_evidence(
            topics=["BIZRA", "SAPE"],
            limit=5,
        )

        # May be empty if Gold Mine not available
        assert isinstance(evidence, list)

        for item in evidence:
            assert "node_id" in item
            assert "label" in item
            assert "kind" in item
            assert "relevance" in item

    def test_format_gold_mine_evidence(self):
        """Test formatting Gold Mine evidence."""
        from core.sape import format_gold_mine_evidence

        evidence = [
            {
                "node_id": "test1",
                "label": "Test Document",
                "kind": "Document",
                "source": "/path/to/file.pdf",
                "relevance": 0.9,
                "hop_distance": 0,
            },
            {
                "node_id": "test2",
                "label": "Test Entity",
                "kind": "Entity",
                "source": None,
                "relevance": 0.7,
                "hop_distance": 1,
            },
        ]

        formatted = format_gold_mine_evidence(evidence)

        assert "[Gold Mine Knowledge Graph Evidence]" in formatted
        assert "[Document] Test Document" in formatted
        assert "[Entity] Test Entity" in formatted
        assert "hop=1" in formatted

    def test_format_empty_evidence(self):
        """Test formatting empty evidence."""
        from core.sape import format_gold_mine_evidence

        formatted = format_gold_mine_evidence([])
        assert "no Gold Mine evidence" in formatted


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
