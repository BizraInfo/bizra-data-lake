"""
Tests for Knowledge Integration — BIZRA Data Lake + MoMo R&D
============================================================
Validates the knowledge integrator and swarm knowledge bridge.
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from core.sovereign.knowledge_integrator import (
    KnowledgeIntegrator,
    KnowledgeQuery,
    KnowledgeResult,
    KnowledgeSource,
    create_knowledge_integrator,
)
from core.sovereign.swarm_knowledge_bridge import (
    AgentKnowledgeContext,
    KnowledgeInjection,
    ROLE_KNOWLEDGE_ACCESS,
    SwarmKnowledgeBridge,
    create_swarm_knowledge_bridge,
)
from core.sovereign.team_planner import AgentRole


# =============================================================================
# KNOWLEDGE INTEGRATOR TESTS
# =============================================================================

class TestKnowledgeIntegrator:
    """Tests for KnowledgeIntegrator."""

    def test_knowledge_source_dataclass(self):
        """Test KnowledgeSource dataclass."""
        source = KnowledgeSource(
            name="Test Source",
            path="test/path.json",
            source_type="json",
            category="test",
            snr_score=0.9,
            ihsan_score=0.95,
            priority="HIGH",
        )
        assert source.name == "Test Source"
        assert source.snr_score == 0.9
        assert source.priority == "HIGH"
        assert not source.loaded

    def test_knowledge_query_dataclass(self):
        """Test KnowledgeQuery dataclass."""
        query = KnowledgeQuery(
            query="test query",
            max_results=5,
            min_snr=0.8,
            categories=["memory", "graph"],
            requester="MASTER_REASONER",
        )
        assert query.query == "test query"
        assert query.max_results == 5
        assert "memory" in query.categories

    def test_integrator_initialization(self):
        """Test KnowledgeIntegrator initialization."""
        integrator = KnowledgeIntegrator(ihsan_threshold=0.95)
        assert integrator.ihsan_threshold == 0.95
        assert integrator.cache_enabled
        # Should have built catalog from available sources
        stats = integrator.stats()
        assert "sources_discovered" in stats
        assert "query_count" in stats

    def test_integrator_catalog(self):
        """Test knowledge source catalog."""
        integrator = KnowledgeIntegrator()
        catalog = integrator.get_source_catalog()
        assert isinstance(catalog, list)
        for item in catalog:
            assert "name" in item
            assert "category" in item
            assert "priority" in item

    @pytest.mark.asyncio
    async def test_integrator_query(self):
        """Test knowledge query execution."""
        integrator = KnowledgeIntegrator()
        await integrator.initialize()

        query = KnowledgeQuery(
            query="test",
            max_results=3,
            min_snr=0.5,  # Low threshold for test
        )
        result = await integrator.query(query)

        assert isinstance(result, KnowledgeResult)
        assert result.query_id.startswith("kq-")
        assert isinstance(result.results, list)

    @pytest.mark.asyncio
    async def test_integrator_momo_context(self):
        """Test MoMo context retrieval."""
        integrator = KnowledgeIntegrator()
        await integrator.initialize()

        context = integrator.get_momo_context()
        assert isinstance(context, dict)

    @pytest.mark.asyncio
    async def test_integrator_cache(self):
        """Test query caching."""
        integrator = KnowledgeIntegrator(cache_enabled=True)
        await integrator.initialize()

        query = KnowledgeQuery(query="cache test", min_snr=0.5)

        # First query
        result1 = await integrator.query(query)
        assert not result1.from_cache

        # Second identical query should hit cache
        result2 = await integrator.query(query)
        assert result2.from_cache

    @pytest.mark.asyncio
    async def test_create_knowledge_integrator_factory(self):
        """Test factory function."""
        integrator = await create_knowledge_integrator(ihsan_threshold=0.9)
        assert isinstance(integrator, KnowledgeIntegrator)
        assert integrator.ihsan_threshold == 0.9


# =============================================================================
# SWARM KNOWLEDGE BRIDGE TESTS
# =============================================================================

class TestSwarmKnowledgeBridge:
    """Tests for SwarmKnowledgeBridge."""

    def test_role_knowledge_access_matrix(self):
        """Test role-based access matrix."""
        # Master Reasoner should have wide access
        master_access = ROLE_KNOWLEDGE_ACCESS.get(AgentRole.MASTER_REASONER, {})
        assert master_access.get("priority_access") is True
        assert master_access.get("can_write") is True
        assert "memory" in master_access.get("categories", set())

        # Data Analyzer should have analysis access
        analyzer_access = ROLE_KNOWLEDGE_ACCESS.get(AgentRole.DATA_ANALYZER, {})
        assert "corpus" in analyzer_access.get("categories", set())
        assert "embedding" in analyzer_access.get("categories", set())

        # Security Guardian should have security focus
        security_access = ROLE_KNOWLEDGE_ACCESS.get(AgentRole.SECURITY_GUARDIAN, {})
        assert security_access.get("priority_access") is True
        assert "session" in security_access.get("categories", set())

    def test_agent_knowledge_context_dataclass(self):
        """Test AgentKnowledgeContext dataclass."""
        context = AgentKnowledgeContext(
            agent_role=AgentRole.MASTER_REASONER,
            accessible_categories={"memory", "graph"},
        )
        assert context.agent_role == AgentRole.MASTER_REASONER
        assert "memory" in context.accessible_categories

    def test_knowledge_injection_dataclass(self):
        """Test KnowledgeInjection dataclass."""
        injection = KnowledgeInjection(
            agent_role=AgentRole.DATA_ANALYZER,
            knowledge_type="context",
            content={"key": "value"},
            source="test_source",
            snr_score=0.9,
        )
        assert injection.knowledge_type == "context"
        assert injection.snr_score == 0.9

    @pytest.mark.asyncio
    async def test_bridge_initialization(self):
        """Test SwarmKnowledgeBridge initialization."""
        # Create with mock integrator
        mock_integrator = MagicMock()
        mock_integrator.stats.return_value = {"test": True}
        mock_integrator.get_momo_context.return_value = {"user": "MoMo"}
        mock_integrator.get_standing_on_giants.return_value = []

        bridge = SwarmKnowledgeBridge(
            integrator=mock_integrator,
            ihsan_threshold=0.95,
        )
        result = await bridge.initialize()

        assert result["integrator_initialized"] is True
        assert result["agents_configured"] == len(AgentRole)

    @pytest.mark.asyncio
    async def test_bridge_query_for_agent(self):
        """Test knowledge query for specific agent role."""
        # Create mock integrator
        mock_integrator = MagicMock()
        mock_result = KnowledgeResult(
            query_id="test-001",
            results=[{"data": "test"}],
            snr_score=0.9,
        )
        mock_integrator.query = AsyncMock(return_value=mock_result)
        mock_integrator.stats.return_value = {}
        mock_integrator.get_momo_context.return_value = {}
        mock_integrator.get_standing_on_giants.return_value = []

        bridge = SwarmKnowledgeBridge(integrator=mock_integrator)
        await bridge.initialize()

        result = await bridge.query_for_agent(
            role=AgentRole.MASTER_REASONER,
            query="test query",
            max_results=5,
        )

        assert result.query_id == "test-001"
        assert bridge._queries_served == 1

    @pytest.mark.asyncio
    async def test_bridge_inject_knowledge(self):
        """Test knowledge injection into agent context."""
        mock_integrator = MagicMock()
        mock_integrator.stats.return_value = {}
        mock_integrator.get_momo_context.return_value = {}
        mock_integrator.get_standing_on_giants.return_value = []

        bridge = SwarmKnowledgeBridge(integrator=mock_integrator)
        await bridge.initialize()

        injection = KnowledgeInjection(
            agent_role=AgentRole.DATA_ANALYZER,
            knowledge_type="reference",
            content={"important": "data"},
            source="test",
            snr_score=0.9,  # Above threshold
        )

        success = await bridge.inject_knowledge(injection)
        assert success is True
        assert bridge._injections_made == 1

    @pytest.mark.asyncio
    async def test_bridge_reject_low_snr_injection(self):
        """Test that low SNR injections are rejected."""
        mock_integrator = MagicMock()
        mock_integrator.stats.return_value = {}
        mock_integrator.get_momo_context.return_value = {}
        mock_integrator.get_standing_on_giants.return_value = []

        bridge = SwarmKnowledgeBridge(
            integrator=mock_integrator,
            ihsan_threshold=0.95,
        )
        await bridge.initialize()

        injection = KnowledgeInjection(
            agent_role=AgentRole.DATA_ANALYZER,
            knowledge_type="reference",
            content={"low_quality": "data"},
            source="test",
            snr_score=0.7,  # Below threshold
        )

        success = await bridge.inject_knowledge(injection)
        assert success is False

    @pytest.mark.asyncio
    async def test_bridge_share_knowledge(self):
        """Test cross-agent knowledge sharing."""
        mock_integrator = MagicMock()
        mock_integrator.stats.return_value = {}
        mock_integrator.get_momo_context.return_value = {"user": "MoMo"}
        mock_integrator.get_standing_on_giants.return_value = []

        bridge = SwarmKnowledgeBridge(integrator=mock_integrator)
        await bridge.initialize()

        # Share knowledge from Master Reasoner to Data Analyzer
        success = await bridge.share_knowledge(
            from_role=AgentRole.MASTER_REASONER,
            to_role=AgentRole.DATA_ANALYZER,
            knowledge_key="momo_context",
        )

        assert success is True
        assert bridge._cross_agent_shares == 1

    def test_bridge_get_agent_context(self):
        """Test getting agent knowledge context."""
        mock_integrator = MagicMock()
        mock_integrator.stats.return_value = {}

        bridge = SwarmKnowledgeBridge(integrator=mock_integrator)
        # Manually initialize contexts for test
        bridge._agent_contexts[AgentRole.MASTER_REASONER] = AgentKnowledgeContext(
            agent_role=AgentRole.MASTER_REASONER,
            accessible_categories={"memory", "graph"},
        )
        bridge._injection_queue[AgentRole.MASTER_REASONER] = []

        context = bridge.get_agent_context(AgentRole.MASTER_REASONER)

        assert context["role"] == "master_reasoner"
        assert "memory" in context["accessible_categories"]

    @pytest.mark.asyncio
    async def test_bridge_stats(self):
        """Test bridge statistics."""
        mock_integrator = MagicMock()
        mock_integrator.stats.return_value = {"sources": 5}
        mock_integrator.get_momo_context.return_value = {}
        mock_integrator.get_standing_on_giants.return_value = []

        bridge = SwarmKnowledgeBridge(integrator=mock_integrator)
        await bridge.initialize()

        stats = bridge.stats()

        assert "queries_served" in stats
        assert "injections_made" in stats
        assert "cross_agent_shares" in stats
        assert "agents_with_context" in stats


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestKnowledgeIntegrationE2E:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_full_knowledge_pipeline(self):
        """Test complete knowledge retrieval pipeline."""
        # Create real integrator
        integrator = KnowledgeIntegrator(ihsan_threshold=0.9)
        await integrator.initialize()

        # Create bridge with real integrator
        bridge = SwarmKnowledgeBridge(
            integrator=integrator,
            ihsan_threshold=0.9,
        )
        await bridge.initialize()

        # Query for Master Reasoner
        result = await bridge.query_for_agent(
            role=AgentRole.MASTER_REASONER,
            query="sovereign",
            max_results=5,
            min_snr=0.5,
        )

        assert isinstance(result, KnowledgeResult)

        # Check bridge stats
        stats = bridge.stats()
        assert stats["queries_served"] >= 1

    @pytest.mark.asyncio
    async def test_role_based_access_enforcement(self):
        """Test that role-based access is enforced."""
        integrator = KnowledgeIntegrator()
        await integrator.initialize()

        bridge = SwarmKnowledgeBridge(integrator=integrator)
        await bridge.initialize()

        # Master Reasoner has wide access
        master_ctx = bridge.get_agent_context(AgentRole.MASTER_REASONER)
        assert "memory" in master_ctx["accessible_categories"]
        assert "graph" in master_ctx["accessible_categories"]

        # Communicator has limited access
        comm_ctx = bridge.get_agent_context(AgentRole.COMMUNICATOR)
        assert "memory" not in comm_ctx["accessible_categories"]
        assert "session" in comm_ctx["accessible_categories"]

    @pytest.mark.asyncio
    async def test_momo_context_propagation(self):
        """Test MoMo context is available to priority agents."""
        integrator = KnowledgeIntegrator()
        await integrator.initialize()

        bridge = SwarmKnowledgeBridge(integrator=integrator)
        await bridge.initialize()

        # MoMo context should be available via bridge
        momo = bridge.get_momo_context()
        assert isinstance(momo, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
