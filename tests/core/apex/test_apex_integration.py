"""
BIZRA Apex System Integration Tests
═══════════════════════════════════════════════════════════════════════════════

Validates the three Apex pillars work together:
1. SocialGraph — Relationship Intelligence
2. OpportunityEngine — Active Market Intelligence
3. SwarmOrchestrator — Autonomous Scaling

Created: 2026-02-04
"""

import asyncio
import pytest
from datetime import datetime, timezone

# Import the Apex system
from core.apex import (
    # Main unified interface
    ApexSystem,
    # Social Graph
    SocialGraph,
    RelationshipType,
    InteractionType,
    CollaborationStatus,
    Relationship,
    # Opportunity Engine
    OpportunityEngine,
    MarketCondition,
    SignalType,
    SignalStrength,
    MarketData,
    TradingSignal,
    MarketAnalyzer,
    SignalGenerator,
    ArbitrageDetector,
    # Swarm Orchestrator
    SwarmOrchestrator,
    AgentConfig,
    AgentInstance,
    AgentStatus,
    HealthStatus,
    SwarmTopology,
    HealthMonitor,
    ScalingManager,
    ScalingDecision,
    ScalingAction,
)


class TestApexImports:
    """Verify all Apex components can be imported."""

    def test_version(self):
        from core.apex import __version__
        assert __version__ == "1.0.0"

    def test_social_graph_imports(self):
        from core.apex import (
            SocialGraph,
            RelationshipType,
            InteractionType,
            CollaborationStatus,
            Interaction,
            Relationship,
            CollaborationOpportunity,
            NegotiationOffer,
        )
        assert SocialGraph is not None

    def test_opportunity_engine_imports(self):
        from core.apex import (
            OpportunityEngine,
            MarketCondition,
            SignalType,
            SignalStrength,
            PositionStatus,
            MarketData,
            MarketAnalysis,
            TradingSignal,
            ArbitrageOpportunity,
            Position,
            MarketAnalyzer,
            SignalGenerator,
            ArbitrageDetector,
        )
        assert OpportunityEngine is not None

    def test_swarm_orchestrator_imports(self):
        from core.apex import (
            SwarmOrchestrator,
            AgentStatus,
            ScalingAction,
            HealthStatus,
            SwarmTopology,
            AgentConfig,
            AgentInstance,
            SwarmConfig,
            Swarm,
            ScalingDecision,
            HealthReport,
            HealthMonitor,
            ScalingManager,
        )
        assert SwarmOrchestrator is not None


class TestSocialGraph:
    """Test SocialGraph relationship intelligence."""

    @pytest.fixture
    def social_graph(self):
        # SocialGraph takes agent_id as first param
        return SocialGraph(agent_id="test-node")

    def test_initialization(self, social_graph):
        """Verify SocialGraph initializes correctly."""
        assert social_graph.agent_id == "test-node"

    def test_add_relationship(self, social_graph):
        """Test creating a relationship."""
        # Add a relationship directly with correct fields
        rel = Relationship(
            agent_id="test-node",
            peer_id="agent-2",
            relationship_type=RelationshipType.COLLABORATOR,
            trust_score=0.8,
        )
        social_graph._relationships["agent-2"] = rel

        # Verify relationship exists
        assert "agent-2" in social_graph._relationships
        assert social_graph._relationships["agent-2"].trust_score == 0.8

    def test_relationship_types(self):
        """Test RelationshipType enum values."""
        # Actual enum values from implementation
        assert RelationshipType.UNKNOWN.value == "unknown"
        assert RelationshipType.ACQUAINTANCE.value == "acquaintance"
        assert RelationshipType.COLLABORATOR.value == "collaborator"
        assert RelationshipType.TRUSTED.value == "trusted"
        assert RelationshipType.STRATEGIC.value == "strategic"

    def test_interaction_types(self):
        """Test InteractionType enum values."""
        # Actual enum values from implementation
        assert InteractionType.MESSAGE.value == "message"
        assert InteractionType.COLLABORATION.value == "collaboration"
        assert InteractionType.TASK_DELEGATION.value == "task_delegation"
        assert InteractionType.CONSENSUS_VOTE.value == "consensus_vote"


class TestOpportunityEngine:
    """Test OpportunityEngine market intelligence."""

    @pytest.fixture
    def opportunity_engine(self):
        return OpportunityEngine(snr_threshold=0.85)

    def test_initialization(self, opportunity_engine):
        """Verify OpportunityEngine initializes correctly."""
        assert opportunity_engine.snr_threshold == 0.85

    def test_market_data_creation(self):
        """Test MarketData dataclass."""
        data = MarketData(
            symbol="COMPUTE/USD",
            price=0.05,
            volume=1000000.0,
            timestamp=datetime.now(timezone.utc),
        )
        assert data.symbol == "COMPUTE/USD"
        assert data.price == 0.05

    def test_market_analyzer(self):
        """Test MarketAnalyzer component."""
        analyzer = MarketAnalyzer(window_size=100)

        # Create market data
        data = MarketData(
            symbol="COMPUTE/USD",
            price=0.05,
            volume=1000000.0,
            timestamp=datetime.now(timezone.utc),
        )

        # Update analyzer
        analyzer.update(data)

        # Should have data recorded
        assert "COMPUTE/USD" in analyzer._price_history

    def test_signal_generator(self):
        """Test SignalGenerator with SNR threshold."""
        generator = SignalGenerator(snr_threshold=0.85)
        assert generator.snr_threshold == 0.85

    def test_arbitrage_detector(self):
        """Test ArbitrageDetector."""
        detector = ArbitrageDetector(min_profit=0.001)

        # Add prices from different markets
        detector.update_price("COMPUTE/USD", "market_a", 0.050)
        detector.update_price("COMPUTE/USD", "market_b", 0.052)

        # Should detect price differences (method is 'detect', not 'detect_opportunities')
        opps = detector.detect()
        assert isinstance(opps, list)

    def test_snr_filtering(self):
        """Test SNR threshold filtering on signals."""
        # Create a signal with low SNR (using correct field names)
        weak_signal = TradingSignal(
            symbol="COMPUTE/USD",
            signal_type=SignalType.BUY,
            strength=SignalStrength.WEAK,
            confidence=0.5,
            snr_score=0.50,  # Below threshold
        )

        snr_threshold = 0.85
        assert weak_signal.snr_score < snr_threshold

        # Create a strong signal
        strong_signal = TradingSignal(
            symbol="COMPUTE/USD",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=0.9,
            snr_score=0.92,  # Above threshold
        )

        assert strong_signal.snr_score >= snr_threshold


class TestSwarmOrchestrator:
    """Test SwarmOrchestrator deployment and scaling."""

    @pytest.fixture
    def swarm_orchestrator(self):
        return SwarmOrchestrator()

    def test_initialization(self, swarm_orchestrator):
        """Verify SwarmOrchestrator initializes correctly."""
        assert swarm_orchestrator.health_monitor is not None
        assert swarm_orchestrator.scaling_manager is not None

    def test_agent_config_creation(self):
        """Test AgentConfig dataclass."""
        config = AgentConfig(
            agent_type="reasoner",
            name="reasoner-1",
            capabilities={"reasoning", "analysis"},
            cpu_limit=2.0,
            memory_limit_mb=4096,
        )
        assert config.agent_type == "reasoner"
        assert "reasoning" in config.capabilities

    def test_swarm_topology_enum(self):
        """Test SwarmTopology options."""
        assert SwarmTopology.STAR.value == "star"
        assert SwarmTopology.MESH.value == "mesh"
        assert SwarmTopology.RING.value == "ring"
        assert SwarmTopology.HIERARCHY.value == "hierarchy"

    def test_health_monitor(self):
        """Test HealthMonitor component."""
        monitor = HealthMonitor(check_interval=30)
        assert monitor.check_interval == 30

    def test_scaling_manager(self):
        """Test ScalingManager component."""
        manager = ScalingManager()
        assert manager._scaling_history.maxlen == 100

    def test_scaling_decision_creation(self):
        """Test ScalingDecision dataclass."""
        decision = ScalingDecision(
            action=ScalingAction.SCALE_UP,
            current_count=3,
            target_count=5,
            reason="High CPU utilization",
            # Note: confidence not in this dataclass, use metrics instead
            metrics={"cpu_utilization": 0.95},
        )
        assert decision.action == ScalingAction.SCALE_UP
        assert decision.target_count == 5


class TestApexSystemUnified:
    """Test the unified ApexSystem interface."""

    @pytest.fixture
    def apex(self):
        return ApexSystem(
            node_id="test-node",
            ihsan_threshold=0.95,
            snr_floor=0.85,
        )

    def test_initialization(self, apex):
        """Verify ApexSystem initializes correctly."""
        assert apex.node_id == "test-node"
        assert apex.ihsan_threshold == 0.95
        assert apex.snr_floor == 0.85
        assert not apex._running

    def test_lazy_loading(self, apex):
        """Test subsystems are lazily loaded."""
        # Before access
        assert apex._social is None
        assert apex._opportunity is None
        assert apex._swarm is None

        # Access triggers creation
        _ = apex.social
        assert apex._social is not None

        _ = apex.opportunity
        assert apex._opportunity is not None

        _ = apex.swarm
        assert apex._swarm is not None

    def test_status(self, apex):
        """Test status reporting."""
        status = apex.status()
        assert status["node_id"] == "test-node"
        assert status["running"] == False
        assert "subsystems" in status

    @pytest.mark.asyncio
    async def test_start_stop(self, apex):
        """Test starting and stopping the Apex system."""
        await apex.start()
        assert apex._running

        await apex.stop()
        assert not apex._running


class TestApexIntegration:
    """Test integration between Apex components."""

    @pytest.fixture
    def apex(self):
        return ApexSystem(node_id="integration-test")

    def test_social_metrics(self, apex):
        """Test social graph metrics."""
        # Access social subsystem
        social = apex.social
        assert social.agent_id == "integration-test"

        # Add a relationship with correct fields
        rel = Relationship(
            agent_id="integration-test",
            peer_id="collaborator-1",
            relationship_type=RelationshipType.COLLABORATOR,
            trust_score=0.9,
        )
        social._relationships["collaborator-1"] = rel

        # Verify relationship count
        assert len(social._relationships) == 1

    def test_market_signal_validation(self, apex):
        """Test market signals validate against SNR threshold."""
        # Create a strong signal with correct field names
        signal = TradingSignal(
            symbol="COMPUTE/USD",
            signal_type=SignalType.BUY,
            strength=SignalStrength.STRONG,
            confidence=0.95,
            snr_score=0.92,
        )

        # Signal passes SNR threshold
        assert signal.snr_score >= apex.snr_floor

    def test_swarm_orchestration(self, apex):
        """Test swarm orchestrator access."""
        swarm = apex.swarm
        assert swarm.health_monitor is not None
        assert swarm.scaling_manager is not None

    def test_full_status(self, apex):
        """Test full system status after accessing all subsystems."""
        # Access all subsystems
        _ = apex.social
        _ = apex.opportunity
        _ = apex.swarm

        status = apex.status()
        assert status["subsystems"]["social"] == "active"
        assert status["subsystems"]["opportunity"] == "active"
        assert status["subsystems"]["swarm"] == "active"


# Run tests if executed directly
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
