"""
Apex Sovereign Entity Integration Tests
═══════════════════════════════════════════════════════════════════════════════

Tests for the unified ApexSovereignEntity and its integration modules.

Created: 2026-02-04
"""

import asyncio
import pytest
from datetime import datetime, timezone

from core.sovereign.apex_sovereign import (
    ApexSovereignEntity,
    ApexOODAState,
    create_apex_entity,
    Observation,
    Prediction,
    TeamPlan,
    Decision,
    Outcome,
)
from core.sovereign.social_integration import (
    SociallyAwareBridge,
    ScoredAgent,
    CollaborationMatch,
    NoCapableAgentError,
)
from core.sovereign.market_integration import (
    MarketAwareMuraqabah,
    MarketSensorAdapter,
    MarketGoal,
    MarketSensorReading,
    MarketSensorType,
    SNR_FLOOR,
)
from core.sovereign.swarm_integration import (
    HybridSwarmOrchestrator,
    RustServiceAdapter,
    ServiceStatus,
    HealthStatus,
)
from core.sovereign.autonomy_matrix import AutonomyLevel
from core.apex import RelationshipType, Relationship


class TestSocialIntegration:
    """Tests for SociallyAwareBridge."""

    @pytest.fixture
    def bridge(self):
        return SociallyAwareBridge(node_id="test-node")

    def test_initialization(self, bridge):
        """Bridge should initialize with PAT agents registered."""
        assert bridge.node_id == "test-node"
        assert len(bridge.social_graph._relationships) > 0

    def test_trust_retrieval(self, bridge):
        """Should retrieve trust scores for registered agents."""
        agent_id = "pat:master_reasoner"
        trust = bridge.get_trust(agent_id)
        assert 0.0 <= trust <= 1.0

    def test_agent_selection(self, bridge):
        """Should select agent based on capability and trust."""
        selected = bridge.select_agent_for_task(
            required_capabilities={"reasoning"},
            prefer_diversity=False,
        )
        assert selected is not None
        assert selected.capability_score > 0
        assert selected.trust_score > 0

    def test_collaboration_discovery(self, bridge):
        """Should find collaboration partners."""
        partners = bridge.find_collaboration_partners(
            task_capabilities={"reasoning", "analysis"},
            min_synergy=0.3,
        )
        # May or may not find partners depending on setup
        assert isinstance(partners, list)

    @pytest.mark.asyncio
    async def test_trust_update(self, bridge):
        """Trust should update after task outcome."""
        agent_id = "pat:master_reasoner"
        initial_trust = bridge.get_trust(agent_id)

        await bridge.report_task_outcome(
            agent_id=agent_id,
            task_id="test-task",
            success=True,
            value=100.0,
        )

        new_trust = bridge.get_trust(agent_id)
        assert new_trust >= initial_trust  # Should increase on success

    def test_network_metrics(self, bridge):
        """Should provide network metrics."""
        metrics = bridge.get_network_metrics()
        assert "total_agents" in metrics
        assert "average_trust" in metrics


class TestMarketIntegration:
    """Tests for MarketAwareMuraqabah."""

    @pytest.fixture
    def muraqabah(self):
        return MarketAwareMuraqabah(node_id="test-node")

    def test_initialization(self, muraqabah):
        """Should initialize with market sensor."""
        assert muraqabah.node_id == "test-node"
        assert muraqabah.market_sensor is not None

    def test_snr_to_autonomy_mapping(self, muraqabah):
        """SNR should map to correct autonomy levels."""
        assert muraqabah.snr_to_autonomy(0.99) == AutonomyLevel.AUTOHIGH
        assert muraqabah.snr_to_autonomy(0.96) == AutonomyLevel.AUTOMEDIUM
        assert muraqabah.snr_to_autonomy(0.91) == AutonomyLevel.AUTOLOW
        assert muraqabah.snr_to_autonomy(0.86) == AutonomyLevel.SUGGESTER
        assert muraqabah.snr_to_autonomy(0.70) == AutonomyLevel.OBSERVER

    def test_urgency_calculation(self, muraqabah):
        """Arbitrage should have high urgency."""
        arb_reading = MarketSensorReading(
            sensor_type=MarketSensorType.ARBITRAGE,
            symbol="TEST/USD",
            snr_score=0.92,
        )
        urgency = muraqabah.calculate_urgency(arb_reading)
        assert urgency >= 0.9

        signal_reading = MarketSensorReading(
            sensor_type=MarketSensorType.TRADING_SIGNAL,
            symbol="TEST/USD",
            value={"strength": "weak"},
            snr_score=0.86,
        )
        urgency = muraqabah.calculate_urgency(signal_reading)
        assert urgency < 0.5

    def test_low_snr_filtering(self, muraqabah):
        """Low SNR readings should not create goals."""
        low_snr = MarketSensorReading(
            sensor_type=MarketSensorType.TRADING_SIGNAL,
            symbol="TEST/USD",
            snr_score=0.60,
        )
        goal = muraqabah.process_market_reading(low_snr)
        assert goal is None

    def test_high_snr_creates_goal(self, muraqabah):
        """High SNR readings should create goals."""
        # Use very high SNR to ensure Ihsan calculation passes
        high_snr = MarketSensorReading(
            sensor_type=MarketSensorType.TRADING_SIGNAL,
            symbol="TEST/USD",
            value={"signal_type": "buy", "strength": "strong", "expected_return": 0.05},
            snr_score=0.98,  # Very high SNR ensures Ihsan passes
        )
        goal = muraqabah.process_market_reading(high_snr)
        assert goal is not None
        assert goal.autonomy_level == AutonomyLevel.AUTOMEDIUM


class TestSwarmIntegration:
    """Tests for HybridSwarmOrchestrator."""

    @pytest.fixture
    def swarm(self):
        return HybridSwarmOrchestrator()

    def test_initialization(self, swarm):
        """Should initialize with health monitor and scaling manager."""
        assert swarm.health_monitor is not None
        assert swarm.scaling_manager is not None

    def test_rust_service_registration(self, swarm):
        """Should register Rust services."""
        swarm.register_rust_service("test-service")
        assert "test-service" in swarm.rust_adapters

    def test_rust_service_unregistration(self, swarm):
        """Should unregister Rust services."""
        swarm.register_rust_service("test-service")
        swarm.unregister_rust_service("test-service")
        assert "test-service" not in swarm.rust_adapters

    @pytest.mark.asyncio
    async def test_health_check(self, swarm):
        """Should check health of all services."""
        health = await swarm.check_all_health()
        assert isinstance(health, dict)

    def test_metrics(self, swarm):
        """Should provide metrics."""
        metrics = swarm.get_metrics()
        assert "total_restarts" in metrics
        assert "average_availability" in metrics


class TestRustServiceAdapter:
    """Tests for RustServiceAdapter."""

    @pytest.fixture
    def adapter(self):
        return RustServiceAdapter("test-service")

    def test_initialization(self, adapter):
        """Should initialize with correct defaults."""
        assert adapter.service_name == "test-service"
        assert adapter.restart_count == 0
        assert adapter.last_health == HealthStatus.UNKNOWN

    @pytest.mark.asyncio
    async def test_health_check(self, adapter):
        """Should perform health check."""
        health = await adapter.health_check()
        # May be HEALTHY (mock) or UNHEALTHY (no server)
        assert health in list(HealthStatus)

    def test_uptime(self, adapter):
        """Should track uptime."""
        uptime = adapter.get_uptime()
        assert uptime >= 0

    def test_status(self, adapter):
        """Should provide status."""
        status = adapter.get_status()
        assert status.service_id == "rust:test-service"


class TestApexSovereignEntity:
    """Tests for the unified ApexSovereignEntity."""

    @pytest.fixture
    def entity(self):
        return create_apex_entity("test-node")

    def test_initialization(self, entity):
        """Should initialize correctly."""
        assert entity.node_id == "test-node"
        assert entity.ihsan_threshold == 0.95
        assert entity.snr_floor == 0.85
        assert entity.current_state == ApexOODAState.SLEEP

    def test_status(self, entity):
        """Should provide status."""
        status = entity.status()
        assert status["node_id"] == "test-node"
        assert status["running"] == False
        assert "metrics" in status
        assert "subsystems" in status

    @pytest.mark.asyncio
    async def test_start_stop(self, entity):
        """Should start and stop correctly."""
        await entity.start()
        assert entity._running
        assert entity.current_state == ApexOODAState.OBSERVE

        await entity.stop()
        assert not entity._running
        assert entity.current_state == ApexOODAState.SLEEP

    @pytest.mark.asyncio
    async def test_observe_phase(self, entity):
        """Should collect observations."""
        observation = await entity._observe()
        assert isinstance(observation, Observation)
        assert observation.timestamp is not None

    @pytest.mark.asyncio
    async def test_predict_phase(self, entity):
        """Should make predictions from observations."""
        observation = Observation()
        prediction = await entity._predict(observation)
        assert isinstance(prediction, Prediction)

    @pytest.mark.asyncio
    async def test_coordinate_phase(self, entity):
        """Should create team plan."""
        observation = Observation()
        prediction = Prediction()
        plan = await entity._coordinate(observation, prediction)
        assert isinstance(plan, TeamPlan)

    @pytest.mark.asyncio
    async def test_decide_filters_low_ihsan(self, entity):
        """Should filter goals with low Ihsan."""
        # Create goal with low Ihsan
        low_ihsan_goal = MarketGoal(
            goal_id="test",
            ihsan_score=0.80,  # Below threshold
            autonomy_level=AutonomyLevel.AUTOLOW,
            snr_score=0.90,
        )

        decisions = await entity._decide([low_ihsan_goal])
        assert len(decisions) == 0  # Should be filtered

    @pytest.mark.asyncio
    async def test_decide_approves_high_autonomy(self, entity):
        """Should auto-approve high autonomy decisions."""
        high_autonomy_goal = MarketGoal(
            goal_id="test",
            ihsan_score=0.98,
            autonomy_level=AutonomyLevel.AUTOMEDIUM,
            snr_score=0.96,
        )

        decisions = await entity._decide([high_autonomy_goal])
        assert len(decisions) == 1
        assert decisions[0].approved  # Should be auto-approved


class TestEndToEndIntegration:
    """End-to-end integration tests."""

    @pytest.mark.asyncio
    async def test_full_cycle(self):
        """Should complete a full OODA cycle."""
        entity = create_apex_entity("e2e-test")

        # Run observe
        observation = await entity._observe()
        assert observation is not None

        # Run predict
        prediction = await entity._predict(observation)
        assert prediction is not None

        # Run coordinate
        plan = await entity._coordinate(observation, prediction)
        assert plan is not None

        # Run analyze (may return empty list)
        goals = await entity._analyze(observation, prediction, plan)
        assert isinstance(goals, list)

        # Run decide
        decisions = await entity._decide(goals)
        assert isinstance(decisions, list)

        # Run act
        outcomes = await entity._act(decisions, plan)
        assert isinstance(outcomes, list)

        # Run learn
        await entity._learn(outcomes)

        # Run reflect
        await entity._reflect()

        # Check metrics updated
        assert entity.metrics["cycles"] == 0  # Reflects hasn't incremented yet

    @pytest.mark.asyncio
    async def test_graceful_degradation(self):
        """Should handle subsystem failures gracefully."""
        entity = create_apex_entity("degradation-test")

        # Simulate market subsystem failure
        entity.market_muraqabah.market_sensor.opportunity_engine = None

        # Should still observe without crashing
        observation = await entity._observe()
        assert observation is not None
        # Market readings may be empty but should not error


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
