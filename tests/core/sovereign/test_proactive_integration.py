"""
Tests for Proactive Sovereign Entity
====================================
Comprehensive tests for the complete proactive sovereign architecture.
"""

import asyncio
import pytest
from datetime import datetime, timezone, timedelta

# Event Bus
from core.sovereign.event_bus import EventBus, Event, EventPriority, get_event_bus

# State Checkpointer
from core.sovereign.state_checkpointer import StateCheckpointer, Checkpoint

# Team Planner
from core.sovereign.team_planner import (
    TeamPlanner, Goal, TeamTask, AgentRole, TaskComplexity, TaskAllocation
)

# Dual-Agentic Bridge
from core.sovereign.dual_agentic_bridge import (
    DualAgenticBridge, ActionProposal, ConsensusResult, VetoReason
)

# Collective Intelligence
from core.sovereign.collective_intelligence import (
    CollectiveIntelligence, AgentContribution, AggregationMethod
)

# Proactive Scheduler
from core.sovereign.proactive_scheduler import (
    ProactiveScheduler, ScheduledJob, ScheduleType, JobPriority
)

# Predictive Monitor
from core.sovereign.predictive_monitor import (
    PredictiveMonitor, TrendDirection, AlertSeverity
)

# Muraqabah Engine
from core.sovereign.muraqabah_engine import (
    MuraqabahEngine, MonitorDomain, Opportunity, SensorReading
)

# Autonomy Matrix
from core.sovereign.autonomy_matrix import (
    AutonomyMatrix, AutonomyLevel, ActionContext, AutonomyConstraints
)

# Enhanced Team Planner
from core.sovereign.enhanced_team_planner import (
    EnhancedTeamPlanner, ProactiveGoal, ExecutionPlan
)

# Proactive Integration
from core.sovereign.proactive_integration import (
    ProactiveSovereignEntity, EntityConfig, EntityMode, create_proactive_entity
)


# =============================================================================
# EVENT BUS TESTS
# =============================================================================

class TestEventBus:
    """Tests for EventBus."""

    @pytest.mark.asyncio
    async def test_publish_subscribe(self):
        """Test basic pub/sub."""
        bus = EventBus()
        received = []

        async def handler(event: Event):
            received.append(event)

        bus.subscribe("test.topic", handler)
        await bus.emit("test.topic", {"data": "value"})

        # Process event manually (no background loop)
        event = (await bus._event_queue.get())[2]
        await bus._process_event(event)

        assert len(received) == 1
        assert received[0].payload["data"] == "value"

    @pytest.mark.asyncio
    async def test_wildcard_subscription(self):
        """Test wildcard subscriptions."""
        bus = EventBus()
        received = []

        async def handler(event: Event):
            received.append(event.topic)

        bus.subscribe("test.*", handler)
        await bus.emit("test.one", {})
        await bus.emit("test.two", {})
        await bus.emit("other.topic", {})

        # Process events
        while not bus._event_queue.empty():
            event = (await bus._event_queue.get())[2]
            await bus._process_event(event)

        assert len(received) == 2
        assert "test.one" in received
        assert "test.two" in received


# =============================================================================
# AUTONOMY MATRIX TESTS
# =============================================================================

class TestAutonomyMatrix:
    """Tests for AutonomyMatrix."""

    def test_determine_autonomy_low_risk(self):
        """Test low-risk action gets AUTOLOW."""
        matrix = AutonomyMatrix(default_level=AutonomyLevel.SUGGESTER)

        context = ActionContext(
            action_type="simple_task",
            risk_score=0.1,
            cost_percent=0.5,
            ihsan_score=0.98,
            is_reversible=True,
        )

        decision = matrix.determine_autonomy(context)
        assert decision.determined_level == AutonomyLevel.AUTOLOW
        assert decision.constraints_met

    def test_determine_autonomy_high_risk(self):
        """Test high-risk action requires approval."""
        matrix = AutonomyMatrix(default_level=AutonomyLevel.SUGGESTER)

        context = ActionContext(
            action_type="risky_task",
            risk_score=0.8,
            cost_percent=15.0,
            ihsan_score=0.9,
            is_reversible=False,
        )

        decision = matrix.determine_autonomy(context)
        assert decision.determined_level == AutonomyLevel.OBSERVER
        assert not decision.can_execute

    def test_emergency_override(self):
        """Test emergency gets SOVEREIGN level."""
        matrix = AutonomyMatrix()

        context = ActionContext(
            action_type="emergency_task",
            is_emergency=True,
        )

        decision = matrix.determine_autonomy(context)
        assert decision.determined_level == AutonomyLevel.SOVEREIGN


# =============================================================================
# DUAL-AGENTIC BRIDGE TESTS
# =============================================================================

class TestDualAgenticBridge:
    """Tests for DualAgenticBridge."""

    @pytest.mark.asyncio
    async def test_proposal_approval(self):
        """Test proposal gets approved with good scores."""
        bridge = DualAgenticBridge(ihsan_threshold=0.95)

        proposal = ActionProposal(
            task_id="task-1",
            action_type="safe_action",
            ihsan_estimate=0.98,
            risk_estimate=0.1,
        )

        await bridge.submit_proposal(proposal)
        outcome = await bridge.validate(proposal.id)

        assert outcome.result == ConsensusResult.APPROVED
        assert outcome.quorum_met

    @pytest.mark.asyncio
    async def test_proposal_veto_security(self):
        """Test dangerous proposal gets vetoed."""
        bridge = DualAgenticBridge(ihsan_threshold=0.95)

        proposal = ActionProposal(
            task_id="task-2",
            action_type="dangerous_action",
            parameters={"command": "rm -rf /"},  # Dangerous pattern
            ihsan_estimate=0.98,
            risk_estimate=0.9,
        )

        await bridge.submit_proposal(proposal)
        outcome = await bridge.validate(proposal.id)

        assert outcome.result == ConsensusResult.VETOED


# =============================================================================
# COLLECTIVE INTELLIGENCE TESTS
# =============================================================================

class TestCollectiveIntelligence:
    """Tests for CollectiveIntelligence."""

    @pytest.mark.asyncio
    async def test_weighted_average(self):
        """Test weighted average aggregation."""
        ci = CollectiveIntelligence()

        contributions = [
            AgentContribution(
                agent_role=AgentRole.MASTER_REASONER,
                content=0.8,
                confidence=0.9,
            ),
            AgentContribution(
                agent_role=AgentRole.DATA_ANALYZER,
                content=0.7,
                confidence=0.8,
            ),
        ]

        decision = await ci.collect("What's the value?", contributions)

        assert decision.result is not None
        assert 0.7 <= decision.result <= 0.8
        assert decision.confidence > 0

    @pytest.mark.asyncio
    async def test_synergy_score(self):
        """Test synergy score calculation."""
        ci = CollectiveIntelligence(synergy_bonus=0.1)

        contributions = [
            AgentContribution(content=0.9, confidence=0.9),
            AgentContribution(content=0.9, confidence=0.9),
            AgentContribution(content=0.9, confidence=0.9),
        ]

        decision = await ci.collect("Test", contributions)

        # With high agreement, confidence should be boosted
        assert decision.confidence > 0.9


# =============================================================================
# TEAM PLANNER TESTS
# =============================================================================

class TestTeamPlanner:
    """Tests for TeamPlanner."""

    @pytest.mark.asyncio
    async def test_goal_decomposition(self):
        """Test goal decomposition into tasks."""
        planner = TeamPlanner()

        goal = Goal(
            description="Create a simple report",
            success_criteria=["Report completed"],
            priority=0.5,
        )

        tasks = await planner.decompose_goal(goal)

        assert len(tasks) >= 1
        assert all(isinstance(t, TeamTask) for t in tasks)

    def test_task_allocation(self):
        """Test task allocation to roles."""
        planner = TeamPlanner()

        task = TeamTask(
            name="Analyze data patterns",
            description="Analyze data patterns in the dataset",
            complexity=TaskComplexity.SIMPLE,
        )

        allocations = planner.allocate_task(task)

        assert len(allocations) >= 1
        assert any(a.role == AgentRole.DATA_ANALYZER for a in allocations)


# =============================================================================
# PREDICTIVE MONITOR TESTS
# =============================================================================

class TestPredictiveMonitor:
    """Tests for PredictiveMonitor."""

    def test_trend_detection(self):
        """Test trend detection from readings."""
        monitor = PredictiveMonitor()

        # Record increasing values
        for i in range(10):
            monitor.record("test_metric", 0.5 + i * 0.05)

        analysis = monitor.analyze("test_metric")

        assert analysis is not None
        assert analysis.direction == TrendDirection.RISING
        assert analysis.slope > 0


# =============================================================================
# MURAQABAH ENGINE TESTS
# =============================================================================

class TestMuraqabahEngine:
    """Tests for MuraqabahEngine."""

    @pytest.mark.asyncio
    async def test_sensor_registration(self):
        """Test sensor registration."""
        engine = MuraqabahEngine()

        def test_sensor():
            return {"test_value": 0.5}

        engine.register_sensor(MonitorDomain.ENVIRONMENTAL, "test", test_sensor)

        stats = engine.stats()
        assert stats["sensors_by_domain"]["environmental"] >= 1

    @pytest.mark.asyncio
    async def test_scan_domain(self):
        """Test domain scanning."""
        engine = MuraqabahEngine()

        result = await engine.scan(MonitorDomain.ENVIRONMENTAL)

        assert "readings" in result
        assert result["readings"] >= 0


# =============================================================================
# PROACTIVE SOVEREIGN ENTITY TESTS
# =============================================================================

class TestProactiveSovereignEntity:
    """Tests for ProactiveSovereignEntity."""

    def test_entity_creation(self):
        """Test entity creation."""
        entity = create_proactive_entity(
            mode=EntityMode.PROACTIVE_PARTNER,
            ihsan_threshold=0.95,
            autonomy_level=AutonomyLevel.AUTOLOW,
        )

        assert entity is not None
        assert entity.config.mode == EntityMode.PROACTIVE_PARTNER
        assert entity.config.ihsan_threshold == 0.95

    def test_entity_stats(self):
        """Test entity stats collection."""
        entity = create_proactive_entity()

        stats = entity.stats()

        assert "running" in stats
        assert "mode" in stats
        assert "autonomy" in stats
        assert "bridge" in stats

    @pytest.mark.asyncio
    async def test_single_cycle(self):
        """Test running a single cycle."""
        entity = create_proactive_entity(
            mode=EntityMode.REACTIVE,  # Simpler mode for testing
        )

        result = await entity.run_cycle()

        assert result.cycle_number == 1
        assert result.health_score >= 0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestIntegration:
    """Integration tests for the complete system."""

    @pytest.mark.asyncio
    async def test_opportunity_to_execution_flow(self):
        """Test full flow from opportunity to execution."""
        # Create planner with all components
        planner = EnhancedTeamPlanner(ihsan_threshold=0.95)

        # Create an opportunity
        opportunity = Opportunity(
            domain=MonitorDomain.ENVIRONMENTAL,
            description="Optimize CPU usage",
            estimated_value=0.6,
            urgency=0.5,
            confidence=0.9,
        )

        # Handle opportunity
        goal = await planner.handle_opportunity(opportunity)

        assert goal is not None
        assert goal.domain == MonitorDomain.ENVIRONMENTAL

        # Plan the goal
        plan = await planner.plan_goal(goal)

        assert plan is not None
        assert len(plan.tasks) >= 1

    @pytest.mark.asyncio
    async def test_autonomy_enforced_in_execution(self):
        """Test that autonomy constraints are enforced."""
        entity = create_proactive_entity(
            mode=EntityMode.PROACTIVE_AUTO,
            autonomy_level=AutonomyLevel.SUGGESTER,
        )

        # High-risk context should not auto-execute
        context = ActionContext(
            action_type="risky_action",
            risk_score=0.9,
            ihsan_score=0.85,
        )

        decision = entity.autonomy.determine_autonomy(context)

        assert not decision.can_execute
        assert decision.requires_approval


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
