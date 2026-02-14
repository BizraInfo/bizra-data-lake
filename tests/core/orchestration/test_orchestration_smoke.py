"""
Orchestration Module Smoke Tests
=================================
Validates that key orchestration components can be instantiated
and their core interfaces respond correctly.

Created: 2026-02-07 | BIZRA Mastermind Sprint
"""

import pytest


# ============================================================================
# EventBus — Core event infrastructure
# ============================================================================


class TestEventBus:
    """EventBus must support publish/subscribe with priority."""

    def test_import_and_create(self):
        from core.orchestration import Event, EventBus, EventPriority

        bus = EventBus()
        assert bus is not None
        # Verify enum values exist
        assert EventPriority.HIGH is not None
        assert EventPriority.LOW is not None

    def test_event_creation(self):
        from core.orchestration import Event

        event = Event(topic="test.topic", payload={"key": "value"})
        assert event.topic == "test.topic"
        assert event.payload["key"] == "value"

    @pytest.mark.asyncio
    async def test_publish_accepts_event(self):
        from core.orchestration import Event, EventBus

        bus = EventBus()
        event = Event(topic="test.topic", payload={"msg": "hello"})
        await bus.publish(event)


# ============================================================================
# TeamPlanner — Multi-agent task decomposition
# ============================================================================


class TestTeamPlanner:
    """TeamPlanner must instantiate and expose planning interface."""

    def test_import_and_create(self):
        from core.orchestration import TeamPlanner

        planner = TeamPlanner()
        assert planner is not None
        assert hasattr(planner, "allocate_task")


# ============================================================================
# MuraqabahEngine — Monitoring engine
# ============================================================================


class TestMuraqabahEngine:
    """Muraqabah (monitoring) engine must instantiate."""

    def test_import_and_create(self):
        from core.orchestration import MuraqabahEngine

        engine = MuraqabahEngine()
        assert engine is not None

    def test_sensor_hub_import(self):
        from core.orchestration import MuraqabahSensorHub

        hub = MuraqabahSensorHub()
        assert hub is not None


# ============================================================================
# Lazy imports — Cross-package dependencies
# ============================================================================


class TestLazyImports:
    """Lazy-loaded symbols must resolve without circular import errors."""

    def test_enhanced_team_planner(self):
        from core.orchestration import EnhancedTeamPlanner

        assert EnhancedTeamPlanner is not None

    def test_background_agent(self):
        from core.orchestration import BackgroundAgent, BackgroundAgentRegistry

        assert BackgroundAgent is not None
        assert BackgroundAgentRegistry is not None

    def test_opportunity_pipeline(self):
        from core.orchestration import OpportunityPipeline, PipelineStage

        assert OpportunityPipeline is not None
        assert PipelineStage is not None

    def test_proactive_team(self):
        from core.orchestration import ProactiveTeam

        assert ProactiveTeam is not None

    def test_proactive_sovereign_entity(self):
        from core.orchestration import ProactiveSovereignEntity

        assert ProactiveSovereignEntity is not None


# ============================================================================
# __all__ completeness
# ============================================================================


def test_all_exports_resolvable():
    """Every name in __all__ must be accessible."""
    import core.orchestration as mod

    for name in mod.__all__:
        attr = getattr(mod, name, None)
        assert attr is not None, f"core.orchestration.__all__ exports '{name}' but it's None"
