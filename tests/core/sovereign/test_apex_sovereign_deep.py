"""
Apex Sovereign Entity Deep Unit Tests
===============================================================================

Comprehensive test suite for ApexSovereignEntity covering:
- Enum and dataclass invariants
- Lazy property initialization
- OODA phase internals (decide, act, learn, reflect, predict, observe)
- Error handling and edge cases
- Factory function
- Status reporting

Complements the integration tests in tests/integration/test_apex_sovereign.py
which cover basic init, status, start/stop, observe, predict, coordinate,
decide (low/high ihsan), and a full e2e cycle.

Created: 2026-02-11
"""

from __future__ import annotations

import asyncio
import logging
import time
from datetime import datetime, timezone
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.sovereign.apex_sovereign import (
    ApexOODAState,
    ApexSovereignEntity,
    Decision,
    Observation,
    Outcome,
    Prediction,
    TeamPlan,
    create_apex_entity,
)
from core.sovereign.autonomy_matrix import AutonomyLevel
from core.sovereign.market_integration import (
    MarketGoal,
    MarketSensorReading,
    MarketSensorType,
)
from core.sovereign.social_integration import ScoredAgent
from core.sovereign.swarm_integration import HealthStatus


# ---------------------------------------------------------------------------
# 1. TestApexOODAState — Enum completeness and correctness
# ---------------------------------------------------------------------------
class TestApexOODAState:
    """Verify the extended OODA state enum is complete and well-formed."""

    def test_all_nine_states_exist(self):
        """Enum should define exactly 9 states."""
        assert len(ApexOODAState) == 9

    def test_state_names(self):
        """Each member name should match its expected string."""
        expected = {
            "OBSERVE": "observe",
            "PREDICT": "predict",
            "COORDINATE": "coordinate",
            "ANALYZE": "analyze",
            "DECIDE": "decide",
            "ACT": "act",
            "LEARN": "learn",
            "REFLECT": "reflect",
            "SLEEP": "sleep",
        }
        for name, value in expected.items():
            assert ApexOODAState[name].value == value

    def test_sleep_is_initial_state(self):
        """SLEEP is the default state for a newly created entity."""
        entity = create_apex_entity("state-test")
        assert entity.current_state == ApexOODAState.SLEEP

    def test_states_are_strings(self):
        """All state values should be plain strings (str enum)."""
        for state in ApexOODAState:
            assert isinstance(state.value, str)

    def test_enum_membership(self):
        """Construction from string value should return the correct member."""
        assert ApexOODAState("observe") is ApexOODAState.OBSERVE
        assert ApexOODAState("sleep") is ApexOODAState.SLEEP


# ---------------------------------------------------------------------------
# 2. TestObservation — Dataclass defaults
# ---------------------------------------------------------------------------
class TestObservation:
    """Verify Observation dataclass defaults and construction."""

    def test_default_market_readings_empty(self):
        obs = Observation()
        assert obs.market_readings == []

    def test_default_swarm_health_empty(self):
        obs = Observation()
        assert obs.swarm_health == {}

    def test_default_social_metrics_empty(self):
        obs = Observation()
        assert obs.social_metrics == {}

    def test_timestamp_is_utc(self):
        obs = Observation()
        assert obs.timestamp.tzinfo is not None
        assert obs.timestamp.tzinfo == timezone.utc

    def test_custom_values(self):
        reading = MarketSensorReading(symbol="TEST/USD", snr_score=0.92)
        ts = datetime(2026, 1, 1, tzinfo=timezone.utc)
        obs = Observation(
            market_readings=[reading],
            swarm_health={"svc1": "ok"},
            social_metrics={"trust": 0.8},
            timestamp=ts,
        )
        assert len(obs.market_readings) == 1
        assert obs.swarm_health["svc1"] == "ok"
        assert obs.social_metrics["trust"] == 0.8
        assert obs.timestamp == ts


# ---------------------------------------------------------------------------
# 3. TestPrediction — Dataclass defaults
# ---------------------------------------------------------------------------
class TestPrediction:
    """Verify Prediction dataclass defaults."""

    def test_default_market_trends_empty(self):
        pred = Prediction()
        assert pred.market_trends == {}

    def test_default_workload_forecast_none(self):
        pred = Prediction()
        assert pred.workload_forecast is None

    def test_default_scaling_recommendation_none(self):
        pred = Prediction()
        assert pred.scaling_recommendation is None

    def test_default_confidence(self):
        pred = Prediction()
        assert pred.confidence == 0.5

    def test_default_got_result_none(self):
        pred = Prediction()
        assert pred.got_result is None

    def test_default_reasoning_path_empty(self):
        pred = Prediction()
        assert pred.reasoning_path == []


# ---------------------------------------------------------------------------
# 4. TestTeamPlan — Dataclass defaults
# ---------------------------------------------------------------------------
class TestTeamPlan:
    """Verify TeamPlan dataclass defaults."""

    def test_default_task_assignments_empty(self):
        plan = TeamPlan()
        assert plan.task_assignments == []

    def test_default_collaborations_empty(self):
        plan = TeamPlan()
        assert plan.collaborations == []

    def test_default_selected_agents_empty(self):
        plan = TeamPlan()
        assert plan.selected_agents == []


# ---------------------------------------------------------------------------
# 5. TestDecision — Dataclass fields and defaults
# ---------------------------------------------------------------------------
class TestDecision:
    """Verify Decision dataclass fields."""

    @pytest.fixture
    def goal(self):
        return MarketGoal(
            goal_id="g-1",
            ihsan_score=0.97,
            autonomy_level=AutonomyLevel.AUTOLOW,
            snr_score=0.91,
        )

    def test_required_fields(self, goal):
        d = Decision(
            goal=goal,
            ihsan_score=0.97,
            autonomy_level=AutonomyLevel.AUTOLOW,
            requires_approval=False,
        )
        assert d.goal is goal
        assert d.ihsan_score == 0.97
        assert d.autonomy_level == AutonomyLevel.AUTOLOW
        assert d.requires_approval is False

    def test_approved_defaults_false(self, goal):
        d = Decision(
            goal=goal,
            ihsan_score=0.97,
            autonomy_level=AutonomyLevel.AUTOLOW,
            requires_approval=True,
        )
        assert d.approved is False

    def test_rejection_reason_defaults_none(self, goal):
        d = Decision(
            goal=goal,
            ihsan_score=0.97,
            autonomy_level=AutonomyLevel.AUTOLOW,
            requires_approval=True,
        )
        assert d.rejection_reason is None

    def test_custom_approved_true(self, goal):
        d = Decision(
            goal=goal,
            ihsan_score=0.97,
            autonomy_level=AutonomyLevel.AUTOLOW,
            requires_approval=False,
            approved=True,
        )
        assert d.approved is True

    def test_custom_rejection_reason(self, goal):
        d = Decision(
            goal=goal,
            ihsan_score=0.80,
            autonomy_level=AutonomyLevel.OBSERVER,
            requires_approval=True,
            rejection_reason="below threshold",
        )
        assert d.rejection_reason == "below threshold"


# ---------------------------------------------------------------------------
# 6. TestOutcome — Dataclass defaults
# ---------------------------------------------------------------------------
class TestOutcome:
    """Verify Outcome dataclass defaults."""

    @pytest.fixture
    def decision(self):
        goal = MarketGoal(
            goal_id="g-o",
            ihsan_score=0.96,
            autonomy_level=AutonomyLevel.AUTOLOW,
            snr_score=0.90,
        )
        return Decision(
            goal=goal,
            ihsan_score=0.96,
            autonomy_level=AutonomyLevel.AUTOLOW,
            requires_approval=False,
            approved=True,
        )

    def test_default_value_zero(self, decision):
        o = Outcome(decision=decision, success=True)
        assert o.value == 0.0

    def test_default_agents_used_empty(self, decision):
        o = Outcome(decision=decision, success=True)
        assert o.agents_used == []

    def test_default_execution_time_zero(self, decision):
        o = Outcome(decision=decision, success=True)
        assert o.execution_time_ms == 0.0

    def test_default_error_none(self, decision):
        o = Outcome(decision=decision, success=False)
        assert o.error is None

    def test_default_giants_attribution_empty(self, decision):
        o = Outcome(decision=decision, success=True)
        assert o.giants_attribution == []


# ---------------------------------------------------------------------------
# 7. TestLazyProperties — Lazy initialization of runtime engines
# ---------------------------------------------------------------------------
class TestLazyProperties:
    """Verify lazy initialization of SNRMaximizer, GoTBridge, GiantsRegistry."""

    @pytest.fixture
    def entity(self):
        return create_apex_entity("lazy-test")

    def test_snr_maximizer_none_initially(self, entity):
        assert entity._snr_maximizer is None

    def test_snr_maximizer_created_on_access(self, entity):
        maximizer = entity.snr_maximizer
        assert maximizer is not None
        assert entity._snr_maximizer is maximizer

    def test_snr_maximizer_same_instance_on_second_access(self, entity):
        first = entity.snr_maximizer
        second = entity.snr_maximizer
        assert first is second

    def test_got_bridge_none_initially(self, entity):
        assert entity._got_bridge is None

    def test_got_bridge_created_on_access(self, entity):
        bridge = entity.got_bridge
        assert bridge is not None
        assert entity._got_bridge is bridge

    def test_got_bridge_same_instance(self, entity):
        first = entity.got_bridge
        second = entity.got_bridge
        assert first is second

    def test_giants_registry_none_initially(self, entity):
        assert entity._giants_registry is None

    def test_giants_registry_created_on_access(self, entity):
        registry = entity.giants_registry
        assert registry is not None
        assert entity._giants_registry is registry

    def test_giants_registry_same_instance(self, entity):
        first = entity.giants_registry
        second = entity.giants_registry
        assert first is second

    def test_get_giants_attribution_returns_six_strings(self, entity):
        attrs = entity.get_giants_attribution("test_method")
        assert isinstance(attrs, list)
        assert len(attrs) == 6
        for attr in attrs:
            assert isinstance(attr, str)
            assert len(attr) > 0


# ---------------------------------------------------------------------------
# 8. TestDecidePhaseDeep — Thorough _decide phase testing
# ---------------------------------------------------------------------------
class TestDecidePhaseDeep:
    """Deep tests for the _decide phase logic."""

    @pytest.fixture
    def entity(self):
        return create_apex_entity("decide-test")

    @pytest.mark.asyncio
    async def test_empty_goals_returns_empty(self, entity):
        decisions = await entity._decide([])
        assert decisions == []

    @pytest.mark.asyncio
    async def test_goal_below_ihsan_filtered(self, entity):
        goal = MarketGoal(
            goal_id="low-ihsan",
            ihsan_score=0.80,
            autonomy_level=AutonomyLevel.AUTOLOW,
            snr_score=0.90,
        )
        decisions = await entity._decide([goal])
        assert len(decisions) == 0

    @pytest.mark.asyncio
    async def test_goal_at_threshold_accepted(self, entity):
        """Goal with ihsan_score exactly at threshold should pass."""
        goal = MarketGoal(
            goal_id="at-threshold",
            ihsan_score=0.95,
            autonomy_level=AutonomyLevel.AUTOLOW,
            snr_score=0.90,
        )
        decisions = await entity._decide([goal])
        assert len(decisions) == 1

    @pytest.mark.asyncio
    async def test_multiple_goals_some_filtered(self, entity):
        goals = [
            MarketGoal(
                goal_id="pass-1",
                ihsan_score=0.97,
                autonomy_level=AutonomyLevel.AUTOLOW,
                snr_score=0.91,
            ),
            MarketGoal(
                goal_id="fail-1",
                ihsan_score=0.80,
                autonomy_level=AutonomyLevel.OBSERVER,
                snr_score=0.70,
            ),
            MarketGoal(
                goal_id="pass-2",
                ihsan_score=0.99,
                autonomy_level=AutonomyLevel.AUTOMEDIUM,
                snr_score=0.96,
            ),
        ]
        decisions = await entity._decide(goals)
        assert len(decisions) == 2
        assert all(d.ihsan_score >= 0.95 for d in decisions)

    @pytest.mark.asyncio
    async def test_suggester_requires_approval(self, entity):
        goal = MarketGoal(
            goal_id="suggest",
            ihsan_score=0.96,
            autonomy_level=AutonomyLevel.SUGGESTER,
            snr_score=0.88,
        )
        decisions = await entity._decide([goal])
        assert len(decisions) == 1
        assert decisions[0].requires_approval is True
        assert decisions[0].approved is False

    @pytest.mark.asyncio
    async def test_observer_requires_approval(self, entity):
        goal = MarketGoal(
            goal_id="observe",
            ihsan_score=0.96,
            autonomy_level=AutonomyLevel.OBSERVER,
            snr_score=0.86,
        )
        decisions = await entity._decide([goal])
        assert len(decisions) == 1
        assert decisions[0].requires_approval is True
        assert decisions[0].approved is False

    @pytest.mark.asyncio
    async def test_autolow_auto_approved(self, entity):
        goal = MarketGoal(
            goal_id="auto-low",
            ihsan_score=0.97,
            autonomy_level=AutonomyLevel.AUTOLOW,
            snr_score=0.91,
        )
        decisions = await entity._decide([goal])
        assert len(decisions) == 1
        assert decisions[0].requires_approval is False
        assert decisions[0].approved is True

    @pytest.mark.asyncio
    async def test_automedium_auto_approved(self, entity):
        goal = MarketGoal(
            goal_id="auto-med",
            ihsan_score=0.98,
            autonomy_level=AutonomyLevel.AUTOMEDIUM,
            snr_score=0.96,
        )
        decisions = await entity._decide([goal])
        assert decisions[0].approved is True

    @pytest.mark.asyncio
    async def test_ihsan_history_tracked(self, entity):
        goal = MarketGoal(
            goal_id="track",
            ihsan_score=0.97,
            autonomy_level=AutonomyLevel.AUTOLOW,
            snr_score=0.91,
        )
        await entity._decide([goal])
        assert 0.97 in entity._ihsan_history

    @pytest.mark.asyncio
    async def test_ihsan_history_capped_at_100(self, entity):
        """Ihsan history should not exceed 100 entries."""
        goals = [
            MarketGoal(
                goal_id=f"cap-{i}",
                ihsan_score=0.96,
                autonomy_level=AutonomyLevel.AUTOLOW,
                snr_score=0.91,
            )
            for i in range(110)
        ]
        await entity._decide(goals)
        assert len(entity._ihsan_history) <= 100


# ---------------------------------------------------------------------------
# 9. TestActPhaseDeep — Thorough _act phase testing
# ---------------------------------------------------------------------------
class TestActPhaseDeep:
    """Deep tests for the _act phase logic."""

    @pytest.fixture
    def entity(self):
        return create_apex_entity("act-test")

    @pytest.fixture
    def mock_agent(self):
        """A mock ScoredAgent returned by select_agent_for_task."""
        from core.sovereign.team_planner import AgentRole

        return ScoredAgent(
            agent_id="pat:master_reasoner",
            role=AgentRole.MASTER_REASONER,
            capability_score=1.0,
            trust_score=0.8,
            combined_score=0.9,
            capabilities={"reasoning", "execution"},
        )

    def _make_decision(
        self,
        goal_id: str = "g-act",
        ihsan_score: float = 0.97,
        snr_score: float = 0.91,
        autonomy_level: AutonomyLevel = AutonomyLevel.AUTOLOW,
        approved: bool = True,
    ) -> Decision:
        goal = MarketGoal(
            goal_id=goal_id,
            ihsan_score=ihsan_score,
            autonomy_level=autonomy_level,
            snr_score=snr_score,
            estimated_value=100.0,
        )
        return Decision(
            goal=goal,
            ihsan_score=ihsan_score,
            autonomy_level=autonomy_level,
            requires_approval=not approved,
            approved=approved,
        )

    @pytest.mark.asyncio
    async def test_empty_decisions_returns_empty(self, entity):
        outcomes = await entity._act([], TeamPlan())
        assert outcomes == []

    @pytest.mark.asyncio
    async def test_unapproved_decisions_skipped(self, entity):
        d = self._make_decision(approved=False)
        outcomes = await entity._act([d], TeamPlan())
        assert len(outcomes) == 0

    @pytest.mark.asyncio
    async def test_success_when_ihsan_and_snr_high(self, entity, mock_agent):
        d = self._make_decision(ihsan_score=0.97, snr_score=0.91)
        with patch.object(
            entity.social_bridge,
            "select_agent_for_task",
            return_value=mock_agent,
        ):
            outcomes = await entity._act([d], TeamPlan())
        assert len(outcomes) == 1
        assert outcomes[0].success is True
        assert outcomes[0].value == 100.0

    @pytest.mark.asyncio
    async def test_failure_when_ihsan_below_095(self, entity, mock_agent):
        """Success requires decision.ihsan_score >= 0.95."""
        d = self._make_decision(ihsan_score=0.94, snr_score=0.91)
        with patch.object(
            entity.social_bridge,
            "select_agent_for_task",
            return_value=mock_agent,
        ):
            outcomes = await entity._act([d], TeamPlan())
        assert len(outcomes) == 1
        assert outcomes[0].success is False
        assert outcomes[0].value == 0.0

    @pytest.mark.asyncio
    async def test_failure_when_snr_below_085(self, entity, mock_agent):
        """Success requires decision.goal.snr_score >= 0.85."""
        d = self._make_decision(ihsan_score=0.97, snr_score=0.84)
        with patch.object(
            entity.social_bridge,
            "select_agent_for_task",
            return_value=mock_agent,
        ):
            outcomes = await entity._act([d], TeamPlan())
        assert len(outcomes) == 1
        assert outcomes[0].success is False

    @pytest.mark.asyncio
    async def test_actions_taken_incremented(self, entity, mock_agent):
        d = self._make_decision()
        assert entity.metrics["actions_taken"] == 0
        with patch.object(
            entity.social_bridge,
            "select_agent_for_task",
            return_value=mock_agent,
        ):
            await entity._act([d], TeamPlan())
        assert entity.metrics["actions_taken"] == 1

    @pytest.mark.asyncio
    async def test_autonomous_actions_incremented_for_autolow(self, entity, mock_agent):
        d = self._make_decision(autonomy_level=AutonomyLevel.AUTOLOW)
        with patch.object(
            entity.social_bridge,
            "select_agent_for_task",
            return_value=mock_agent,
        ):
            await entity._act([d], TeamPlan())
        assert entity.metrics["autonomous_actions"] == 1

    @pytest.mark.asyncio
    async def test_autonomous_actions_incremented_for_automedium(self, entity, mock_agent):
        d = self._make_decision(autonomy_level=AutonomyLevel.AUTOMEDIUM)
        with patch.object(
            entity.social_bridge,
            "select_agent_for_task",
            return_value=mock_agent,
        ):
            await entity._act([d], TeamPlan())
        assert entity.metrics["autonomous_actions"] == 1

    @pytest.mark.asyncio
    async def test_autonomous_actions_not_incremented_for_suggester(self, entity, mock_agent):
        d = self._make_decision(autonomy_level=AutonomyLevel.SUGGESTER)
        with patch.object(
            entity.social_bridge,
            "select_agent_for_task",
            return_value=mock_agent,
        ):
            await entity._act([d], TeamPlan())
        assert entity.metrics["autonomous_actions"] == 0

    @pytest.mark.asyncio
    async def test_success_history_tracked(self, entity, mock_agent):
        d = self._make_decision(ihsan_score=0.97, snr_score=0.91)
        with patch.object(
            entity.social_bridge,
            "select_agent_for_task",
            return_value=mock_agent,
        ):
            await entity._act([d], TeamPlan())
        assert len(entity._success_history) == 1
        assert entity._success_history[0] is True

    @pytest.mark.asyncio
    async def test_success_history_capped_at_100(self, entity, mock_agent):
        """Success path caps _success_history at 100 entries."""
        with patch.object(
            entity.social_bridge,
            "select_agent_for_task",
            return_value=mock_agent,
        ):
            for i in range(110):
                d = self._make_decision(goal_id=f"cap-{i}")
                await entity._act([d], TeamPlan())
        assert len(entity._success_history) <= 100

    @pytest.mark.asyncio
    async def test_error_path_appends_false_to_history(self, entity):
        """Error path appends False to _success_history (no cap)."""
        d = self._make_decision()
        # select_agent_for_task will fail naturally (no agent has both caps)
        outcomes = await entity._act([d], TeamPlan())
        assert len(outcomes) == 1
        assert outcomes[0].success is False
        # Error path always appends False
        assert entity._success_history[-1] is False

    @pytest.mark.asyncio
    async def test_giants_attribution_added(self, entity, mock_agent):
        d = self._make_decision()
        with patch.object(
            entity.social_bridge,
            "select_agent_for_task",
            return_value=mock_agent,
        ):
            outcomes = await entity._act([d], TeamPlan())
        assert len(outcomes) == 1
        assert len(outcomes[0].giants_attribution) == 6
        for attr in outcomes[0].giants_attribution:
            assert isinstance(attr, str)

    @pytest.mark.asyncio
    async def test_giants_attribution_on_error(self, entity):
        """Giants attribution should be added even when execution fails."""
        d = self._make_decision()
        with patch.object(
            entity.social_bridge,
            "select_agent_for_task",
            side_effect=RuntimeError("agent selection failed"),
        ):
            outcomes = await entity._act([d], TeamPlan())
        assert len(outcomes) == 1
        assert outcomes[0].success is False
        assert len(outcomes[0].giants_attribution) == 6

    @pytest.mark.asyncio
    async def test_error_handling_in_act(self, entity):
        """If agent selection fails, an error outcome is produced."""
        d = self._make_decision()
        with patch.object(
            entity.social_bridge,
            "select_agent_for_task",
            side_effect=RuntimeError("agent selection failed"),
        ):
            outcomes = await entity._act([d], TeamPlan())
        assert len(outcomes) == 1
        assert outcomes[0].success is False
        assert "agent selection failed" in outcomes[0].error

    @pytest.mark.asyncio
    async def test_execution_time_tracked(self, entity, mock_agent):
        d = self._make_decision()
        with patch.object(
            entity.social_bridge,
            "select_agent_for_task",
            return_value=mock_agent,
        ):
            outcomes = await entity._act([d], TeamPlan())
        assert outcomes[0].execution_time_ms >= 0.0

    @pytest.mark.asyncio
    async def test_agents_used_populated(self, entity, mock_agent):
        d = self._make_decision()
        with patch.object(
            entity.social_bridge,
            "select_agent_for_task",
            return_value=mock_agent,
        ):
            outcomes = await entity._act([d], TeamPlan())
        assert len(outcomes[0].agents_used) == 1
        assert outcomes[0].agents_used[0] == "pat:master_reasoner"


# ---------------------------------------------------------------------------
# 10. TestReflectPhase — _reflect metrics updates
# ---------------------------------------------------------------------------
class TestReflectPhase:
    """Tests for the _reflect phase that updates metrics."""

    @pytest.fixture
    def entity(self):
        return create_apex_entity("reflect-test")

    @pytest.mark.asyncio
    async def test_updates_cycle_count(self, entity):
        entity.cycle_count = 5
        await entity._reflect()
        assert entity.metrics["cycles"] == 5

    @pytest.mark.asyncio
    async def test_updates_ihsan_average(self, entity):
        entity._ihsan_history = [0.96, 0.97, 0.98]
        await entity._reflect()
        expected_avg = (0.96 + 0.97 + 0.98) / 3
        assert abs(entity.metrics["ihsan_average"] - expected_avg) < 1e-9

    @pytest.mark.asyncio
    async def test_updates_snr_average(self, entity):
        entity._snr_history = [0.88, 0.92, 0.90]
        await entity._reflect()
        expected_avg = (0.88 + 0.92 + 0.90) / 3
        assert abs(entity.metrics["snr_average"] - expected_avg) < 1e-9

    @pytest.mark.asyncio
    async def test_updates_success_rate(self, entity):
        entity._success_history = [True, True, False, True]
        await entity._reflect()
        assert abs(entity.metrics["success_rate"] - 0.75) < 1e-9

    @pytest.mark.asyncio
    async def test_empty_histories_no_error(self, entity):
        """Reflect should not raise when histories are empty."""
        entity._ihsan_history = []
        entity._snr_history = []
        entity._success_history = []
        await entity._reflect()
        assert entity.metrics["ihsan_average"] == 0.0
        assert entity.metrics["snr_average"] == 0.0
        assert entity.metrics["success_rate"] == 0.0

    @pytest.mark.asyncio
    async def test_snr_floor_increases_when_ihsan_drops(self, entity):
        """When ihsan_average < threshold - 0.05, snr_floor should increase."""
        initial_snr_floor = entity.snr_floor
        # Set ihsan average well below threshold - 0.05
        entity._ihsan_history = [0.85, 0.86, 0.87]
        await entity._reflect()
        assert entity.snr_floor > initial_snr_floor
        assert entity.snr_floor == initial_snr_floor + 0.02

    @pytest.mark.asyncio
    async def test_snr_floor_capped_at_095(self, entity):
        """snr_floor should never exceed 0.95."""
        entity.snr_floor = 0.94
        entity._ihsan_history = [0.85, 0.86, 0.87]
        await entity._reflect()
        assert entity.snr_floor <= 0.95

    @pytest.mark.asyncio
    async def test_snr_floor_not_increased_when_ihsan_ok(self, entity):
        """snr_floor stays the same if ihsan_average is near threshold."""
        initial = entity.snr_floor
        entity._ihsan_history = [0.96, 0.97, 0.98]
        await entity._reflect()
        assert entity.snr_floor == initial


# ---------------------------------------------------------------------------
# 11. TestLearnPhase — _learn trust updates
# ---------------------------------------------------------------------------
class TestLearnPhase:
    """Tests for the _learn phase that updates social trust."""

    @pytest.fixture
    def entity(self):
        return create_apex_entity("learn-test")

    @pytest.mark.asyncio
    async def test_empty_outcomes_no_error(self, entity):
        await entity._learn([])
        # Should not raise

    @pytest.mark.asyncio
    async def test_calls_report_task_outcome_for_each_agent(self, entity):
        goal = MarketGoal(
            goal_id="learn-g",
            ihsan_score=0.97,
            autonomy_level=AutonomyLevel.AUTOLOW,
            snr_score=0.91,
        )
        d = Decision(
            goal=goal,
            ihsan_score=0.97,
            autonomy_level=AutonomyLevel.AUTOLOW,
            requires_approval=False,
            approved=True,
        )
        outcome = Outcome(
            decision=d,
            success=True,
            value=50.0,
            agents_used=["pat:master_reasoner", "pat:data_analyzer"],
        )
        with patch.object(
            entity.social_bridge,
            "report_task_outcome",
            new_callable=AsyncMock,
        ) as mock_report:
            await entity._learn([outcome])
            assert mock_report.call_count == 2
            # Verify both agent_ids were called
            called_agents = {call.kwargs["agent_id"] for call in mock_report.call_args_list}
            assert "pat:master_reasoner" in called_agents
            assert "pat:data_analyzer" in called_agents

    @pytest.mark.asyncio
    async def test_learn_multiple_outcomes(self, entity):
        goal = MarketGoal(goal_id="lm", ihsan_score=0.97, snr_score=0.91)
        d = Decision(
            goal=goal,
            ihsan_score=0.97,
            autonomy_level=AutonomyLevel.AUTOLOW,
            requires_approval=False,
            approved=True,
        )
        outcomes = [
            Outcome(decision=d, success=True, value=10.0, agents_used=["agent-a"]),
            Outcome(decision=d, success=False, value=0.0, agents_used=["agent-b"]),
        ]
        with patch.object(
            entity.social_bridge,
            "report_task_outcome",
            new_callable=AsyncMock,
        ) as mock_report:
            await entity._learn(outcomes)
            assert mock_report.call_count == 2

    @pytest.mark.asyncio
    async def test_learn_outcome_with_no_agents(self, entity):
        goal = MarketGoal(goal_id="no-agents", ihsan_score=0.97, snr_score=0.91)
        d = Decision(
            goal=goal,
            ihsan_score=0.97,
            autonomy_level=AutonomyLevel.AUTOLOW,
            requires_approval=False,
            approved=True,
        )
        outcome = Outcome(decision=d, success=True, agents_used=[])
        with patch.object(
            entity.social_bridge,
            "report_task_outcome",
            new_callable=AsyncMock,
        ) as mock_report:
            await entity._learn([outcome])
            mock_report.assert_not_called()

    @pytest.mark.asyncio
    async def test_learn_passes_success_and_value(self, entity):
        goal = MarketGoal(goal_id="vals", ihsan_score=0.97, snr_score=0.91)
        d = Decision(
            goal=goal,
            ihsan_score=0.97,
            autonomy_level=AutonomyLevel.AUTOLOW,
            requires_approval=False,
            approved=True,
        )
        outcome = Outcome(
            decision=d, success=True, value=42.0, agents_used=["pat:master_reasoner"]
        )
        with patch.object(
            entity.social_bridge,
            "report_task_outcome",
            new_callable=AsyncMock,
        ) as mock_report:
            await entity._learn([outcome])
            mock_report.assert_called_once_with(
                agent_id="pat:master_reasoner",
                task_id="vals",
                success=True,
                value=42.0,
            )


# ---------------------------------------------------------------------------
# 12. TestHandleCycleError — Error handling
# ---------------------------------------------------------------------------
class TestHandleCycleError:
    """Tests for _handle_cycle_error."""

    @pytest.fixture
    def entity(self):
        return create_apex_entity("error-test")

    @pytest.mark.asyncio
    async def test_logs_error(self, entity, caplog):
        with caplog.at_level(logging.ERROR):
            await entity._handle_cycle_error(RuntimeError("boom"))
        assert any("boom" in record.message for record in caplog.records)

    @pytest.mark.asyncio
    async def test_sleeps_one_second(self, entity):
        with patch("core.sovereign.apex_sovereign.asyncio.sleep", new_callable=AsyncMock) as mock_sleep:
            await entity._handle_cycle_error(RuntimeError("oops"))
            mock_sleep.assert_called_once_with(1)

    @pytest.mark.asyncio
    async def test_does_not_raise(self, entity):
        """_handle_cycle_error should swallow the error, not re-raise."""
        # Should complete without raising
        await entity._handle_cycle_error(ValueError("test error"))


# ---------------------------------------------------------------------------
# 13. TestStatusDeep — Comprehensive status dictionary
# ---------------------------------------------------------------------------
class TestStatusDeep:
    """Deep tests for the status() method."""

    @pytest.fixture
    def entity(self):
        return create_apex_entity("status-test")

    def test_standing_on_giants_has_six_entries(self, entity):
        status = entity.status()
        assert len(status["standing_on_giants"]) == 6

    def test_standing_on_giants_are_strings(self, entity):
        status = entity.status()
        for entry in status["standing_on_giants"]:
            assert isinstance(entry, str)

    def test_runtime_engines_empty_when_not_initialized(self, entity):
        status = entity.status()
        assert "runtime_engines" not in status

    def test_runtime_engines_populated_after_snr_access(self, entity):
        _ = entity.snr_maximizer  # trigger lazy init
        status = entity.status()
        assert "runtime_engines" in status
        assert "snr_maximizer" in status["runtime_engines"]

    def test_runtime_engines_populated_after_giants_access(self, entity):
        _ = entity.giants_registry  # trigger lazy init
        status = entity.status()
        assert "runtime_engines" in status
        assert "giants_registry" in status["runtime_engines"]

    def test_all_expected_top_level_keys(self, entity):
        status = entity.status()
        expected_keys = {
            "node_id",
            "running",
            "current_state",
            "cycle_count",
            "ihsan_threshold",
            "snr_floor",
            "metrics",
            "subsystems",
            "standing_on_giants",
        }
        assert expected_keys.issubset(set(status.keys()))

    def test_subsystems_has_required_keys(self, entity):
        status = entity.status()
        assert "apex" in status["subsystems"]
        assert "social" in status["subsystems"]
        assert "swarm" in status["subsystems"]

    def test_metrics_dict_has_expected_keys(self, entity):
        status = entity.status()
        expected_metric_keys = {
            "cycles",
            "actions_taken",
            "autonomous_actions",
            "ihsan_average",
            "snr_average",
            "success_rate",
        }
        assert expected_metric_keys == set(status["metrics"].keys())


# ---------------------------------------------------------------------------
# 14. TestPredictPhaseDeep — _predict edge cases and logic
# ---------------------------------------------------------------------------
class TestPredictPhaseDeep:
    """Deep tests for the _predict phase."""

    @pytest.fixture
    def entity(self):
        return create_apex_entity("predict-test")

    @pytest.mark.asyncio
    async def test_none_observation_returns_empty_prediction(self, entity):
        pred = await entity._predict(None)
        assert pred.market_trends == {}
        assert pred.confidence == 0.5  # default, never overwritten

    @pytest.mark.asyncio
    async def test_readings_below_snr_floor_excluded_from_trends(self, entity):
        low_snr_reading = MarketSensorReading(
            symbol="LOW/USD",
            snr_score=0.50,
            sensor_type=MarketSensorType.TRADING_SIGNAL,
        )
        obs = Observation(market_readings=[low_snr_reading])
        pred = await entity._predict(obs)
        assert "LOW/USD" not in pred.market_trends

    @pytest.mark.asyncio
    async def test_readings_above_snr_floor_included_in_trends(self, entity):
        high_snr = MarketSensorReading(
            symbol="HIGH/USD",
            snr_score=0.92,
            sensor_type=MarketSensorType.TRADING_SIGNAL,
            value={"signal_type": "buy"},
        )
        obs = Observation(market_readings=[high_snr])
        pred = await entity._predict(obs)
        assert "HIGH/USD" in pred.market_trends
        assert pred.market_trends["HIGH/USD"]["snr"] == 0.92

    @pytest.mark.asyncio
    async def test_workload_forecast_from_unhealthy_swarm(self, entity):
        """When healthy ratio < 0.8, scaling_recommendation = 'scale_up'."""
        health = {
            "svc1": "HealthStatus.HEALTHY",
            "svc2": "HealthStatus.UNHEALTHY",
            "svc3": "HealthStatus.UNHEALTHY",
            "svc4": "HealthStatus.UNHEALTHY",
            "svc5": "HealthStatus.UNHEALTHY",
        }
        obs = Observation(swarm_health=health)
        pred = await entity._predict(obs)
        assert pred.scaling_recommendation == "scale_up"

    @pytest.mark.asyncio
    async def test_workload_forecast_optimal(self, entity):
        """When healthy ratio > 0.95, workload_forecast is 'optimal'."""
        health = {
            f"svc{i}": "HealthStatus.HEALTHY" for i in range(20)
        }
        obs = Observation(swarm_health=health)
        pred = await entity._predict(obs)
        assert pred.workload_forecast == {"status": "optimal"}

    @pytest.mark.asyncio
    async def test_confidence_override_with_readings(self, entity):
        """Confidence is always 0.7 when market_readings is non-empty."""
        reading = MarketSensorReading(symbol="ANY/USD", snr_score=0.92)
        obs = Observation(market_readings=[reading])
        pred = await entity._predict(obs)
        assert pred.confidence == 0.7

    @pytest.mark.asyncio
    async def test_confidence_override_without_readings(self, entity):
        """Confidence is always 0.3 when market_readings is empty."""
        obs = Observation(market_readings=[])
        pred = await entity._predict(obs)
        assert pred.confidence == 0.3

    @pytest.mark.asyncio
    async def test_got_not_triggered_with_fewer_than_two_high_snr(self, entity):
        """GoT bridge should not be invoked with < 2 readings at SNR >= 0.90."""
        reading = MarketSensorReading(symbol="ONE/USD", snr_score=0.92)
        obs = Observation(market_readings=[reading])
        pred = await entity._predict(obs)
        # got_result should remain None since only 1 high-SNR reading
        assert pred.got_result is None


# ---------------------------------------------------------------------------
# 15. TestCreateApexEntity — Factory function
# ---------------------------------------------------------------------------
class TestCreateApexEntity:
    """Tests for the create_apex_entity factory function."""

    def test_factory_creates_entity(self):
        entity = create_apex_entity("factory-test")
        assert isinstance(entity, ApexSovereignEntity)
        assert entity.node_id == "factory-test"

    def test_default_node_id(self):
        entity = create_apex_entity()
        assert entity.node_id == "node-0"

    def test_factory_entity_is_not_running(self):
        entity = create_apex_entity("factory-check")
        assert entity._running is False
        assert entity.current_state == ApexOODAState.SLEEP


# ---------------------------------------------------------------------------
# 16. TestStartStop — Start/stop edge cases
# ---------------------------------------------------------------------------
class TestStartStop:
    """Edge cases for start/stop lifecycle."""

    @pytest.mark.asyncio
    async def test_double_start_is_idempotent(self):
        entity = create_apex_entity("double-start")
        await entity.start()
        try:
            await entity.start()  # should not raise
            assert entity._running is True
        finally:
            await entity.stop()

    @pytest.mark.asyncio
    async def test_stop_when_not_running_is_noop(self):
        entity = create_apex_entity("stop-noop")
        await entity.stop()  # should not raise
        assert entity._running is False

    @pytest.mark.asyncio
    async def test_state_transitions_on_start(self):
        entity = create_apex_entity("start-state")
        assert entity.current_state == ApexOODAState.SLEEP
        await entity.start()
        try:
            assert entity.current_state == ApexOODAState.OBSERVE
        finally:
            await entity.stop()

    @pytest.mark.asyncio
    async def test_state_returns_to_sleep_on_stop(self):
        entity = create_apex_entity("stop-state")
        await entity.start()
        await entity.stop()
        assert entity.current_state == ApexOODAState.SLEEP


# ---------------------------------------------------------------------------
# 17. TestOODALoopStateMachine — State machine transitions
# ---------------------------------------------------------------------------
class TestOODALoopStateMachine:
    """Verify the OODA loop state transitions."""

    @pytest.fixture
    def entity(self):
        return create_apex_entity("ooda-sm-test")

    @pytest.mark.asyncio
    async def test_observe_transitions_to_predict(self, entity):
        entity.current_state = ApexOODAState.OBSERVE
        await entity._observe()
        # The _run_ooda_loop sets state after calling _observe;
        # verify the method does not crash and returns Observation
        obs = await entity._observe()
        assert isinstance(obs, Observation)

    @pytest.mark.asyncio
    async def test_predict_transitions_returns_prediction(self, entity):
        obs = Observation()
        pred = await entity._predict(obs)
        assert isinstance(pred, Prediction)

    @pytest.mark.asyncio
    async def test_coordinate_returns_team_plan(self, entity):
        obs = Observation()
        pred = Prediction()
        plan = await entity._coordinate(obs, pred)
        assert isinstance(plan, TeamPlan)

    @pytest.mark.asyncio
    async def test_coordinate_with_none_observation_returns_empty_plan(self, entity):
        pred = Prediction()
        plan = await entity._coordinate(None, pred)
        assert plan.collaborations == []
        assert plan.task_assignments == []

    @pytest.mark.asyncio
    async def test_analyze_with_none_observation_returns_empty(self, entity):
        goals = await entity._analyze(None, None, None)
        assert goals == []


# ---------------------------------------------------------------------------
# 18. TestAnalyzePhase — _analyze internals
# ---------------------------------------------------------------------------
class TestAnalyzePhase:
    """Tests for the _analyze phase."""

    @pytest.fixture
    def entity(self):
        return create_apex_entity("analyze-test")

    @pytest.mark.asyncio
    async def test_readings_converted_to_goals(self, entity):
        reading = MarketSensorReading(
            sensor_type=MarketSensorType.TRADING_SIGNAL,
            symbol="TEST/USD",
            value={"signal_type": "buy", "strength": "strong", "expected_return": 0.05},
            snr_score=0.98,
        )
        obs = Observation(market_readings=[reading])
        goals = await entity._analyze(obs, Prediction(), TeamPlan())
        # Whether a goal is created depends on the Ihsan calculation in
        # process_market_reading; with 0.98 SNR it should pass
        assert isinstance(goals, list)

    @pytest.mark.asyncio
    async def test_snr_history_updated(self, entity):
        reading = MarketSensorReading(
            sensor_type=MarketSensorType.TRADING_SIGNAL,
            symbol="SNR-TRACK/USD",
            value={"signal_type": "buy", "strength": "strong", "expected_return": 0.05},
            snr_score=0.98,
        )
        obs = Observation(market_readings=[reading])
        goals = await entity._analyze(obs, Prediction(), TeamPlan())
        if goals:  # Goal created = SNR tracked
            assert 0.98 in entity._snr_history

    @pytest.mark.asyncio
    async def test_snr_history_capped_at_100(self, entity):
        """_snr_history should not exceed 100 entries."""
        # Pre-fill with 99 entries
        entity._snr_history = [0.90] * 99
        reading = MarketSensorReading(
            sensor_type=MarketSensorType.TRADING_SIGNAL,
            symbol="CAP/USD",
            value={"signal_type": "buy", "strength": "strong", "expected_return": 0.05},
            snr_score=0.98,
        )
        obs = Observation(market_readings=[reading])
        # Multiple readings to test cap
        obs.market_readings = [reading] * 5
        goals = await entity._analyze(obs, Prediction(), TeamPlan())
        assert len(entity._snr_history) <= 100


# ---------------------------------------------------------------------------
# 19. TestInitialization — Constructor parameter validation
# ---------------------------------------------------------------------------
class TestInitialization:
    """Tests for __init__ parameter handling."""

    def test_default_ihsan_threshold(self):
        entity = create_apex_entity("init-1")
        assert entity.ihsan_threshold == 0.95

    def test_default_snr_floor(self):
        entity = create_apex_entity("init-2")
        assert entity.snr_floor == 0.85

    def test_custom_ihsan_threshold(self):
        entity = ApexSovereignEntity("custom", ihsan_threshold=0.99)
        assert entity.ihsan_threshold == 0.99

    def test_custom_snr_floor(self):
        entity = ApexSovereignEntity("custom", snr_floor=0.90)
        assert entity.snr_floor == 0.90

    def test_custom_cycle_interval(self):
        entity = ApexSovereignEntity("custom", cycle_interval_ms=500)
        assert entity.cycle_interval_ms == 500

    def test_initial_metrics(self):
        entity = create_apex_entity("init-metrics")
        assert entity.metrics["cycles"] == 0
        assert entity.metrics["actions_taken"] == 0
        assert entity.metrics["autonomous_actions"] == 0
        assert entity.metrics["ihsan_average"] == 0.0
        assert entity.metrics["snr_average"] == 0.0
        assert entity.metrics["success_rate"] == 0.0

    def test_initial_histories_empty(self):
        entity = create_apex_entity("init-hist")
        assert entity._ihsan_history == []
        assert entity._snr_history == []
        assert entity._success_history == []

    def test_initial_cycle_count_zero(self):
        entity = create_apex_entity("init-cc")
        assert entity.cycle_count == 0

    def test_subsystems_initialized(self):
        entity = create_apex_entity("init-sub")
        assert entity.apex is not None
        assert entity.social_bridge is not None
        assert entity.market_muraqabah is not None
        assert entity.swarm is not None


# ---------------------------------------------------------------------------
# 20. TestObservePhaseDeep — _observe error handling and SNR filtering
# ---------------------------------------------------------------------------
class TestObservePhaseDeep:
    """Deep tests for the _observe phase."""

    @pytest.fixture
    def entity(self):
        return create_apex_entity("observe-deep")

    @pytest.mark.asyncio
    async def test_market_scan_failure_swallowed(self, entity):
        """If market scan throws, observation still returns with empty readings."""
        with patch.object(
            entity.market_muraqabah,
            "scan_financial_domain",
            new_callable=AsyncMock,
            side_effect=RuntimeError("scan failed"),
        ):
            obs = await entity._observe()
        assert obs.market_readings == []
        assert isinstance(obs, Observation)

    @pytest.mark.asyncio
    async def test_swarm_health_failure_swallowed(self, entity):
        with patch.object(
            entity.swarm,
            "check_all_health",
            new_callable=AsyncMock,
            side_effect=RuntimeError("health check failed"),
        ):
            obs = await entity._observe()
        assert obs.swarm_health == {}

    @pytest.mark.asyncio
    async def test_social_metrics_failure_swallowed(self, entity):
        with patch.object(
            entity.social_bridge,
            "get_network_metrics",
            side_effect=RuntimeError("metrics failed"),
        ):
            obs = await entity._observe()
        assert obs.social_metrics == {}

    @pytest.mark.asyncio
    async def test_observation_timestamp_set(self, entity):
        obs = await entity._observe()
        assert obs.timestamp is not None
        assert obs.timestamp.tzinfo == timezone.utc


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
