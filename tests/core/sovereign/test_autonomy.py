"""
Tests for the BIZRA Sovereign Engine - Autonomous Loop Module
=============================================================

Comprehensive test suite for the OBSERVE -> PREDICT -> COORDINATE ->
ANALYZE -> DECIDE -> ACT -> REFLECT -> LEARN cycle with DecisionGate
for Ihsan-compliant autonomous operation.

Standing on Giants: Boyd (OODA Loop, 1976), Deming (Plan-Do-Check-Act)

Covers:
  1. Enum correctness (LoopState, DecisionType, GateResult)
  2. Dataclass defaults and computed properties (SystemMetrics, etc.)
  3. DecisionGate evaluation logic (risk, confidence, rollback, health)
  4. AutonomousLoop full OODA cycle phases
  5. Extended OODA phases (Predict, Coordinate, Learn)
  6. Loop control (start, stop, pause, resume)
  7. Callback registration and invocation
  8. Factory function
"""

import asyncio
from collections import deque
from datetime import datetime
from typing import Any, Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from core.sovereign.autonomy import (
    AutonomousLoop,
    DecisionCandidate,
    DecisionGate,
    DecisionOutcome,
    DecisionType,
    GateResult,
    LoopState,
    SystemMetrics,
    create_autonomous_loop,
)


# =============================================================================
# ENUM TESTS
# =============================================================================


class TestLoopState:
    """Tests for the LoopState enum."""

    def test_all_twelve_states_exist(self):
        """LoopState must define exactly 12 states."""
        expected = {
            "IDLE",
            "OBSERVING",
            "PREDICTING",
            "COORDINATING",
            "ANALYZING",
            "PLANNING",
            "ACTING",
            "REFLECTING",
            "LEARNING",
            "ADAPTING",
            "PAUSED",
            "EMERGENCY",
        }
        actual = {s.name for s in LoopState}
        assert actual == expected

    def test_states_are_unique(self):
        """Each state must have a unique value."""
        values = [s.value for s in LoopState]
        assert len(values) == len(set(values))

    def test_idle_is_default_initial(self):
        """IDLE should be the first state defined."""
        assert LoopState.IDLE.value == 1


class TestDecisionType:
    """Tests for the DecisionType enum."""

    def test_all_six_types_exist(self):
        """DecisionType must define exactly 6 types."""
        expected = {
            "ROUTINE",
            "ADAPTIVE",
            "CORRECTIVE",
            "PREVENTIVE",
            "INNOVATIVE",
            "EMERGENCY",
        }
        actual = {t.name for t in DecisionType}
        assert actual == expected

    def test_emergency_is_valid_type(self):
        """EMERGENCY must be a valid DecisionType member."""
        assert isinstance(DecisionType.EMERGENCY, DecisionType)

    def test_types_are_unique(self):
        """Each type must have a unique value."""
        values = [t.value for t in DecisionType]
        assert len(values) == len(set(values))


class TestGateResult:
    """Tests for the GateResult enum."""

    def test_all_four_results_exist(self):
        """GateResult must define exactly 4 results."""
        expected = {"PASS", "REJECT", "DEFER", "ESCALATE"}
        actual = {r.name for r in GateResult}
        assert actual == expected

    def test_results_are_unique(self):
        """Each result must have a unique value."""
        values = [r.value for r in GateResult]
        assert len(values) == len(set(values))

    def test_pass_is_first(self):
        """PASS should be the first result defined."""
        assert GateResult.PASS.value == 1


# =============================================================================
# DATACLASS TESTS
# =============================================================================


class TestSystemMetrics:
    """Tests for the SystemMetrics dataclass."""

    def test_default_values(self):
        """Default SystemMetrics should have zero scores."""
        m = SystemMetrics()
        assert m.snr_score == 0.0
        assert m.ihsan_score == 0.0
        assert m.latency_ms == 0.0
        assert m.error_rate == 0.0
        assert m.throughput == 0.0
        assert m.memory_usage == 0.0
        assert m.active_tasks == 0
        assert m.pending_decisions == 0

    def test_timestamp_auto_generated(self):
        """Timestamp should be auto-generated on creation."""
        m = SystemMetrics()
        assert isinstance(m.timestamp, datetime)

    def test_health_score_perfect_metrics(self):
        """Perfect metrics should yield a high health score."""
        m = SystemMetrics(
            snr_score=0.99,
            ihsan_score=0.99,
            latency_ms=10.0,
            error_rate=0.0,
            memory_usage=0.1,
        )
        score = m.health_score()
        assert score > 0.95

    def test_health_score_zero_metrics(self):
        """Zero metrics should yield a low health score."""
        m = SystemMetrics()
        score = m.health_score()
        # snr_norm=0, ihsan_norm=0, error_norm=1.0, latency_norm=1.0, memory_norm=1.0
        # = 0.25*0 + 0.25*0 + 0.20*1 + 0.15*1 + 0.15*1 = 0.50
        assert score == pytest.approx(0.50, abs=0.01)

    def test_health_score_high_error_rate_penalizes(self):
        """High error rate should significantly reduce health score."""
        good = SystemMetrics(snr_score=0.95, ihsan_score=0.95, error_rate=0.0)
        bad = SystemMetrics(snr_score=0.95, ihsan_score=0.95, error_rate=0.5)
        assert bad.health_score() < good.health_score()

    def test_health_score_high_latency_penalizes(self):
        """High latency should reduce health score."""
        good = SystemMetrics(snr_score=0.95, ihsan_score=0.95, latency_ms=100)
        bad = SystemMetrics(snr_score=0.95, ihsan_score=0.95, latency_ms=4000)
        assert bad.health_score() < good.health_score()

    def test_health_score_high_memory_penalizes(self):
        """High memory usage should reduce health score."""
        good = SystemMetrics(snr_score=0.95, ihsan_score=0.95, memory_usage=0.2)
        bad = SystemMetrics(snr_score=0.95, ihsan_score=0.95, memory_usage=0.9)
        assert bad.health_score() < good.health_score()

    def test_is_healthy_with_default_threshold(self):
        """is_healthy should use 0.85 as default threshold."""
        healthy = SystemMetrics(
            snr_score=0.95, ihsan_score=0.95, error_rate=0.0, memory_usage=0.1
        )
        assert healthy.is_healthy() is True

    def test_is_healthy_with_custom_threshold(self):
        """is_healthy should respect a custom threshold."""
        m = SystemMetrics(snr_score=0.7, ihsan_score=0.7)
        # Score will be moderate; check both sides of a threshold
        assert m.is_healthy(threshold=0.3) is True
        assert m.is_healthy(threshold=0.99) is False

    def test_snr_normalization_caps_at_one(self):
        """SNR normalization should cap at 1.0 even when score >> 0.95."""
        m = SystemMetrics(snr_score=10.0, ihsan_score=0.0)
        # snr_norm = min(1.0, 10.0/0.95) = 1.0
        score = m.health_score()
        # max contribution from snr alone = 0.25*1.0 = 0.25
        assert score <= 1.0

    def test_ihsan_normalization_caps_at_one(self):
        """Ihsan normalization should cap at 1.0."""
        m = SystemMetrics(ihsan_score=5.0, snr_score=0.0)
        score = m.health_score()
        assert score <= 1.0

    def test_error_normalization_floor_at_zero(self):
        """Error normalization should floor at 0.0 for extreme error rates."""
        m = SystemMetrics(error_rate=1.0)
        # error_norm = max(0.0, 1.0 - 1.0*10) = max(0.0, -9.0) = 0.0
        score = m.health_score()
        assert score >= 0.0

    def test_latency_normalization_floor_at_zero(self):
        """Latency normalization should floor at 0.0 for extreme latency."""
        m = SystemMetrics(latency_ms=10000)
        # latency_norm = max(0.0, 1.0 - 10000/5000) = max(0.0, -1.0) = 0.0
        score = m.health_score()
        assert score >= 0.0

    def test_memory_normalization_floor_at_zero(self):
        """Memory normalization should floor at 0.0 for usage > 1.0."""
        m = SystemMetrics(memory_usage=2.0)
        # memory_norm = max(0.0, 1.0 - 2.0) = 0.0
        score = m.health_score()
        assert score >= 0.0

    def test_health_score_weighted_sum_is_correct(self):
        """Verify the weighted sum calculation explicitly."""
        m = SystemMetrics(
            snr_score=0.95,
            ihsan_score=0.95,
            error_rate=0.0,
            latency_ms=0.0,
            memory_usage=0.0,
        )
        # All norms = 1.0
        expected = 0.25 * 1.0 + 0.25 * 1.0 + 0.20 * 1.0 + 0.15 * 1.0 + 0.15 * 1.0
        assert m.health_score() == pytest.approx(expected, abs=1e-6)


class TestDecisionCandidate:
    """Tests for the DecisionCandidate dataclass."""

    def test_default_values(self):
        """Default DecisionCandidate should have sensible zero-values."""
        c = DecisionCandidate()
        assert c.decision_type == DecisionType.ROUTINE
        assert c.action == ""
        assert c.parameters == {}
        assert c.expected_impact == 0.0
        assert c.risk_score == 0.0
        assert c.confidence == 0.0
        assert c.rationale == ""
        assert c.rollback_plan == ""
        assert c.timeout_seconds == 60.0

    def test_id_is_generated(self):
        """ID should be auto-generated and non-empty."""
        c = DecisionCandidate()
        assert isinstance(c.id, str)
        assert len(c.id) == 8

    def test_ids_are_unique(self):
        """Each candidate should get a unique ID."""
        ids = {DecisionCandidate().id for _ in range(50)}
        assert len(ids) == 50

    def test_custom_parameters(self):
        """Custom parameters should be stored correctly."""
        params = {"target": 0.95, "mode": "aggressive"}
        c = DecisionCandidate(
            decision_type=DecisionType.CORRECTIVE,
            action="boost_snr",
            parameters=params,
            risk_score=0.2,
            confidence=0.9,
            rationale="SNR below threshold",
            rollback_plan="revert_boost",
        )
        assert c.decision_type == DecisionType.CORRECTIVE
        assert c.action == "boost_snr"
        assert c.parameters == params
        assert c.risk_score == 0.2
        assert c.confidence == 0.9

    def test_created_at_auto_generated(self):
        """created_at timestamp should be auto-generated."""
        c = DecisionCandidate()
        assert isinstance(c.created_at, datetime)


class TestDecisionOutcome:
    """Tests for the DecisionOutcome dataclass."""

    def test_default_values(self):
        """Default DecisionOutcome should have sensible zero-values."""
        o = DecisionOutcome()
        assert o.decision_id == ""
        assert o.gate_result == GateResult.PASS
        assert o.executed is False
        assert o.success is False
        assert o.actual_impact == 0.0
        assert o.execution_time_ms == 0.0
        assert o.error_message == ""
        assert o.metrics_before is None
        assert o.metrics_after is None

    def test_completed_at_auto_generated(self):
        """completed_at timestamp should be auto-generated."""
        o = DecisionOutcome()
        assert isinstance(o.completed_at, datetime)

    def test_custom_values(self):
        """DecisionOutcome should store custom values correctly."""
        metrics = SystemMetrics(snr_score=0.95)
        o = DecisionOutcome(
            decision_id="abc12345",
            gate_result=GateResult.PASS,
            executed=True,
            success=True,
            actual_impact=0.15,
            execution_time_ms=42.5,
            metrics_before=metrics,
        )
        assert o.decision_id == "abc12345"
        assert o.executed is True
        assert o.success is True
        assert o.actual_impact == 0.15
        assert o.execution_time_ms == 42.5
        assert o.metrics_before is metrics

    def test_error_message_storage(self):
        """Error messages should be stored when execution fails."""
        o = DecisionOutcome(
            decision_id="fail0001",
            executed=False,
            success=False,
            error_message="Execution timeout",
        )
        assert o.error_message == "Execution timeout"
        assert o.success is False

    def test_gate_result_reject(self):
        """DecisionOutcome should accept non-PASS gate results."""
        o = DecisionOutcome(gate_result=GateResult.REJECT)
        assert o.gate_result == GateResult.REJECT


# =============================================================================
# DECISION GATE TESTS
# =============================================================================


class TestDecisionGate:
    """Tests for the DecisionGate class."""

    def test_init_defaults(self):
        """Default DecisionGate should have standard thresholds."""
        gate = DecisionGate()
        assert gate.ihsan_threshold == 0.95
        assert gate.require_rollback is True
        assert gate.max_concurrent == 5
        assert gate.passed_count == 0
        assert gate.blocked_count == 0
        assert len(gate.active_decisions) == 0
        assert len(gate.decision_history) == 0

    def test_init_custom_params(self):
        """DecisionGate should accept custom initialization."""
        gate = DecisionGate(
            ihsan_threshold=0.90, require_rollback=False, max_concurrent=10
        )
        assert gate.ihsan_threshold == 0.90
        assert gate.require_rollback is False
        assert gate.max_concurrent == 10

    def test_risk_thresholds_has_all_decision_types(self):
        """RISK_THRESHOLDS must cover every DecisionType."""
        for dt in DecisionType:
            assert dt in DecisionGate.RISK_THRESHOLDS

    def test_confidence_requirements_has_all_decision_types(self):
        """CONFIDENCE_REQUIREMENTS must cover every DecisionType."""
        for dt in DecisionType:
            assert dt in DecisionGate.CONFIDENCE_REQUIREMENTS

    def test_risk_thresholds_values(self):
        """Verify specific risk threshold values."""
        assert DecisionGate.RISK_THRESHOLDS[DecisionType.ROUTINE] == 0.3
        assert DecisionGate.RISK_THRESHOLDS[DecisionType.EMERGENCY] == 0.7

    def test_confidence_requirements_values(self):
        """Verify specific confidence requirement values."""
        assert DecisionGate.CONFIDENCE_REQUIREMENTS[DecisionType.ROUTINE] == 0.7
        assert DecisionGate.CONFIDENCE_REQUIREMENTS[DecisionType.EMERGENCY] == 0.6
        assert DecisionGate.CONFIDENCE_REQUIREMENTS[DecisionType.INNOVATIVE] == 0.85

    @pytest.mark.asyncio
    async def test_evaluate_concurrent_limit_defers(self):
        """Exceeding max_concurrent should return DEFER."""
        gate = DecisionGate(max_concurrent=2)
        healthy_metrics = SystemMetrics(
            snr_score=0.95, ihsan_score=0.95, error_rate=0.0, memory_usage=0.1
        )
        # Fill up active decisions
        gate.active_decisions["a"] = DecisionCandidate()
        gate.active_decisions["b"] = DecisionCandidate()

        decision = DecisionCandidate(confidence=0.9, risk_score=0.1)
        result = await gate.evaluate(decision, healthy_metrics)
        assert result == GateResult.DEFER

    @pytest.mark.asyncio
    async def test_evaluate_high_risk_rejects(self):
        """Risk above threshold should return REJECT and increment blocked_count."""
        gate = DecisionGate()
        metrics = SystemMetrics(
            snr_score=0.95, ihsan_score=0.95, error_rate=0.0, memory_usage=0.1
        )
        # ROUTINE threshold is 0.3, so risk=0.5 should REJECT
        decision = DecisionCandidate(
            decision_type=DecisionType.ROUTINE,
            risk_score=0.5,
            confidence=0.9,
        )
        result = await gate.evaluate(decision, metrics)
        assert result == GateResult.REJECT
        assert gate.blocked_count == 1

    @pytest.mark.asyncio
    async def test_evaluate_multiple_rejects_increment_blocked_count(self):
        """Each rejection should increment blocked_count."""
        gate = DecisionGate()
        metrics = SystemMetrics(
            snr_score=0.95, ihsan_score=0.95, error_rate=0.0, memory_usage=0.1
        )
        for _ in range(3):
            d = DecisionCandidate(
                decision_type=DecisionType.ROUTINE,
                risk_score=0.5,
                confidence=0.9,
            )
            await gate.evaluate(d, metrics)
        assert gate.blocked_count == 3

    @pytest.mark.asyncio
    async def test_evaluate_low_confidence_defers(self):
        """Confidence below requirement should return DEFER."""
        gate = DecisionGate()
        metrics = SystemMetrics(
            snr_score=0.95, ihsan_score=0.95, error_rate=0.0, memory_usage=0.1
        )
        # ROUTINE requires 0.7 confidence
        decision = DecisionCandidate(
            decision_type=DecisionType.ROUTINE,
            risk_score=0.1,
            confidence=0.5,
        )
        result = await gate.evaluate(decision, metrics)
        assert result == GateResult.DEFER

    @pytest.mark.asyncio
    async def test_evaluate_missing_rollback_for_non_routine_defers(self):
        """Non-routine decision without rollback plan should DEFER."""
        gate = DecisionGate(require_rollback=True)
        metrics = SystemMetrics(
            snr_score=0.95, ihsan_score=0.95, error_rate=0.0, memory_usage=0.1
        )
        decision = DecisionCandidate(
            decision_type=DecisionType.ADAPTIVE,
            risk_score=0.1,
            confidence=0.9,
            rollback_plan="",  # Missing rollback
        )
        result = await gate.evaluate(decision, metrics)
        assert result == GateResult.DEFER

    @pytest.mark.asyncio
    async def test_evaluate_unhealthy_system_for_non_emergency_defers(self):
        """Unhealthy system should DEFER non-emergency decisions."""
        gate = DecisionGate()
        # System with very poor health
        bad_metrics = SystemMetrics(
            snr_score=0.0,
            ihsan_score=0.0,
            error_rate=0.5,
            latency_ms=4000,
            memory_usage=0.95,
        )
        decision = DecisionCandidate(
            decision_type=DecisionType.ROUTINE,
            risk_score=0.1,
            confidence=0.9,
            rollback_plan="rollback",
        )
        result = await gate.evaluate(decision, bad_metrics)
        assert result == GateResult.DEFER

    @pytest.mark.asyncio
    async def test_evaluate_pass_increments_passed_count(self):
        """Passing a decision should increment passed_count."""
        gate = DecisionGate()
        metrics = SystemMetrics(
            snr_score=0.95, ihsan_score=0.95, error_rate=0.0, memory_usage=0.1
        )
        decision = DecisionCandidate(
            decision_type=DecisionType.ROUTINE,
            risk_score=0.1,
            confidence=0.9,
        )
        result = await gate.evaluate(decision, metrics)
        assert result == GateResult.PASS
        assert gate.passed_count == 1

    @pytest.mark.asyncio
    async def test_evaluate_pass_adds_to_active_decisions(self):
        """Passing a decision should add it to active_decisions."""
        gate = DecisionGate()
        metrics = SystemMetrics(
            snr_score=0.95, ihsan_score=0.95, error_rate=0.0, memory_usage=0.1
        )
        decision = DecisionCandidate(
            decision_type=DecisionType.ROUTINE,
            risk_score=0.1,
            confidence=0.9,
        )
        await gate.evaluate(decision, metrics)
        assert decision.id in gate.active_decisions
        assert gate.active_decisions[decision.id] is decision

    @pytest.mark.asyncio
    async def test_evaluate_routine_does_not_need_rollback(self):
        """ROUTINE decisions should pass even without a rollback plan."""
        gate = DecisionGate(require_rollback=True)
        metrics = SystemMetrics(
            snr_score=0.95, ihsan_score=0.95, error_rate=0.0, memory_usage=0.1
        )
        decision = DecisionCandidate(
            decision_type=DecisionType.ROUTINE,
            risk_score=0.1,
            confidence=0.9,
            rollback_plan="",  # No rollback but ROUTINE
        )
        result = await gate.evaluate(decision, metrics)
        assert result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_evaluate_emergency_bypasses_health_check(self):
        """EMERGENCY decisions should pass even with unhealthy system."""
        gate = DecisionGate()
        bad_metrics = SystemMetrics(
            snr_score=0.0,
            ihsan_score=0.0,
            error_rate=0.5,
            latency_ms=4000,
            memory_usage=0.95,
        )
        decision = DecisionCandidate(
            decision_type=DecisionType.EMERGENCY,
            risk_score=0.5,  # Below EMERGENCY threshold of 0.7
            confidence=0.7,  # Above EMERGENCY requirement of 0.6
            rollback_plan="emergency_rollback",
        )
        result = await gate.evaluate(decision, bad_metrics)
        assert result == GateResult.PASS

    @pytest.mark.asyncio
    async def test_evaluate_emergency_lower_confidence_requirement(self):
        """EMERGENCY decisions should require only 0.6 confidence."""
        gate = DecisionGate()
        bad_metrics = SystemMetrics(
            snr_score=0.0, ihsan_score=0.0, error_rate=0.5, memory_usage=0.95
        )
        decision = DecisionCandidate(
            decision_type=DecisionType.EMERGENCY,
            risk_score=0.3,
            confidence=0.65,  # Above 0.6 EMERGENCY threshold
            rollback_plan="emergency_rollback",
        )
        result = await gate.evaluate(decision, bad_metrics)
        assert result == GateResult.PASS

    def test_complete_decision_removes_from_active(self):
        """complete_decision should remove the decision from active_decisions."""
        gate = DecisionGate()
        candidate = DecisionCandidate()
        gate.active_decisions[candidate.id] = candidate

        outcome = DecisionOutcome(decision_id=candidate.id)
        gate.complete_decision(candidate.id, outcome)
        assert candidate.id not in gate.active_decisions

    def test_complete_decision_appends_to_history(self):
        """complete_decision should append outcome to decision_history."""
        gate = DecisionGate()
        outcome = DecisionOutcome(decision_id="test123")
        gate.complete_decision("test123", outcome)
        assert len(gate.decision_history) == 1
        assert gate.decision_history[0] is outcome

    def test_complete_decision_nonexistent_id_no_error(self):
        """Completing a non-existent decision should not raise."""
        gate = DecisionGate()
        outcome = DecisionOutcome(decision_id="nonexistent")
        gate.complete_decision("nonexistent", outcome)
        assert len(gate.decision_history) == 1

    def test_approval_rate_no_decisions(self):
        """With no decisions, approval_rate should be 1.0."""
        gate = DecisionGate()
        assert gate.approval_rate == 1.0

    def test_approval_rate_calculation(self):
        """approval_rate should be passed / (passed + blocked)."""
        gate = DecisionGate()
        gate.passed_count = 3
        gate.blocked_count = 1
        assert gate.approval_rate == pytest.approx(0.75)

    def test_approval_rate_all_blocked(self):
        """All blocked should yield 0.0 approval rate."""
        gate = DecisionGate()
        gate.blocked_count = 5
        assert gate.approval_rate == 0.0

    def test_get_stats_returns_all_fields(self):
        """get_stats should return a dict with all expected keys."""
        gate = DecisionGate()
        gate.passed_count = 10
        gate.blocked_count = 2
        gate.active_decisions["x"] = DecisionCandidate()

        stats = gate.get_stats()
        assert stats["passed"] == 10
        assert stats["blocked"] == 2
        assert stats["approval_rate"] == pytest.approx(10 / 12)
        assert stats["active"] == 1
        assert stats["history_size"] == 0


# =============================================================================
# AUTONOMOUS LOOP TESTS
# =============================================================================


class TestAutonomousLoop:
    """Tests for the AutonomousLoop class."""

    def test_init_defaults(self):
        """Default AutonomousLoop should have standard initial state."""
        loop = AutonomousLoop()
        assert loop.snr_threshold == 0.95
        assert loop.ihsan_threshold == 0.95
        assert loop.cycle_interval == 5.0
        assert loop.max_decisions_per_cycle == 3
        assert loop.state == LoopState.IDLE
        assert loop.cycle_count == 0
        assert loop._running is False
        assert loop._paused is False

    def test_init_custom_params(self):
        """AutonomousLoop should accept custom initialization."""
        gate = DecisionGate(ihsan_threshold=0.90)
        loop = AutonomousLoop(
            decision_gate=gate,
            snr_threshold=0.85,
            ihsan_threshold=0.90,
            cycle_interval=10.0,
            max_decisions_per_cycle=5,
        )
        assert loop.gate is gate
        assert loop.snr_threshold == 0.85
        assert loop.ihsan_threshold == 0.90
        assert loop.cycle_interval == 10.0
        assert loop.max_decisions_per_cycle == 5

    def test_init_creates_default_gate_if_none(self):
        """AutonomousLoop should create a DecisionGate if none provided."""
        loop = AutonomousLoop()
        assert isinstance(loop.gate, DecisionGate)

    def test_init_history_deques_have_correct_maxlen(self):
        """History deques should have correct maxlen."""
        loop = AutonomousLoop()
        assert loop.observations.maxlen == 1000
        assert loop.decisions.maxlen == 500
        assert loop.outcomes.maxlen == 500

    # -------------------------------------------------------------------------
    # OBSERVE PHASE
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_observe_returns_system_metrics(self):
        """observe() should return SystemMetrics with default values."""
        loop = AutonomousLoop()
        metrics = await loop.observe()
        assert isinstance(metrics, SystemMetrics)
        assert metrics.snr_score == 0.92
        assert metrics.ihsan_score == 0.94
        assert metrics.latency_ms == 150
        assert metrics.error_rate == 0.02
        assert metrics.throughput == 100
        assert metrics.memory_usage == 0.45

    @pytest.mark.asyncio
    async def test_observe_sets_state_to_observing(self):
        """observe() should set state to OBSERVING."""
        loop = AutonomousLoop()
        await loop.observe()
        # State may change later, but it was OBSERVING during observe()
        # After observe returns, state is OBSERVING (last set)
        assert loop.state == LoopState.OBSERVING

    @pytest.mark.asyncio
    async def test_observe_appends_to_observations_deque(self):
        """observe() should append metrics to the observations deque."""
        loop = AutonomousLoop()
        assert len(loop.observations) == 0
        metrics = await loop.observe()
        assert len(loop.observations) == 1
        assert loop.observations[0] is metrics

    @pytest.mark.asyncio
    async def test_observe_calls_registered_observers(self):
        """observe() should call all registered observer callbacks."""
        loop = AutonomousLoop()
        called_with = []

        async def mock_observer(m):
            called_with.append(m)

        loop.register_observer(mock_observer)
        metrics = await loop.observe()
        assert len(called_with) == 1
        assert called_with[0] is metrics

    @pytest.mark.asyncio
    async def test_observe_handles_observer_errors_gracefully(self):
        """observe() should catch and log observer exceptions."""
        loop = AutonomousLoop()

        async def failing_observer(m):
            raise ValueError("Observer exploded")

        loop.register_observer(failing_observer)
        # Should not raise
        metrics = await loop.observe()
        assert isinstance(metrics, SystemMetrics)
        assert len(loop.observations) == 1

    @pytest.mark.asyncio
    async def test_observe_sets_active_tasks_from_gate(self):
        """observe() should report active_tasks from gate.active_decisions."""
        loop = AutonomousLoop()
        loop.gate.active_decisions["x"] = DecisionCandidate()
        loop.gate.active_decisions["y"] = DecisionCandidate()
        metrics = await loop.observe()
        assert metrics.active_tasks == 2

    # -------------------------------------------------------------------------
    # PREDICT PHASE
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_predict_with_few_observations_returns_empty_trends(self):
        """predict() should return empty trends with < 5 observations."""
        loop = AutonomousLoop()
        for _ in range(4):
            await loop.observe()
        metrics = SystemMetrics(snr_score=0.92, ihsan_score=0.94)
        predictions = await loop.predict(metrics)
        assert predictions["trends"] == {}

    @pytest.mark.asyncio
    async def test_predict_with_enough_observations_calculates_trends(self):
        """predict() should calculate trends with >= 5 observations."""
        loop = AutonomousLoop()
        for _ in range(6):
            await loop.observe()
        metrics = SystemMetrics(snr_score=0.92, ihsan_score=0.94)
        predictions = await loop.predict(metrics)
        assert "snr" in predictions["trends"]
        assert "ihsan" in predictions["trends"]

    @pytest.mark.asyncio
    async def test_predict_detects_rising_snr(self):
        """predict() should detect rising SNR as an opportunity."""
        loop = AutonomousLoop()
        # Manually inject observations with rising SNR
        for i in range(6):
            m = SystemMetrics(snr_score=0.80 + i * 0.03, ihsan_score=0.94)
            loop.observations.append(m)
        metrics = SystemMetrics(snr_score=0.95, ihsan_score=0.94)
        predictions = await loop.predict(metrics)
        assert predictions["trends"]["snr"]["direction"] == "rising"
        assert len(predictions["opportunities"]) > 0

    @pytest.mark.asyncio
    async def test_predict_detects_falling_ihsan_as_risk(self):
        """predict() should detect falling Ihsan as a risk."""
        loop = AutonomousLoop()
        for i in range(6):
            m = SystemMetrics(snr_score=0.92, ihsan_score=0.98 - i * 0.03)
            loop.observations.append(m)
        metrics = SystemMetrics(snr_score=0.92, ihsan_score=0.83)
        predictions = await loop.predict(metrics)
        assert predictions["trends"]["ihsan"]["direction"] == "falling"
        assert len(predictions["risks"]) > 0

    @pytest.mark.asyncio
    async def test_predict_stable_trends(self):
        """predict() should detect stable trends when values are consistent."""
        loop = AutonomousLoop()
        for _ in range(6):
            m = SystemMetrics(snr_score=0.92, ihsan_score=0.94)
            loop.observations.append(m)
        metrics = SystemMetrics(snr_score=0.92, ihsan_score=0.94)
        predictions = await loop.predict(metrics)
        assert predictions["trends"]["snr"]["direction"] == "stable"
        assert predictions["trends"]["ihsan"]["direction"] == "stable"

    @pytest.mark.asyncio
    async def test_predict_sets_state_to_predicting(self):
        """predict() should set state to PREDICTING."""
        loop = AutonomousLoop()
        metrics = SystemMetrics()
        await loop.predict(metrics)
        assert loop.state == LoopState.PREDICTING

    @pytest.mark.asyncio
    async def test_predict_calls_registered_predictors(self):
        """predict() should call registered predictor callbacks."""
        loop = AutonomousLoop()
        called = []

        async def mock_predictor(metrics, predictions):
            called.append(True)
            return {"custom_forecast": "good"}

        loop.register_predictor(mock_predictor)
        metrics = SystemMetrics()
        result = await loop.predict(metrics)
        assert len(called) == 1
        assert result.get("custom_forecast") == "good"

    # -------------------------------------------------------------------------
    # COORDINATE PHASE
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_coordinate_low_health_returns_defensive(self):
        """coordinate() should return defensive mode when health < 0.7."""
        loop = AutonomousLoop()
        bad_metrics = SystemMetrics(
            snr_score=0.0, ihsan_score=0.0, error_rate=0.5, memory_usage=0.95
        )
        assert bad_metrics.health_score() < 0.7
        predictions: Dict[str, Any] = {"opportunities": [], "risks": []}
        result = await loop.coordinate(bad_metrics, predictions)
        assert result["resource_allocation"]["mode"] == "defensive"

    @pytest.mark.asyncio
    async def test_coordinate_with_opportunities_returns_proactive(self):
        """coordinate() should return proactive mode when opportunities exist."""
        loop = AutonomousLoop()
        good_metrics = SystemMetrics(
            snr_score=0.95, ihsan_score=0.95, error_rate=0.0, memory_usage=0.1
        )
        predictions: Dict[str, Any] = {
            "opportunities": [{"type": "optimization"}],
            "risks": [],
        }
        result = await loop.coordinate(good_metrics, predictions)
        assert result["resource_allocation"]["mode"] == "proactive"

    @pytest.mark.asyncio
    async def test_coordinate_default_returns_balanced(self):
        """coordinate() should return balanced mode by default."""
        loop = AutonomousLoop()
        good_metrics = SystemMetrics(
            snr_score=0.95, ihsan_score=0.95, error_rate=0.0, memory_usage=0.1
        )
        predictions: Dict[str, Any] = {"opportunities": [], "risks": []}
        result = await loop.coordinate(good_metrics, predictions)
        assert result["resource_allocation"]["mode"] == "balanced"

    @pytest.mark.asyncio
    async def test_coordinate_sets_state_to_coordinating(self):
        """coordinate() should set state to COORDINATING."""
        loop = AutonomousLoop()
        metrics = SystemMetrics(snr_score=0.95, ihsan_score=0.95)
        await loop.coordinate(metrics, {})
        assert loop.state == LoopState.COORDINATING

    @pytest.mark.asyncio
    async def test_coordinate_calls_registered_coordinators(self):
        """coordinate() should call registered coordinator callbacks."""
        loop = AutonomousLoop()
        called = []

        async def mock_coordinator(metrics, predictions, coordination):
            called.append(True)
            return {"custom_allocation": "special"}

        loop.register_coordinator(mock_coordinator)
        metrics = SystemMetrics(snr_score=0.95, ihsan_score=0.95)
        result = await loop.coordinate(metrics, {})
        assert len(called) == 1
        assert result.get("custom_allocation") == "special"

    # -------------------------------------------------------------------------
    # ANALYZE PHASE
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_analyze_generates_snr_correction_candidate(self):
        """analyze() should generate SNR correction when below threshold-0.1."""
        loop = AutonomousLoop(snr_threshold=0.95)
        # snr_score = 0.80 < 0.95 - 0.1 = 0.85
        metrics = SystemMetrics(snr_score=0.80, ihsan_score=0.95, error_rate=0.01)
        candidates = await loop.analyze(metrics)
        snr_candidates = [c for c in candidates if c.action == "boost_snr"]
        assert len(snr_candidates) == 1
        assert snr_candidates[0].decision_type == DecisionType.CORRECTIVE

    @pytest.mark.asyncio
    async def test_analyze_no_snr_correction_when_above_threshold(self):
        """analyze() should not generate SNR correction when above threshold-0.1."""
        loop = AutonomousLoop(snr_threshold=0.95)
        # snr_score = 0.90 >= 0.95 - 0.1 = 0.85
        metrics = SystemMetrics(snr_score=0.90, ihsan_score=0.95, error_rate=0.01)
        candidates = await loop.analyze(metrics)
        snr_candidates = [c for c in candidates if c.action == "boost_snr"]
        assert len(snr_candidates) == 0

    @pytest.mark.asyncio
    async def test_analyze_generates_error_correction_candidate(self):
        """analyze() should generate error correction when error_rate > 0.1."""
        loop = AutonomousLoop()
        metrics = SystemMetrics(snr_score=0.95, ihsan_score=0.95, error_rate=0.15)
        candidates = await loop.analyze(metrics)
        error_candidates = [c for c in candidates if c.action == "reduce_errors"]
        assert len(error_candidates) == 1
        assert error_candidates[0].decision_type == DecisionType.CORRECTIVE

    @pytest.mark.asyncio
    async def test_analyze_no_error_correction_when_below_threshold(self):
        """analyze() should not generate error correction when error_rate <= 0.1."""
        loop = AutonomousLoop()
        metrics = SystemMetrics(snr_score=0.95, ihsan_score=0.95, error_rate=0.05)
        candidates = await loop.analyze(metrics)
        error_candidates = [c for c in candidates if c.action == "reduce_errors"]
        assert len(error_candidates) == 0

    @pytest.mark.asyncio
    async def test_analyze_respects_max_decisions_per_cycle(self):
        """analyze() should limit candidates to max_decisions_per_cycle."""
        loop = AutonomousLoop(max_decisions_per_cycle=1)

        async def many_candidates(m):
            return [
                DecisionCandidate(action=f"action_{i}") for i in range(5)
            ]

        loop.register_analyzer(many_candidates)
        metrics = SystemMetrics(snr_score=0.95, ihsan_score=0.95, error_rate=0.01)
        candidates = await loop.analyze(metrics)
        assert len(candidates) <= 1

    @pytest.mark.asyncio
    async def test_analyze_calls_registered_analyzers(self):
        """analyze() should call all registered analyzer callbacks."""
        loop = AutonomousLoop()
        called = []

        async def mock_analyzer(m):
            called.append(True)
            return [
                DecisionCandidate(
                    decision_type=DecisionType.INNOVATIVE,
                    action="custom_action",
                    confidence=0.9,
                )
            ]

        loop.register_analyzer(mock_analyzer)
        metrics = SystemMetrics(snr_score=0.95, ihsan_score=0.95, error_rate=0.01)
        candidates = await loop.analyze(metrics)
        assert len(called) == 1
        custom = [c for c in candidates if c.action == "custom_action"]
        assert len(custom) == 1

    @pytest.mark.asyncio
    async def test_analyze_handles_analyzer_error(self):
        """analyze() should catch and log analyzer exceptions."""
        loop = AutonomousLoop()

        async def failing_analyzer(m):
            raise RuntimeError("Analyzer crashed")

        loop.register_analyzer(failing_analyzer)
        metrics = SystemMetrics(snr_score=0.95, ihsan_score=0.95)
        # Should not raise
        candidates = await loop.analyze(metrics)
        assert isinstance(candidates, list)

    @pytest.mark.asyncio
    async def test_analyze_sets_state_to_analyzing(self):
        """analyze() should set state to ANALYZING."""
        loop = AutonomousLoop()
        metrics = SystemMetrics(snr_score=0.95, ihsan_score=0.95)
        await loop.analyze(metrics)
        assert loop.state == LoopState.ANALYZING

    # -------------------------------------------------------------------------
    # DECIDE PHASE
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_decide_filters_through_gate(self):
        """decide() should only return candidates that pass the gate."""
        loop = AutonomousLoop()
        metrics = SystemMetrics(
            snr_score=0.95, ihsan_score=0.95, error_rate=0.0, memory_usage=0.1
        )
        good = DecisionCandidate(
            decision_type=DecisionType.ROUTINE,
            risk_score=0.1,
            confidence=0.9,
        )
        bad = DecisionCandidate(
            decision_type=DecisionType.ROUTINE,
            risk_score=0.5,  # Above ROUTINE risk threshold of 0.3
            confidence=0.9,
        )
        approved = await loop.decide([good, bad], metrics)
        assert len(approved) == 1
        assert approved[0] is good

    @pytest.mark.asyncio
    async def test_decide_appends_approved_to_decisions_deque(self):
        """decide() should append approved decisions to the decisions deque."""
        loop = AutonomousLoop()
        metrics = SystemMetrics(
            snr_score=0.95, ihsan_score=0.95, error_rate=0.0, memory_usage=0.1
        )
        decision = DecisionCandidate(
            decision_type=DecisionType.ROUTINE,
            risk_score=0.1,
            confidence=0.9,
        )
        await loop.decide([decision], metrics)
        assert len(loop.decisions) == 1
        assert loop.decisions[0] is decision

    @pytest.mark.asyncio
    async def test_decide_sets_state_to_planning(self):
        """decide() should set state to PLANNING."""
        loop = AutonomousLoop()
        metrics = SystemMetrics(snr_score=0.95, ihsan_score=0.95)
        await loop.decide([], metrics)
        assert loop.state == LoopState.PLANNING

    @pytest.mark.asyncio
    async def test_decide_empty_candidates_returns_empty(self):
        """decide() with no candidates should return empty list."""
        loop = AutonomousLoop()
        metrics = SystemMetrics(snr_score=0.95, ihsan_score=0.95)
        approved = await loop.decide([], metrics)
        assert approved == []

    # -------------------------------------------------------------------------
    # ACT PHASE
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_act_with_registered_executor(self):
        """act() should use registered executor for matching action."""
        loop = AutonomousLoop()

        async def mock_executor(decision):
            return True

        loop.register_executor("test_action", mock_executor)
        decision = DecisionCandidate(action="test_action")
        # Need to add to gate active decisions for complete_decision to work
        loop.gate.active_decisions[decision.id] = decision

        metrics = SystemMetrics(snr_score=0.95, ihsan_score=0.95)
        outcomes = await loop.act([decision], metrics)
        assert len(outcomes) == 1
        assert outcomes[0].success is True
        assert outcomes[0].executed is True

    @pytest.mark.asyncio
    async def test_act_no_executor_default_log_only(self):
        """act() without executor should default to log-only success."""
        loop = AutonomousLoop()
        decision = DecisionCandidate(action="unknown_action")
        loop.gate.active_decisions[decision.id] = decision

        metrics = SystemMetrics(snr_score=0.95, ihsan_score=0.95)
        outcomes = await loop.act([decision], metrics)
        assert len(outcomes) == 1
        assert outcomes[0].success is True
        assert outcomes[0].executed is True

    @pytest.mark.asyncio
    async def test_act_handles_timeout(self):
        """act() should handle executor timeout."""
        loop = AutonomousLoop()

        async def slow_executor(decision):
            await asyncio.sleep(10)
            return True

        loop.register_executor("slow_action", slow_executor)
        decision = DecisionCandidate(action="slow_action", timeout_seconds=0.01)
        loop.gate.active_decisions[decision.id] = decision

        metrics = SystemMetrics(snr_score=0.95, ihsan_score=0.95)
        outcomes = await loop.act([decision], metrics)
        assert len(outcomes) == 1
        assert outcomes[0].success is False
        assert outcomes[0].error_message == "Execution timeout"

    @pytest.mark.asyncio
    async def test_act_handles_executor_exception(self):
        """act() should handle executor exceptions gracefully."""
        loop = AutonomousLoop()

        async def broken_executor(decision):
            raise RuntimeError("Executor exploded")

        loop.register_executor("broken_action", broken_executor)
        decision = DecisionCandidate(action="broken_action")
        loop.gate.active_decisions[decision.id] = decision

        metrics = SystemMetrics(snr_score=0.95, ihsan_score=0.95)
        outcomes = await loop.act([decision], metrics)
        assert len(outcomes) == 1
        assert outcomes[0].success is False
        assert "Executor exploded" in outcomes[0].error_message

    @pytest.mark.asyncio
    async def test_act_completes_decision_in_gate(self):
        """act() should call gate.complete_decision for each outcome."""
        loop = AutonomousLoop()
        decision = DecisionCandidate(action="test_action")
        loop.gate.active_decisions[decision.id] = decision

        metrics = SystemMetrics(snr_score=0.95, ihsan_score=0.95)
        await loop.act([decision], metrics)
        assert decision.id not in loop.gate.active_decisions
        assert len(loop.gate.decision_history) == 1

    @pytest.mark.asyncio
    async def test_act_appends_to_outcomes_deque(self):
        """act() should append outcomes to the outcomes deque."""
        loop = AutonomousLoop()
        decision = DecisionCandidate(action="test")
        loop.gate.active_decisions[decision.id] = decision

        metrics = SystemMetrics(snr_score=0.95, ihsan_score=0.95)
        await loop.act([decision], metrics)
        assert len(loop.outcomes) == 1

    @pytest.mark.asyncio
    async def test_act_sets_state_to_acting(self):
        """act() should set state to ACTING."""
        loop = AutonomousLoop()
        metrics = SystemMetrics(snr_score=0.95, ihsan_score=0.95)
        await loop.act([], metrics)
        assert loop.state == LoopState.ACTING

    @pytest.mark.asyncio
    async def test_act_records_execution_time(self):
        """act() should record execution_time_ms in outcome."""
        loop = AutonomousLoop()

        async def slow_ish_executor(decision):
            await asyncio.sleep(0.01)
            return True

        loop.register_executor("timed_action", slow_ish_executor)
        decision = DecisionCandidate(action="timed_action")
        loop.gate.active_decisions[decision.id] = decision

        metrics = SystemMetrics(snr_score=0.95, ihsan_score=0.95)
        outcomes = await loop.act([decision], metrics)
        assert outcomes[0].execution_time_ms > 0

    @pytest.mark.asyncio
    async def test_act_sets_metrics_before(self):
        """act() should set metrics_before on each outcome."""
        loop = AutonomousLoop()
        decision = DecisionCandidate(action="test")
        loop.gate.active_decisions[decision.id] = decision

        metrics = SystemMetrics(snr_score=0.95, ihsan_score=0.95)
        outcomes = await loop.act([decision], metrics)
        assert outcomes[0].metrics_before is metrics

    # -------------------------------------------------------------------------
    # REFLECT PHASE
    # -------------------------------------------------------------------------

    @pytest.mark.asyncio
    async def test_reflect_calculates_stats(self):
        """reflect() should calculate success rate and other stats."""
        loop = AutonomousLoop()
        outcomes = [
            DecisionOutcome(decision_id="a", success=True, execution_time_ms=10),
            DecisionOutcome(decision_id="b", success=True, execution_time_ms=20),
            DecisionOutcome(decision_id="c", success=False, execution_time_ms=30),
        ]
        reflection = await loop.reflect(outcomes)
        assert reflection["decisions_made"] == 3
        assert reflection["successful"] == 2
        assert reflection["success_rate"] == pytest.approx(2 / 3)
        assert reflection["avg_execution_time"] == pytest.approx(20.0)

    @pytest.mark.asyncio
    async def test_reflect_empty_outcomes(self):
        """reflect() with no outcomes should return safe defaults."""
        loop = AutonomousLoop()
        reflection = await loop.reflect([])
        assert reflection["decisions_made"] == 0
        assert reflection["success_rate"] == 1.0
        assert reflection["avg_execution_time"] == 0

    @pytest.mark.asyncio
    async def test_reflect_enters_adapting_on_low_success_rate(self):
        """reflect() should enter ADAPTING if success_rate < 0.5 with >= 3 outcomes."""
        loop = AutonomousLoop()
        outcomes = [
            DecisionOutcome(decision_id="a", success=False),
            DecisionOutcome(decision_id="b", success=False),
            DecisionOutcome(decision_id="c", success=False),
        ]
        await loop.reflect(outcomes)
        assert loop.state == LoopState.ADAPTING

    @pytest.mark.asyncio
    async def test_reflect_does_not_adapt_with_few_outcomes(self):
        """reflect() should not enter ADAPTING with < 3 outcomes even if all fail."""
        loop = AutonomousLoop()
        outcomes = [
            DecisionOutcome(decision_id="a", success=False),
            DecisionOutcome(decision_id="b", success=False),
        ]
        await loop.reflect(outcomes)
        assert loop.state == LoopState.REFLECTING

    @pytest.mark.asyncio
    async def test_reflect_does_not_adapt_on_good_success_rate(self):
        """reflect() should stay REFLECTING when success_rate >= 0.5."""
        loop = AutonomousLoop()
        outcomes = [
            DecisionOutcome(decision_id="a", success=True),
            DecisionOutcome(decision_id="b", success=True),
            DecisionOutcome(decision_id="c", success=False),
        ]
        await loop.reflect(outcomes)
        assert loop.state == LoopState.REFLECTING

    @pytest.mark.asyncio
    async def test_reflect_sets_state_to_reflecting(self):
        """reflect() should set state to REFLECTING."""
        loop = AutonomousLoop()
        await loop.reflect([])
        assert loop.state == LoopState.REFLECTING

    @pytest.mark.asyncio
    async def test_reflect_includes_gate_stats(self):
        """reflect() should include gate statistics."""
        loop = AutonomousLoop()
        reflection = await loop.reflect([])
        assert "gate_stats" in reflection
        assert "passed" in reflection["gate_stats"]

    @pytest.mark.asyncio
    async def test_reflect_includes_cycle_count(self):
        """reflect() should include the current cycle count."""
        loop = AutonomousLoop()
        loop.cycle_count = 42
        reflection = await loop.reflect([])
        assert reflection["cycle"] == 42


# =============================================================================
# LEARN PHASE TESTS
# =============================================================================


class TestLearnPhase:
    """Tests for the learn() method of AutonomousLoop."""

    @pytest.mark.asyncio
    async def test_learn_with_few_outcomes_no_patterns(self):
        """learn() should detect no patterns with < 10 outcomes."""
        loop = AutonomousLoop()
        for i in range(5):
            loop.outcomes.append(
                DecisionOutcome(decision_id=f"d{i}", success=False)
            )
        outcomes = list(loop.outcomes)
        reflection = {"success_rate": 0.0}
        learning = await loop.learn(outcomes, reflection)
        assert learning["patterns_detected"] == []
        assert learning["threshold_adjustments"] == []

    @pytest.mark.asyncio
    async def test_learn_detects_low_success_rate_pattern(self):
        """learn() should detect low_success_rate when rate < 0.7."""
        loop = AutonomousLoop()
        # Add 15 outcomes, mostly failures
        for i in range(15):
            loop.outcomes.append(
                DecisionOutcome(
                    decision_id=f"d{i}",
                    success=(i < 3),  # Only 3 successes out of 15
                    execution_time_ms=50,
                )
            )
        outcomes = list(loop.outcomes)
        reflection = {"success_rate": 0.2}
        learning = await loop.learn(outcomes, reflection)
        patterns = [p["pattern"] for p in learning["patterns_detected"]]
        assert "low_success_rate" in patterns
        # Should also have threshold adjustment
        assert len(learning["threshold_adjustments"]) > 0

    @pytest.mark.asyncio
    async def test_learn_detects_slow_execution_pattern(self):
        """learn() should detect slow_execution when avg_time > 1000ms."""
        loop = AutonomousLoop()
        for i in range(15):
            loop.outcomes.append(
                DecisionOutcome(
                    decision_id=f"d{i}",
                    success=True,
                    execution_time_ms=2000,  # Very slow
                )
            )
        outcomes = list(loop.outcomes)
        reflection = {"success_rate": 1.0}
        learning = await loop.learn(outcomes, reflection)
        patterns = [p["pattern"] for p in learning["patterns_detected"]]
        assert "slow_execution" in patterns

    @pytest.mark.asyncio
    async def test_learn_no_slow_execution_when_fast(self):
        """learn() should not detect slow_execution when avg_time <= 1000ms."""
        loop = AutonomousLoop()
        for i in range(15):
            loop.outcomes.append(
                DecisionOutcome(
                    decision_id=f"d{i}",
                    success=True,
                    execution_time_ms=50,
                )
            )
        outcomes = list(loop.outcomes)
        reflection = {"success_rate": 1.0}
        learning = await loop.learn(outcomes, reflection)
        patterns = [p["pattern"] for p in learning["patterns_detected"]]
        assert "slow_execution" not in patterns

    @pytest.mark.asyncio
    async def test_learn_calls_registered_learners(self):
        """learn() should call registered learner callbacks."""
        loop = AutonomousLoop()
        called = []

        async def mock_learner(outcomes, reflection, learning):
            called.append(True)
            return {"custom_learning": "applied"}

        loop.register_learner(mock_learner)
        learning = await loop.learn([], {})
        assert len(called) == 1
        assert learning.get("custom_learning") == "applied"

    @pytest.mark.asyncio
    async def test_learn_handles_learner_error(self):
        """learn() should catch and log learner exceptions."""
        loop = AutonomousLoop()

        async def broken_learner(outcomes, reflection, learning):
            raise ValueError("Learner crashed")

        loop.register_learner(broken_learner)
        # Should not raise
        learning = await loop.learn([], {})
        assert isinstance(learning, dict)

    @pytest.mark.asyncio
    async def test_learn_sets_state_to_learning(self):
        """learn() should set state to LEARNING."""
        loop = AutonomousLoop()
        await loop.learn([], {})
        assert loop.state == LoopState.LEARNING

    @pytest.mark.asyncio
    async def test_learn_both_patterns_detected_simultaneously(self):
        """learn() should detect both low_success_rate and slow_execution."""
        loop = AutonomousLoop()
        for i in range(15):
            loop.outcomes.append(
                DecisionOutcome(
                    decision_id=f"d{i}",
                    success=(i < 2),  # Only 2 successes out of 15
                    execution_time_ms=2000,
                )
            )
        outcomes = list(loop.outcomes)
        reflection = {"success_rate": 2 / 15}
        learning = await loop.learn(outcomes, reflection)
        patterns = [p["pattern"] for p in learning["patterns_detected"]]
        assert "low_success_rate" in patterns
        assert "slow_execution" in patterns

    @pytest.mark.asyncio
    async def test_learn_uses_last_twenty_outcomes(self):
        """learn() should analyze the last 20 outcomes from the deque."""
        loop = AutonomousLoop()
        # Add 25 successful outcomes
        for i in range(25):
            loop.outcomes.append(
                DecisionOutcome(
                    decision_id=f"d{i}",
                    success=True,
                    execution_time_ms=50,
                )
            )
        # Regardless of total, should look at last 20
        learning = await loop.learn([], {})
        # All successes, no patterns expected
        patterns = [p["pattern"] for p in learning["patterns_detected"]]
        assert "low_success_rate" not in patterns

    @pytest.mark.asyncio
    async def test_learn_returns_structure(self):
        """learn() should return dict with expected keys."""
        loop = AutonomousLoop()
        learning = await loop.learn([], {})
        assert "patterns_detected" in learning
        assert "threshold_adjustments" in learning
        assert "strategy_updates" in learning


# =============================================================================
# RUN CYCLE TESTS
# =============================================================================


class TestRunCycle:
    """Tests for the run_cycle() method."""

    @pytest.mark.asyncio
    async def test_run_cycle_increments_cycle_count(self):
        """run_cycle() should increment cycle_count."""
        loop = AutonomousLoop()
        assert loop.cycle_count == 0
        await loop.run_cycle()
        assert loop.cycle_count == 1
        await loop.run_cycle()
        assert loop.cycle_count == 2

    @pytest.mark.asyncio
    async def test_run_cycle_extended_includes_predict_coordinate_learn(self):
        """run_cycle(extended=True) should include predict, coordinate, learn results."""
        loop = AutonomousLoop()
        result = await loop.run_cycle(extended=True)
        assert "predictions" in result
        assert "coordination" in result
        assert "learning" in result
        # Extended predictions should have forecast_horizon
        assert result["predictions"].get("forecast_horizon") == "1h"

    @pytest.mark.asyncio
    async def test_run_cycle_basic_skips_predict_coordinate_learn(self):
        """run_cycle(extended=False) should skip predict, coordinate, learn."""
        loop = AutonomousLoop()
        result = await loop.run_cycle(extended=False)
        assert result["predictions"] == {}
        assert result["coordination"] == {}
        assert result["learning"] == {}

    @pytest.mark.asyncio
    async def test_run_cycle_returns_full_result_dict(self):
        """run_cycle() should return a dict with all expected keys."""
        loop = AutonomousLoop()
        result = await loop.run_cycle()
        expected_keys = {
            "cycle",
            "state",
            "health",
            "candidates",
            "approved",
            "executed",
            "reflection",
            "predictions",
            "coordination",
            "learning",
        }
        assert set(result.keys()) == expected_keys

    @pytest.mark.asyncio
    async def test_run_cycle_health_is_float(self):
        """run_cycle() result health should be a float."""
        loop = AutonomousLoop()
        result = await loop.run_cycle()
        assert isinstance(result["health"], float)
        assert 0.0 <= result["health"] <= 1.0

    @pytest.mark.asyncio
    async def test_run_cycle_reflection_included(self):
        """run_cycle() should include reflection results."""
        loop = AutonomousLoop()
        result = await loop.run_cycle()
        assert "reflection" in result
        assert "success_rate" in result["reflection"]

    @pytest.mark.asyncio
    async def test_run_cycle_state_is_string(self):
        """run_cycle() result state should be a string (enum name)."""
        loop = AutonomousLoop()
        result = await loop.run_cycle()
        assert isinstance(result["state"], str)

    @pytest.mark.asyncio
    async def test_run_cycle_populates_observations(self):
        """run_cycle() should add at least one observation."""
        loop = AutonomousLoop()
        assert len(loop.observations) == 0
        await loop.run_cycle()
        assert len(loop.observations) >= 1


# =============================================================================
# LOOP CONTROL TESTS
# =============================================================================


class TestLoopControl:
    """Tests for loop control methods (stop, pause, resume, status)."""

    def test_stop_sets_running_false(self):
        """stop() should set _running to False."""
        loop = AutonomousLoop()
        loop._running = True
        loop.stop()
        assert loop._running is False

    def test_pause_sets_paused_true_and_state_paused(self):
        """pause() should set _paused True and state to PAUSED."""
        loop = AutonomousLoop()
        loop.pause()
        assert loop._paused is True
        assert loop.state == LoopState.PAUSED

    def test_resume_sets_paused_false(self):
        """resume() should set _paused to False."""
        loop = AutonomousLoop()
        loop.pause()
        loop.resume()
        assert loop._paused is False

    def test_status_returns_all_fields(self):
        """status() should return dict with all expected keys."""
        loop = AutonomousLoop()
        s = loop.status()
        expected_keys = {
            "state",
            "cycle",
            "running",
            "paused",
            "observations",
            "decisions",
            "outcomes",
            "gate",
        }
        assert set(s.keys()) == expected_keys

    def test_status_reflects_current_state(self):
        """status() should reflect the current state accurately."""
        loop = AutonomousLoop()
        loop.cycle_count = 7
        loop._running = True
        loop._paused = False
        s = loop.status()
        assert s["state"] == "IDLE"
        assert s["cycle"] == 7
        assert s["running"] is True
        assert s["paused"] is False

    def test_status_reflects_paused_state(self):
        """status() should reflect paused state."""
        loop = AutonomousLoop()
        loop.pause()
        s = loop.status()
        assert s["state"] == "PAUSED"
        assert s["paused"] is True

    def test_status_includes_gate_stats(self):
        """status() should include gate statistics."""
        loop = AutonomousLoop()
        s = loop.status()
        assert "gate" in s
        assert "passed" in s["gate"]
        assert "blocked" in s["gate"]

    def test_status_counts_deque_lengths(self):
        """status() should report correct lengths for history deques."""
        loop = AutonomousLoop()
        loop.observations.append(SystemMetrics())
        loop.observations.append(SystemMetrics())
        loop.decisions.append(DecisionCandidate())
        s = loop.status()
        assert s["observations"] == 2
        assert s["decisions"] == 1
        assert s["outcomes"] == 0


# =============================================================================
# REGISTRATION TESTS
# =============================================================================


class TestRegistration:
    """Tests for callback registration methods."""

    def test_register_observer_adds_to_list(self):
        """register_observer should add callback to _observers."""
        loop = AutonomousLoop()
        assert len(loop._observers) == 0

        async def observer(m):
            pass

        loop.register_observer(observer)
        assert len(loop._observers) == 1
        assert loop._observers[0] is observer

    def test_register_analyzer_adds_to_list(self):
        """register_analyzer should add callback to _analyzers."""
        loop = AutonomousLoop()
        assert len(loop._analyzers) == 0

        async def analyzer(m):
            return []

        loop.register_analyzer(analyzer)
        assert len(loop._analyzers) == 1
        assert loop._analyzers[0] is analyzer

    def test_register_executor_adds_to_dict(self):
        """register_executor should add callback to _executors dict."""
        loop = AutonomousLoop()
        assert len(loop._executors) == 0

        async def executor(d):
            return True

        loop.register_executor("my_action", executor)
        assert "my_action" in loop._executors
        assert loop._executors["my_action"] is executor

    def test_register_predictor_creates_list_if_needed(self):
        """register_predictor should create _predictors list if absent."""
        loop = AutonomousLoop()
        assert not hasattr(loop, "_predictors")

        async def predictor(m, p):
            return {}

        loop.register_predictor(predictor)
        assert hasattr(loop, "_predictors")
        assert len(loop._predictors) == 1

    def test_register_coordinator_creates_list_if_needed(self):
        """register_coordinator should create _coordinators list if absent."""
        loop = AutonomousLoop()
        assert not hasattr(loop, "_coordinators")

        async def coordinator(m, p, c):
            return {}

        loop.register_coordinator(coordinator)
        assert hasattr(loop, "_coordinators")
        assert len(loop._coordinators) == 1

    def test_register_learner_creates_list_if_needed(self):
        """register_learner should create _learners list if absent."""
        loop = AutonomousLoop()
        assert not hasattr(loop, "_learners")

        async def learner(o, r, l):
            return {}

        loop.register_learner(learner)
        assert hasattr(loop, "_learners")
        assert len(loop._learners) == 1

    def test_register_multiple_observers(self):
        """Multiple observers should all be registered."""
        loop = AutonomousLoop()

        async def obs1(m):
            pass

        async def obs2(m):
            pass

        loop.register_observer(obs1)
        loop.register_observer(obs2)
        assert len(loop._observers) == 2

    def test_register_multiple_executors(self):
        """Multiple executors for different actions should all be stored."""
        loop = AutonomousLoop()

        async def exec1(d):
            return True

        async def exec2(d):
            return True

        loop.register_executor("action_a", exec1)
        loop.register_executor("action_b", exec2)
        assert len(loop._executors) == 2
        assert "action_a" in loop._executors
        assert "action_b" in loop._executors


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestFactory:
    """Tests for the create_autonomous_loop factory function."""

    def test_create_returns_autonomous_loop(self):
        """create_autonomous_loop should return an AutonomousLoop instance."""
        loop = create_autonomous_loop()
        assert isinstance(loop, AutonomousLoop)

    def test_create_passes_thresholds(self):
        """create_autonomous_loop should pass thresholds to the loop."""
        loop = create_autonomous_loop(
            snr_threshold=0.80,
            ihsan_threshold=0.85,
            cycle_interval=10.0,
        )
        assert loop.snr_threshold == 0.80
        assert loop.ihsan_threshold == 0.85
        assert loop.cycle_interval == 10.0

    def test_create_configures_gate_with_ihsan_threshold(self):
        """create_autonomous_loop should configure gate with ihsan_threshold."""
        loop = create_autonomous_loop(ihsan_threshold=0.90)
        assert loop.gate.ihsan_threshold == 0.90

    def test_create_default_thresholds(self):
        """create_autonomous_loop defaults should match documented values."""
        loop = create_autonomous_loop()
        assert loop.snr_threshold == 0.95
        assert loop.ihsan_threshold == 0.95
        assert loop.cycle_interval == 5.0


# =============================================================================
# INTEGRATION-STYLE TESTS
# =============================================================================


class TestFullCycleIntegration:
    """Integration tests that exercise the full OODA cycle end-to-end."""

    @pytest.mark.asyncio
    async def test_full_cycle_with_snr_correction(self):
        """Full cycle should detect low SNR, generate candidate, and execute."""
        loop = AutonomousLoop(snr_threshold=0.95)

        executed_actions = []

        async def snr_executor(decision):
            executed_actions.append(decision.action)
            return True

        loop.register_executor("boost_snr", snr_executor)

        # Override observe to provide low SNR
        original_observe = loop.observe

        async def low_snr_observe():
            m = await original_observe()
            # Override the default metrics with low SNR
            low_snr_metrics = SystemMetrics(
                snr_score=0.80,  # Below threshold - 0.1 = 0.85
                ihsan_score=0.95,
                latency_ms=100,
                error_rate=0.01,
                throughput=100,
                memory_usage=0.3,
                active_tasks=len(loop.gate.active_decisions),
            )
            # Replace the last observation
            loop.observations[-1] = low_snr_metrics
            return low_snr_metrics

        loop.observe = low_snr_observe

        result = await loop.run_cycle(extended=False)
        assert result["candidates"] >= 1
        assert "boost_snr" in executed_actions

    @pytest.mark.asyncio
    async def test_multiple_cycles_accumulate_history(self):
        """Multiple cycles should accumulate observations."""
        loop = AutonomousLoop()
        for _ in range(5):
            await loop.run_cycle(extended=False)
        assert len(loop.observations) == 5
        assert loop.cycle_count == 5

    @pytest.mark.asyncio
    async def test_full_extended_cycle_runs_all_phases(self):
        """Extended cycle should run all 8 phases without error."""
        loop = AutonomousLoop()
        result = await loop.run_cycle(extended=True)
        assert result["cycle"] == 1
        assert isinstance(result["health"], float)
        assert isinstance(result["predictions"], dict)
        assert isinstance(result["coordination"], dict)
        assert isinstance(result["learning"], dict)
        assert isinstance(result["reflection"], dict)
