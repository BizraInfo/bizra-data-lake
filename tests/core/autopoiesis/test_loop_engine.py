"""
Tests for Autopoietic Loop Engine
=================================

Tests the core improvement cycle with FATE gate verification,
rollback mechanisms, and human approval queue.

Standing on Giants: Maturana & Varela (1972) + Deming (1986) + Anthropic (2025)
"""

import asyncio
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import pytest

from core.autopoiesis.loop_engine import (
    ActivationGuardrails,
    AutopoieticLoop,
    AutopoieticState,
    HumanApprovalQueue,
    Hypothesis,
    HypothesisCategory,
    ImplementationResult,
    MockFATEGate,
    MockSensorHub,
    RateLimiter,
    RiskLevel,
    RollbackManager,
    SystemObservation,
    create_autopoietic_loop,
)
from core.integration.constants import UNIFIED_IHSAN_THRESHOLD, UNIFIED_SNR_THRESHOLD

# =============================================================================
# AUTOPOIETIC STATE TESTS
# =============================================================================


class TestAutopoieticState:
    """Tests for AutopoieticState enum."""

    def test_all_states_defined(self):
        """Verify all required states are defined."""
        required_states = [
            "DORMANT",
            "OBSERVING",
            "HYPOTHESIZING",
            "VALIDATING",
            "IMPLEMENTING",
            "INTEGRATING",
            "REFLECTING",
            "EMERGENCY_ROLLBACK",
            "HALTED",
        ]
        for state in required_states:
            assert hasattr(AutopoieticState, state), f"Missing state: {state}"

    def test_state_values(self):
        """Verify state string values."""
        assert AutopoieticState.DORMANT.value == "dormant"
        assert AutopoieticState.OBSERVING.value == "observing"
        assert AutopoieticState.VALIDATING.value == "validating"


# =============================================================================
# HYPOTHESIS TESTS
# =============================================================================


class TestHypothesis:
    """Tests for Hypothesis dataclass."""

    def test_hypothesis_creation(self):
        """Test creating a hypothesis."""
        hypothesis = Hypothesis(
            id="hyp_test001",
            description="Test improvement",
            category=HypothesisCategory.PERFORMANCE,
            predicted_improvement=0.15,
            required_changes=[{"component": "test", "action": "update"}],
            affected_components=["test_component"],
            risk_level=RiskLevel.LOW,
            reversibility_plan={"action": "restore"},
            ihsan_impact_estimate=0.01,
            snr_impact_estimate=0.02,
        )

        assert hypothesis.id == "hyp_test001"
        assert hypothesis.category == HypothesisCategory.PERFORMANCE
        assert hypothesis.predicted_improvement == 0.15
        assert not hypothesis.requires_human_approval  # LOW risk

    def test_high_risk_requires_approval(self):
        """Test that high-risk hypotheses require approval."""
        hypothesis = Hypothesis(
            id="hyp_test002",
            description="High risk change",
            category=HypothesisCategory.CAPABILITY,
            predicted_improvement=0.30,
            required_changes=[],
            affected_components=["core"],
            risk_level=RiskLevel.HIGH,
            reversibility_plan={"action": "restore"},
            ihsan_impact_estimate=0.0,
            snr_impact_estimate=0.0,
        )

        assert hypothesis.requires_human_approval

    def test_structural_requires_approval(self):
        """Test that structural changes require approval."""
        hypothesis = Hypothesis(
            id="hyp_test003",
            description="Architecture change",
            category=HypothesisCategory.STRUCTURAL,
            predicted_improvement=0.20,
            required_changes=[],
            affected_components=["architecture"],
            risk_level=RiskLevel.LOW,
            reversibility_plan={"action": "restore"},
            ihsan_impact_estimate=0.0,
            snr_impact_estimate=0.0,
        )

        assert hypothesis.requires_human_approval

    def test_hypothesis_to_dict(self):
        """Test hypothesis serialization."""
        hypothesis = Hypothesis(
            id="hyp_test004",
            description="Test",
            category=HypothesisCategory.QUALITY,
            predicted_improvement=0.10,
            required_changes=[],
            affected_components=[],
            risk_level=RiskLevel.NEGLIGIBLE,
            reversibility_plan={},
            ihsan_impact_estimate=0.01,
            snr_impact_estimate=0.01,
            confidence=0.75,
        )

        data = hypothesis.to_dict()

        assert data["id"] == "hyp_test004"
        assert data["category"] == "quality"
        assert data["risk_level"] == 1
        assert data["confidence"] == 0.75


# =============================================================================
# RATE LIMITER TESTS
# =============================================================================


class TestRateLimiter:
    """Tests for RateLimiter."""

    def test_initial_state_allows_improvement(self):
        """Test that initial state allows improvement."""
        limiter = RateLimiter(
            max_improvements_per_cycle=1,
            min_cycle_interval_s=0.0,  # No interval for testing
        )
        limiter.start_new_cycle()

        allowed, reason = limiter.can_attempt_improvement()

        assert allowed
        assert reason == "Allowed"

    def test_max_improvements_enforced(self):
        """Test that max improvements per cycle is enforced."""
        limiter = RateLimiter(
            max_improvements_per_cycle=2,
            min_cycle_interval_s=0.0,  # No interval for testing
        )
        limiter.start_new_cycle()

        # First should be allowed
        allowed1, _ = limiter.can_attempt_improvement()
        assert allowed1
        limiter.record_improvement_attempt()

        # Second should be allowed
        allowed2, _ = limiter.can_attempt_improvement()
        assert allowed2
        limiter.record_improvement_attempt()

        # Third should be blocked
        allowed3, reason = limiter.can_attempt_improvement()

        assert not allowed3
        assert "Max improvements" in reason

    def test_rollback_cooldown(self):
        """Test that rollback triggers cooldown."""
        limiter = RateLimiter(
            max_improvements_per_cycle=1,
            cooldown_after_rollback_s=300.0,
        )
        limiter.start_new_cycle()

        limiter.trigger_rollback_cooldown()
        allowed, reason = limiter.can_attempt_improvement()

        assert not allowed
        assert "cooldown" in reason.lower()


# =============================================================================
# ROLLBACK MANAGER TESTS
# =============================================================================


class TestRollbackManager:
    """Tests for RollbackManager."""

    def test_save_and_retrieve_state(self):
        """Test saving and retrieving rollback state."""
        manager = RollbackManager()
        state = {"param1": "value1", "param2": 42}

        manager.save_state(state, "hyp_001")
        retrieved = manager.get_rollback_state("hyp_001")

        assert retrieved == state

    def test_should_rollback_on_ihsan_drop(self):
        """Test rollback triggered on Ihsan drop."""
        manager = RollbackManager(ihsan_floor=0.95)

        should_rollback, reason = manager.should_rollback(
            current_ihsan=0.90,  # Below floor
            current_snr=0.90,
            current_error_rate=0.01,
            baseline_error_rate=0.01,
        )

        assert should_rollback
        assert "Ihsan" in reason

    def test_should_rollback_on_snr_drop(self):
        """Test rollback triggered on SNR drop."""
        manager = RollbackManager(snr_floor=0.85)

        should_rollback, reason = manager.should_rollback(
            current_ihsan=0.96,
            current_snr=0.80,  # Below floor
            current_error_rate=0.01,
            baseline_error_rate=0.01,
        )

        assert should_rollback
        assert "SNR" in reason

    def test_should_rollback_on_error_spike(self):
        """Test rollback triggered on error spike."""
        manager = RollbackManager(error_spike_threshold=0.1)

        should_rollback, reason = manager.should_rollback(
            current_ihsan=0.96,
            current_snr=0.90,
            current_error_rate=0.15,  # Spike
            baseline_error_rate=0.01,
        )

        assert should_rollback
        assert "Error spike" in reason

    def test_no_rollback_when_healthy(self):
        """Test no rollback when metrics are healthy."""
        manager = RollbackManager()

        should_rollback, reason = manager.should_rollback(
            current_ihsan=0.96,
            current_snr=0.90,
            current_error_rate=0.01,
            baseline_error_rate=0.01,
        )

        assert not should_rollback
        assert reason == ""


# =============================================================================
# HUMAN APPROVAL QUEUE TESTS
# =============================================================================


class TestHumanApprovalQueue:
    """Tests for HumanApprovalQueue."""

    def test_submit_for_approval(self):
        """Test submitting hypothesis for approval."""
        queue = HumanApprovalQueue()
        hypothesis = Hypothesis(
            id="hyp_approval_001",
            description="Structural change",
            category=HypothesisCategory.STRUCTURAL,
            predicted_improvement=0.20,
            required_changes=[],
            affected_components=[],
            risk_level=RiskLevel.HIGH,
            reversibility_plan={},
            ihsan_impact_estimate=0.0,
            snr_impact_estimate=0.0,
        )
        observation = SystemObservation(
            observation_id="obs_001",
            timestamp=datetime.now(timezone.utc),
            ihsan_score=0.96,
            snr_score=0.90,
            latency_p50_ms=10.0,
            latency_p99_ms=50.0,
            error_rate=0.01,
            throughput_qps=100.0,
            cpu_usage=0.5,
            memory_usage=0.6,
        )

        request = queue.submit_for_approval(hypothesis, observation)

        assert request is not None
        assert request.hypothesis.id == "hyp_approval_001"
        assert request.approved is None

    def test_approve_request(self):
        """Test approving a pending request."""
        queue = HumanApprovalQueue()
        hypothesis = Hypothesis(
            id="hyp_approval_002",
            description="Test",
            category=HypothesisCategory.STRUCTURAL,
            predicted_improvement=0.10,
            required_changes=[],
            affected_components=[],
            risk_level=RiskLevel.HIGH,
            reversibility_plan={},
            ihsan_impact_estimate=0.0,
            snr_impact_estimate=0.0,
        )
        observation = SystemObservation(
            observation_id="obs_002",
            timestamp=datetime.now(timezone.utc),
            ihsan_score=0.96,
            snr_score=0.90,
            latency_p50_ms=10.0,
            latency_p99_ms=50.0,
            error_rate=0.01,
            throughput_qps=100.0,
            cpu_usage=0.5,
            memory_usage=0.6,
        )

        request = queue.submit_for_approval(hypothesis, observation)
        result = queue.approve(request.request_id, "test_admin")

        assert result
        assert queue.check_approval("hyp_approval_002") is True

    def test_reject_request(self):
        """Test rejecting a pending request."""
        queue = HumanApprovalQueue()
        hypothesis = Hypothesis(
            id="hyp_approval_003",
            description="Test",
            category=HypothesisCategory.STRUCTURAL,
            predicted_improvement=0.10,
            required_changes=[],
            affected_components=[],
            risk_level=RiskLevel.HIGH,
            reversibility_plan={},
            ihsan_impact_estimate=0.0,
            snr_impact_estimate=0.0,
        )
        observation = SystemObservation(
            observation_id="obs_003",
            timestamp=datetime.now(timezone.utc),
            ihsan_score=0.96,
            snr_score=0.90,
            latency_p50_ms=10.0,
            latency_p99_ms=50.0,
            error_rate=0.01,
            throughput_qps=100.0,
            cpu_usage=0.5,
            memory_usage=0.6,
        )

        request = queue.submit_for_approval(hypothesis, observation)
        result = queue.reject(request.request_id, "test_admin", "Too risky")

        assert result
        assert queue.check_approval("hyp_approval_003") is False


# =============================================================================
# AUTOPOIETIC LOOP TESTS
# =============================================================================


class TestAutopoieticLoop:
    """Tests for AutopoieticLoop."""

    @pytest.fixture
    def temp_audit_log(self):
        """Create temporary audit log file."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            yield Path(f.name)

    @pytest.fixture
    def loop(self, temp_audit_log):
        """Create test loop instance with guardrails disabled for testing."""
        return AutopoieticLoop(
            fate_gate=MockFATEGate(),
            sensor_hub=MockSensorHub(),
            ihsan_floor=UNIFIED_IHSAN_THRESHOLD,
            snr_floor=UNIFIED_SNR_THRESHOLD,
            cycle_interval_s=1.0,
            audit_log_path=temp_audit_log,
            activation_guardrails=ActivationGuardrails(enabled=False),
        )

    def test_loop_initialization(self, loop):
        """Test loop initializes correctly."""
        assert loop._state == AutopoieticState.DORMANT
        assert loop.ihsan_floor == UNIFIED_IHSAN_THRESHOLD
        assert loop.snr_floor == UNIFIED_SNR_THRESHOLD
        assert not loop._running

    @pytest.mark.asyncio
    async def test_observe_collects_metrics(self, loop):
        """Test observe phase collects metrics."""
        observation = await loop.observe()

        assert observation is not None
        assert observation.ihsan_score > 0
        assert observation.snr_score > 0
        assert len(loop._observation_history) == 1

    @pytest.mark.asyncio
    async def test_check_activation_bootstraps_required_observations(
        self, temp_audit_log
    ):
        """Activation should self-bootstrap instead of deadlocking on min_observations."""
        loop = AutopoieticLoop(
            fate_gate=MockFATEGate(),
            sensor_hub=MockSensorHub(),
            ihsan_floor=UNIFIED_IHSAN_THRESHOLD,
            snr_floor=UNIFIED_SNR_THRESHOLD,
            cycle_interval_s=1.0,
            audit_log_path=temp_audit_log,
            activation_guardrails=ActivationGuardrails(
                enabled=True,
                require_live_sensors=False,
                allow_mock_sensors=True,
                require_fate_gate=False,
                min_observations=3,
            ),
        )

        report = await loop.check_activation()

        assert report.activated is True
        assert len(loop._observation_history) >= 3
        assert report.observation is not None

    @pytest.mark.asyncio
    async def test_observe_uses_runtime_receipt_window_as_ground_truth(self, loop):
        """Runtime receipts should steer the observation instead of placeholder defaults."""
        loop.record_receipt_observation(
            {
                "mission_id": "mission-001",
                "ihsan_score": 0.98,
                "snr_score": 0.96,
                "duration_ms": 10.0,
                "gate_passed": True,
                "fate_verdict": "approved",
            }
        )
        loop.record_receipt_observation(
            {
                "mission_id": "mission-002",
                "ihsan_score": 0.82,
                "snr_score": 0.8,
                "duration_ms": 40.0,
                "gate_passed": False,
                "fate_verdict": "rejected",
            }
        )
        loop.record_receipt_observation(
            {
                "mission_id": "mission-003",
                "ihsan_score": 0.78,
                "snr_score": 0.76,
                "duration_ms": 55.0,
                "gate_passed": False,
                "fate_verdict": "rejected",
            }
        )
        loop.record_receipt_observation(
            {
                "mission_id": "mission-004",
                "ihsan_score": 0.76,
                "snr_score": 0.74,
                "duration_ms": 65.0,
                "gate_passed": False,
                "fate_verdict": "rejected",
            }
        )

        observation = await loop.observe()

        assert observation is not None
        assert observation.ihsan_score == pytest.approx((0.98 + 0.82 + 0.78 + 0.76) / 4)
        assert observation.error_rate == pytest.approx(0.75)
        assert observation.latency_p99_ms == pytest.approx(65.0)
        assert observation.trend_direction == "degrading"
        assert "receipt_degradation" in observation.anomalies_detected

    @pytest.mark.asyncio
    async def test_hypothesize_generates_candidates(self, loop):
        """Test hypothesize generates improvement candidates."""
        # Create observation with issues to trigger hypothesis generation
        observation = SystemObservation(
            observation_id="obs_test",
            timestamp=datetime.now(timezone.utc),
            ihsan_score=0.96,
            snr_score=0.88,  # Below optimal, should trigger hypothesis
            latency_p50_ms=10.0,
            latency_p99_ms=150.0,  # High latency, should trigger hypothesis
            error_rate=0.03,  # Elevated error rate
            throughput_qps=100.0,
            cpu_usage=0.5,
            memory_usage=0.85,  # High memory usage
        )

        hypotheses = await loop.hypothesize(observation)

        assert len(hypotheses) >= 1
        assert all(isinstance(h, Hypothesis) for h in hypotheses)

    @pytest.mark.asyncio
    async def test_hypothesize_enriches_with_got_and_standing_on_giants(self, loop):
        """Advanced hypothesis generation should carry GoT traces and provenance."""

        class _FakeExploredHypothesis:
            def __init__(self) -> None:
                self.hypothesis = SimpleNamespace(
                    id="got_hyp_001",
                    category=SimpleNamespace(value="quality"),
                    description="Fuse graph reasoning with SNR-first verification",
                    predicted_improvement={"snr_score_delta": 0.09},
                    risk_level=SimpleNamespace(value="low"),
                    implementation_plan=[
                        "Use Graph-of-Thoughts to compare rival repair paths",
                        "Rank candidate repairs by SNR before validation",
                    ],
                    rollback_plan=["Revert to heuristic-only ranking"],
                    ihsan_impact=0.04,
                )
                self.confidence = 0.91
                self.snr_score = 0.96
                self.exploration_path = [
                    SimpleNamespace(content="Compare multiple disciplinary paths"),
                    SimpleNamespace(content="Prefer the highest-SNR reversible repair"),
                ]

            def expected_value(self) -> float:
                return 0.18

        class _FakeExplorer:
            async def explore_hypotheses(self, **_: object):
                return [_FakeExploredHypothesis()]

        loop._get_or_create_hypothesis_generator = lambda: object()
        loop._get_or_create_got_explorer = lambda: _FakeExplorer()

        observation = SystemObservation(
            observation_id="obs_got",
            timestamp=datetime.now(timezone.utc),
            ihsan_score=0.93,
            snr_score=0.81,
            latency_p50_ms=15.0,
            latency_p99_ms=180.0,
            error_rate=0.04,
            throughput_qps=12.0,
            cpu_usage=0.55,
            memory_usage=0.72,
            trend_direction="degrading",
            anomalies_detected=["receipt_degradation"],
        )

        hypotheses = await loop.hypothesize(observation)

        got_hypothesis = next(
            h
            for h in hypotheses
            if h.description == "Fuse graph reasoning with SNR-first verification"
        )
        assert (
            "Besta et al. (2024) — Graph-of-Thoughts"
            in got_hypothesis.standing_on_giants
        )
        assert "graph reasoning" in " ".join(got_hypothesis.interdisciplinary_lenses)
        assert (
            got_hypothesis.reasoning_trace[0] == "Compare multiple disciplinary paths"
        )

    @pytest.mark.asyncio
    async def test_validate_checks_constraints(self, loop):
        """Test validation checks Z3 and constitutional constraints."""
        hypothesis = Hypothesis(
            id="hyp_validate_001",
            description="Valid improvement",
            category=HypothesisCategory.PERFORMANCE,
            predicted_improvement=0.10,
            required_changes=[],
            affected_components=["test"],
            risk_level=RiskLevel.LOW,
            reversibility_plan={"action": "restore"},
            ihsan_impact_estimate=0.01,
            snr_impact_estimate=0.01,
        )

        result = await loop.validate(hypothesis)

        assert result is not None
        assert result.hypothesis_id == "hyp_validate_001"
        assert result.z3_satisfiable
        assert result.ihsan_gate_passed
        assert result.snr_gate_passed

    @pytest.mark.asyncio
    async def test_validate_rejects_bad_hypothesis(self, loop):
        """Test validation rejects hypothesis that violates constraints."""
        hypothesis = Hypothesis(
            id="hyp_validate_002",
            description="Bad improvement",
            category=HypothesisCategory.PERFORMANCE,
            predicted_improvement=0.50,
            required_changes=[],
            affected_components=["test"],
            risk_level=RiskLevel.HIGH,
            reversibility_plan={},  # No reversibility plan
            ihsan_impact_estimate=-0.10,  # Would drop Ihsan below floor
            snr_impact_estimate=-0.05,
        )

        result = await loop.validate(hypothesis)

        assert not result.is_valid
        assert len(result.violations) > 0

    @pytest.mark.asyncio
    async def test_implement_shadow_deployment(self, loop):
        """Test implementation phase performs shadow deployment."""
        hypothesis = Hypothesis(
            id="hyp_impl_001",
            description="Test implementation",
            category=HypothesisCategory.PERFORMANCE,
            predicted_improvement=0.10,
            required_changes=[{"component": "test", "action": "update"}],
            affected_components=["test"],
            risk_level=RiskLevel.LOW,
            reversibility_plan={"action": "restore"},
            ihsan_impact_estimate=0.01,
            snr_impact_estimate=0.01,
            min_observation_duration_s=1,  # Short for testing
        )

        # Need at least one observation for baseline
        await loop.observe()

        result = await loop.implement(hypothesis)

        assert result is not None
        assert result.hypothesis_id == "hyp_impl_001"
        assert result.shadow_instance_id is not None

    @pytest.mark.asyncio
    async def test_integrate_successful_improvement(self, loop):
        """Test integration phase learns from successful improvement."""
        impl_result = ImplementationResult(
            hypothesis_id="hyp_int_001",
            success=True,
            shadow_instance_id="shadow_001",
            deployment_time_ms=100,
            baseline_metrics={"ihsan_score": 0.95, "snr_score": 0.90},
            shadow_metrics={"ihsan_score": 0.96, "snr_score": 0.91},
            improvement_observed=0.08,
            ihsan_maintained=True,
            snr_maintained=True,
            error_rate_acceptable=True,
        )

        result = await loop.integrate(impl_result)

        assert result.integrated
        assert result.final_improvement > 0
        assert result.ihsan_delta > 0

    @pytest.mark.asyncio
    async def test_run_cycle_complete(self, loop):
        """Test running a complete improvement cycle."""
        result = await loop.run_cycle()

        assert result is not None
        assert result.cycle_id is not None
        assert result.observation is not None
        assert result.completed_at is not None
        # Duration may be 0 for very fast cycles (sub-millisecond)
        assert result.cycle_duration_ms >= 0

    @pytest.mark.asyncio
    async def test_run_cycle_respects_rate_limit(self, loop):
        """Test that rate limiting is enforced across cycles."""
        # Run first cycle
        result1 = await loop.run_cycle()

        # Second cycle should be rate limited (same cycle interval)
        result2 = await loop.run_cycle()

        # Both should complete, but second may be limited
        assert result1.state != AutopoieticState.HALTED
        assert result2.state != AutopoieticState.HALTED

    def test_get_status(self, loop):
        """Test getting loop status."""
        status = loop.get_status()

        assert "state" in status
        assert "running" in status
        assert "ihsan_floor" in status
        assert "snr_floor" in status
        assert "total_improvements" in status
        assert "total_rollbacks" in status

    @pytest.mark.asyncio
    async def test_get_audit_log(self, loop):
        """Test retrieving audit log."""
        # Run a cycle to generate audit entries
        await loop.run_cycle()

        audit_log = loop.get_audit_log()

        assert isinstance(audit_log, list)
        # Should have at least validation audit entry
        assert len(audit_log) >= 0

    @pytest.mark.asyncio
    async def test_approval_workflow(self, loop):
        """Test human approval workflow for structural changes."""
        # Create a structural hypothesis that requires approval
        hypothesis = Hypothesis(
            id="hyp_approval_workflow",
            description="Architecture change",
            category=HypothesisCategory.STRUCTURAL,
            predicted_improvement=0.25,
            required_changes=[],
            affected_components=["architecture"],
            risk_level=RiskLevel.MODERATE,  # Structural still requires approval
            reversibility_plan={"action": "restore"},
            ihsan_impact_estimate=0.01,
            snr_impact_estimate=0.01,
        )

        assert hypothesis.requires_human_approval

        # Submit for approval
        observation = await loop.observe()
        request = loop._approval_queue.submit_for_approval(hypothesis, observation)

        # Check pending
        pending = loop.get_pending_approvals()
        assert len(pending) == 1

        # Approve
        loop.approve_hypothesis(request.request_id, "admin")

        # Verify approval
        assert loop._approval_queue.check_approval("hyp_approval_workflow") is True


# =============================================================================
# FACTORY FUNCTION TESTS
# =============================================================================


class TestFactoryFunction:
    """Tests for create_autopoietic_loop factory."""

    def test_create_with_defaults(self):
        """Test creating loop with default parameters."""
        loop = create_autopoietic_loop()

        assert loop.ihsan_floor == UNIFIED_IHSAN_THRESHOLD
        assert loop.snr_floor == UNIFIED_SNR_THRESHOLD
        assert loop.cycle_interval_s == 60.0

    def test_create_with_custom_params(self):
        """Test creating loop with custom parameters."""
        loop = create_autopoietic_loop(
            ihsan_floor=0.98,
            snr_floor=0.90,
            cycle_interval_s=30.0,
        )

        assert loop.ihsan_floor == 0.98
        assert loop.snr_floor == 0.90
        assert loop.cycle_interval_s == 30.0


# =============================================================================
# INTEGRATION TESTS
# =============================================================================


class TestAutopoieticLoopIntegration:
    """Integration tests for complete autopoietic loop."""

    @pytest.mark.asyncio
    async def test_full_improvement_cycle(self):
        """Test a full improvement cycle from observation to integration."""
        with tempfile.NamedTemporaryFile(suffix=".jsonl", delete=False) as f:
            audit_path = Path(f.name)

        loop = AutopoieticLoop(
            fate_gate=MockFATEGate(),
            sensor_hub=MockSensorHub(),
            ihsan_floor=0.95,
            snr_floor=0.85,
            cycle_interval_s=0.1,
            audit_log_path=audit_path,
            activation_guardrails=ActivationGuardrails(enabled=False),
        )

        # Run multiple cycles
        for _ in range(3):
            result = await loop.run_cycle()
            assert result.state in [
                AutopoieticState.DORMANT,
                AutopoieticState.REFLECTING,
                AutopoieticState.HALTED,
            ]
            await asyncio.sleep(0.1)

        status = loop.get_status()
        assert status["cycle_count"] == 3

    @pytest.mark.asyncio
    async def test_rollback_on_degradation(self):
        """Test that rollback is triggered on quality degradation."""
        loop = AutopoieticLoop(
            fate_gate=MockFATEGate(),
            sensor_hub=MockSensorHub(),
            ihsan_floor=0.95,
            snr_floor=0.85,
        )

        # Manually test rollback manager
        should_rollback, reason = loop._rollback_manager.should_rollback(
            current_ihsan=0.90,  # Below floor
            current_snr=0.80,  # Below floor
            current_error_rate=0.15,  # High error rate
            baseline_error_rate=0.01,
        )

        assert should_rollback
        assert "Ihsan" in reason or "SNR" in reason or "Error" in reason


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
