"""
Tests for Treasury Mode - Graceful Degradation System

Validates the state machine behavior for transitioning between:
- ETHICAL: Full operation with ethical trading
- HIBERNATION: Minimal compute to preserve reserves
- EMERGENCY: Community funding and treasury unlock

GAP-C4: Ensures the Wealth Engine can survive unethical market conditions.
"""

import pytest
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from core.sovereign.treasury_mode import (
    TreasuryMode,
    TreasuryState,
    TreasuryController,
    TreasuryEvent,
    TransitionTrigger,
    TransitionEvent,
    EthicsAssessment,
    TreasuryPersistence,
    create_treasury_controller,
    ETHICS_THRESHOLD_HIBERNATION,
    ETHICS_THRESHOLD_RECOVERY,
    RESERVES_THRESHOLD_EMERGENCY,
    RESERVES_THRESHOLD_HIBERNATION,
    EMERGENCY_TREASURY_UNLOCK_PERCENT,
    COMPUTE_MULTIPLIERS,
    DEFAULT_BURN_RATE,
)


# =============================================================================
# FIXTURES
# =============================================================================

@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir) / "test_memory.db"


@pytest.fixture
def controller(temp_db):
    """Create a test treasury controller."""
    return TreasuryController(
        initial_reserves_seed=10000.0,
        initial_treasury_seed=50000.0,
        db_path=temp_db,
    )


@pytest.fixture
def low_reserves_controller(temp_db):
    """Create a controller with low reserves."""
    ctrl = TreasuryController(
        initial_reserves_seed=500.0,  # Very low
        initial_treasury_seed=50000.0,
        db_path=temp_db,
    )
    return ctrl


# =============================================================================
# ENUM TESTS
# =============================================================================

class TestTreasuryMode:
    """Test TreasuryMode enum."""

    def test_mode_values(self):
        """Test that modes have expected string values."""
        assert TreasuryMode.ETHICAL.value == "ethical"
        assert TreasuryMode.HIBERNATION.value == "hibernation"
        assert TreasuryMode.EMERGENCY.value == "emergency"

    def test_mode_from_string(self):
        """Test mode creation from string."""
        assert TreasuryMode("ethical") == TreasuryMode.ETHICAL
        assert TreasuryMode("hibernation") == TreasuryMode.HIBERNATION
        assert TreasuryMode("emergency") == TreasuryMode.EMERGENCY


class TestTransitionTrigger:
    """Test TransitionTrigger enum."""

    def test_trigger_values(self):
        """Test trigger string values."""
        assert TransitionTrigger.MARKET_ETHICS_DEGRADED.value == "market_ethics_degraded"
        assert TransitionTrigger.RESERVES_DEPLETED.value == "reserves_depleted"
        assert TransitionTrigger.MANUAL_OVERRIDE.value == "manual_override"


# =============================================================================
# DATA CLASS TESTS
# =============================================================================

class TestTreasuryState:
    """Test TreasuryState dataclass."""

    def test_state_creation(self):
        """Test basic state creation."""
        state = TreasuryState(
            mode=TreasuryMode.ETHICAL,
            reserves_days=30.0,
            ethical_score=0.95,
            last_transition=datetime.utcnow(),
            transition_reason="Test",
        )
        assert state.mode == TreasuryMode.ETHICAL
        assert state.reserves_days == 30.0
        assert state.ethical_score == 0.95

    def test_state_serialization(self):
        """Test state serialization and deserialization."""
        original = TreasuryState(
            mode=TreasuryMode.HIBERNATION,
            reserves_days=15.0,
            ethical_score=0.55,
            last_transition=datetime.utcnow(),
            transition_reason="Low ethics",
            burn_rate_seed_per_day=25.0,
            total_reserves_seed=375.0,
        )

        # To dict and back
        as_dict = original.to_dict()
        restored = TreasuryState.from_dict(as_dict)

        assert restored.mode == original.mode
        assert restored.reserves_days == original.reserves_days
        assert restored.ethical_score == original.ethical_score
        assert restored.burn_rate_seed_per_day == original.burn_rate_seed_per_day

    def test_state_json_roundtrip(self):
        """Test JSON serialization roundtrip."""
        original = TreasuryState(
            mode=TreasuryMode.EMERGENCY,
            reserves_days=5.0,
            ethical_score=0.40,
            last_transition=datetime.utcnow(),
            transition_reason="Critical",
        )

        json_str = original.to_json()
        restored = TreasuryState.from_json(json_str)

        assert restored.mode == original.mode
        assert restored.reserves_days == original.reserves_days


class TestEthicsAssessment:
    """Test EthicsAssessment dataclass."""

    def test_assessment_creation(self):
        """Test creating an ethics assessment."""
        assessment = EthicsAssessment(
            overall_score=0.85,
            transparency_score=0.80,
            fairness_score=0.90,
            sustainability_score=0.85,
            compliance_score=0.88,
            ihsan_alignment=0.82,
            confidence=0.95,
        )
        assert assessment.overall_score == 0.85
        assert assessment.confidence == 0.95

    def test_assessment_serialization(self):
        """Test assessment to_dict."""
        assessment = EthicsAssessment(
            overall_score=0.75,
            data_sources=["market_feed", "compliance_api"],
        )
        as_dict = assessment.to_dict()

        assert as_dict["overall_score"] == 0.75
        assert "market_feed" in as_dict["data_sources"]


# =============================================================================
# CONTROLLER TESTS
# =============================================================================

class TestTreasuryController:
    """Test TreasuryController class."""

    def test_initialization(self, controller):
        """Test controller initializes in ETHICAL mode."""
        assert controller.mode == TreasuryMode.ETHICAL
        assert controller.state.ethical_score == 1.0
        assert controller.state.reserves_days > 0

    def test_initial_burn_rate(self, controller):
        """Test initial burn rate is calculated correctly."""
        expected_burn = DEFAULT_BURN_RATE * COMPUTE_MULTIPLIERS["ethical"]
        assert controller.state.burn_rate_seed_per_day == expected_burn

    def test_calculate_burn_rate_by_mode(self, controller):
        """Test burn rate calculation for different modes."""
        ethical_burn = controller.calculate_burn_rate(TreasuryMode.ETHICAL)
        hibernate_burn = controller.calculate_burn_rate(TreasuryMode.HIBERNATION)
        emergency_burn = controller.calculate_burn_rate(TreasuryMode.EMERGENCY)

        assert ethical_burn == DEFAULT_BURN_RATE * COMPUTE_MULTIPLIERS["ethical"]
        assert hibernate_burn == DEFAULT_BURN_RATE * COMPUTE_MULTIPLIERS["hibernation"]
        assert emergency_burn == DEFAULT_BURN_RATE * COMPUTE_MULTIPLIERS["emergency"]
        assert hibernate_burn < ethical_burn
        assert emergency_burn < hibernate_burn


class TestMarketEthicsEvaluation:
    """Test market ethics evaluation."""

    def test_default_ethics_evaluation(self, controller):
        """Test ethics evaluation without market data."""
        score = controller.evaluate_market_ethics()
        assert 0.0 <= score <= 1.0
        assert controller.state.ethical_score == score

    def test_ethics_evaluation_with_good_data(self, controller):
        """Test ethics evaluation with positive market data."""
        market_data = {
            "transparency_score": 0.95,
            "fairness_score": 0.90,
            "manipulation_detected": False,
            "volatility_index": 0.20,
            "liquidity_score": 0.85,
            "compliance_score": 0.95,
        }
        score = controller.evaluate_market_ethics(market_data)
        assert score > 0.70

    def test_ethics_evaluation_with_manipulation(self, controller):
        """Test that manipulation detection severely impacts score."""
        market_data = {
            "manipulation_detected": True,
            "transparency_score": 0.90,
            "fairness_score": 0.90,
        }
        score = controller.evaluate_market_ethics(market_data)
        # Manipulation should significantly reduce score (fairness drops to 0.20)
        assert score < 0.70  # Manipulation should tank the score

    def test_ethics_evaluation_with_poor_data(self, controller):
        """Test ethics evaluation with negative market conditions."""
        market_data = {
            "transparency_score": 0.30,
            "fairness_score": 0.25,
            "volatility_index": 0.90,
            "liquidity_score": 0.20,
            "compliance_flags": ["violation_1", "violation_2"],
        }
        score = controller.evaluate_market_ethics(market_data)
        # Poor conditions should result in low score
        assert score < 0.70


class TestModeTransitions:
    """Test mode transition logic."""

    def test_valid_transition_ethical_to_hibernation(self, controller):
        """Test valid transition from ETHICAL to HIBERNATION."""
        success, msg = controller.transition_mode(
            TreasuryMode.HIBERNATION,
            "Market ethics degraded",
            TransitionTrigger.MARKET_ETHICS_DEGRADED,
        )
        assert success
        assert controller.mode == TreasuryMode.HIBERNATION

    def test_invalid_direct_transition_ethical_to_emergency(self, controller):
        """Test that direct ETHICAL to EMERGENCY is blocked."""
        success, msg = controller.transition_mode(
            TreasuryMode.EMERGENCY,
            "Trying to skip hibernation",
            TransitionTrigger.RESERVES_DEPLETED,
        )
        assert not success
        assert controller.mode == TreasuryMode.ETHICAL

    def test_forced_transition_bypasses_validation(self, controller):
        """Test that forced transitions bypass validation."""
        success, msg = controller.transition_mode(
            TreasuryMode.EMERGENCY,
            "Admin override",
            TransitionTrigger.MANUAL_OVERRIDE,
            force=True,
        )
        assert success
        assert controller.mode == TreasuryMode.EMERGENCY

    def test_transition_updates_burn_rate(self, controller):
        """Test that transitions update the burn rate."""
        initial_burn = controller.state.burn_rate_seed_per_day

        controller.transition_mode(
            TreasuryMode.HIBERNATION,
            "Test transition",
            TransitionTrigger.MARKET_ETHICS_DEGRADED,
        )

        assert controller.state.burn_rate_seed_per_day < initial_burn

    def test_transition_records_history(self, controller):
        """Test that transitions are recorded in history."""
        controller.transition_mode(
            TreasuryMode.HIBERNATION,
            "Test transition",
            TransitionTrigger.MARKET_ETHICS_DEGRADED,
        )

        history = controller._persistence.get_transition_history(limit=1)
        assert len(history) == 1
        assert history[0].to_mode == TreasuryMode.HIBERNATION


class TestConditionChecks:
    """Test condition check methods."""

    def test_should_hibernate_when_ethics_low(self, controller):
        """Test should_hibernate returns True when ethics are low."""
        # Set low ethics
        controller._state.ethical_score = 0.50
        assert controller.should_hibernate()

    def test_should_hibernate_false_when_ethics_ok(self, controller):
        """Test should_hibernate returns False when ethics are good."""
        controller._state.ethical_score = 0.85
        assert not controller.should_hibernate()

    def test_should_emergency_when_reserves_low(self, controller):
        """Test should_emergency when reserves are critical."""
        # First transition to hibernation
        controller.transition_mode(
            TreasuryMode.HIBERNATION,
            "Test",
            TransitionTrigger.MARKET_ETHICS_DEGRADED,
        )
        # Set low reserves
        controller._state.reserves_days = 5.0
        assert controller.should_emergency()

    def test_should_emergency_false_in_ethical_mode(self, controller):
        """Test should_emergency is False in ETHICAL mode even with low reserves."""
        controller._state.reserves_days = 3.0
        assert not controller.should_emergency()

    def test_should_recover_from_hibernation(self, controller):
        """Test recovery conditions from hibernation."""
        controller.transition_mode(
            TreasuryMode.HIBERNATION,
            "Test",
            TransitionTrigger.MARKET_ETHICS_DEGRADED,
        )
        controller._state.ethical_score = 0.80
        controller._state.reserves_days = 35.0
        assert controller.should_recover()

    def test_should_recover_from_emergency(self, controller):
        """Test recovery conditions from emergency."""
        controller.transition_mode(
            TreasuryMode.HIBERNATION,
            "Test",
            TransitionTrigger.MARKET_ETHICS_DEGRADED,
        )
        controller.transition_mode(
            TreasuryMode.EMERGENCY,
            "Test",
            TransitionTrigger.RESERVES_DEPLETED,
        )
        controller._state.reserves_days = 10.0
        assert controller.should_recover()


class TestEmergencyProtocol:
    """Test emergency mode specific behavior."""

    def test_emergency_unlocks_treasury(self, controller):
        """Test that entering emergency mode unlocks treasury."""
        initial_locked = controller.state.locked_treasury_seed
        expected_unlock = initial_locked * EMERGENCY_TREASURY_UNLOCK_PERCENT

        # Force transition to emergency
        controller.transition_mode(
            TreasuryMode.EMERGENCY,
            "Critical reserves",
            TransitionTrigger.RESERVES_DEPLETED,
            force=True,
        )

        assert controller.state.unlocked_treasury_seed == expected_unlock
        assert controller.state.locked_treasury_seed == initial_locked - expected_unlock


class TestReservesManagement:
    """Test reserves management."""

    def test_update_reserves_positive(self, controller):
        """Test adding reserves."""
        initial = controller.state.total_reserves_seed
        controller.update_reserves(1000.0, "Community contribution")
        assert controller.state.total_reserves_seed == initial + 1000.0

    def test_update_reserves_negative(self, controller):
        """Test subtracting reserves."""
        initial = controller.state.total_reserves_seed
        controller.update_reserves(-500.0, "Operational costs")
        assert controller.state.total_reserves_seed == initial - 500.0

    def test_update_reserves_cannot_go_negative(self, controller):
        """Test reserves cannot go below zero."""
        controller.update_reserves(-1000000.0, "Massive loss")
        assert controller.state.total_reserves_seed >= 0

    def test_update_reserves_recalculates_days(self, controller):
        """Test that updating reserves recalculates days remaining."""
        burn_rate = controller.state.burn_rate_seed_per_day
        controller.update_reserves(1000.0, "Test")

        expected_days = controller.state.total_reserves_seed / burn_rate
        assert abs(controller.state.reserves_days - expected_days) < 0.01


class TestEventSystem:
    """Test event emission and handling."""

    def test_event_handler_registration(self, controller):
        """Test registering an event handler."""
        events_received = []

        def handler(data):
            events_received.append(data)

        controller.on_event(TreasuryEvent.RESERVES_UPDATE, handler)
        controller.update_reserves(100.0, "Test")

        assert len(events_received) == 1
        assert events_received[0]["amount"] == 100.0

    def test_mode_transition_emits_event(self, controller):
        """Test that mode transitions emit events."""
        events_received = []

        def handler(data):
            events_received.append(data)

        controller.on_event(TreasuryEvent.MODE_TRANSITION, handler)
        controller.transition_mode(
            TreasuryMode.HIBERNATION,
            "Test",
            TransitionTrigger.MARKET_ETHICS_DEGRADED,
        )

        assert len(events_received) == 1
        assert events_received[0]["to_mode"] == "hibernation"


class TestHealthCheck:
    """Test health check functionality."""

    def test_healthy_system(self, controller):
        """Test health check on healthy system."""
        health = controller.health_check()
        assert health["healthy"]
        assert len(health["warnings"]) == 0

    def test_low_reserves_warning(self, low_reserves_controller):
        """Test health check warns on low reserves."""
        health = low_reserves_controller.health_check()
        assert any("reserves" in w.lower() for w in health["warnings"])

    def test_low_ethics_warning(self, controller):
        """Test health check warns on low ethics."""
        controller._state.ethical_score = 0.55
        health = controller.health_check()
        assert any("ethics" in w.lower() for w in health["warnings"])


class TestPersistence:
    """Test state persistence."""

    def test_state_persists_across_instances(self, temp_db):
        """Test that state persists across controller instances."""
        # Create first controller
        ctrl1 = TreasuryController(
            initial_reserves_seed=10000.0,
            initial_treasury_seed=50000.0,
            db_path=temp_db,
        )
        ctrl1.transition_mode(
            TreasuryMode.HIBERNATION,
            "Test persistence",
            TransitionTrigger.MARKET_ETHICS_DEGRADED,
        )

        # Create second controller with same DB
        ctrl2 = TreasuryController(
            initial_reserves_seed=10000.0,
            initial_treasury_seed=50000.0,
            db_path=temp_db,
        )

        # Should load persisted state
        assert ctrl2.mode == TreasuryMode.HIBERNATION

    def test_transition_history_persists(self, temp_db):
        """Test that transition history persists."""
        ctrl1 = TreasuryController(
            initial_reserves_seed=10000.0,
            initial_treasury_seed=50000.0,
            db_path=temp_db,
        )
        ctrl1.transition_mode(
            TreasuryMode.HIBERNATION,
            "First transition",
            TransitionTrigger.MARKET_ETHICS_DEGRADED,
        )

        # Check history in new controller
        persistence = TreasuryPersistence(temp_db)
        history = persistence.get_transition_history()
        assert len(history) == 1
        assert history[0].reason == "First transition"


class TestFactoryFunction:
    """Test factory function."""

    def test_create_treasury_controller(self, temp_db):
        """Test create_treasury_controller factory."""
        controller = create_treasury_controller(
            initial_reserves=5000.0,
            initial_treasury=25000.0,
            db_path=temp_db,
        )
        assert controller is not None
        assert controller.mode == TreasuryMode.ETHICAL


# =============================================================================
# INTEGRATION TESTS
# =============================================================================

class TestFullWorkflow:
    """Integration tests for complete workflows."""

    @pytest.mark.asyncio
    async def test_full_degradation_and_recovery_cycle(self, temp_db):
        """Test complete degradation and recovery cycle."""
        controller = TreasuryController(
            initial_reserves_seed=5000.0,
            initial_treasury_seed=50000.0,
            db_path=temp_db,
        )

        # Phase 1: Start healthy
        assert controller.mode == TreasuryMode.ETHICAL

        # Phase 2: Market ethics degrade severely
        bad_market = {
            "manipulation_detected": True,
            "transparency_score": 0.10,
            "fairness_score": 0.10,
            "volatility_index": 0.95,
            "liquidity_score": 0.05,
            "compliance_flags": ["v1", "v2", "v3"],
        }
        score = controller.evaluate_market_ethics(bad_market)
        # Ensure score is low enough to trigger hibernation (< 0.60)
        assert score < ETHICS_THRESHOLD_HIBERNATION, f"Score {score} should be < {ETHICS_THRESHOLD_HIBERNATION}"
        assert controller.should_hibernate()

        # Phase 3: Transition to hibernation
        result = await controller.evaluate_and_transition(bad_market)
        assert result is not None
        assert controller.mode == TreasuryMode.HIBERNATION

        # Phase 4: Reserves deplete
        # In hibernation mode, burn rate is 25 SEED/day
        # To trigger emergency (< 7 days), need reserves < 175 SEED
        # Starting with 5000, need to remove > 4825
        controller.update_reserves(-4900.0, "Operational costs")
        # Verify reserves are low enough
        assert controller.state.reserves_days < RESERVES_THRESHOLD_EMERGENCY, \
            f"Reserves {controller.state.reserves_days} should be < {RESERVES_THRESHOLD_EMERGENCY}"
        assert controller.should_emergency()

        # Phase 5: Enter emergency
        result = await controller.evaluate_and_transition()
        assert result is not None
        assert controller.mode == TreasuryMode.EMERGENCY
        assert controller.state.unlocked_treasury_seed > 0

        # Phase 6: Community support arrives
        controller.update_reserves(10000.0, "Community contribution")
        assert controller.should_recover()

        # Phase 7: Begin recovery
        result = await controller.evaluate_and_transition()
        assert result is not None
        assert controller.mode == TreasuryMode.HIBERNATION

        # Phase 8: Market recovers
        good_market = {
            "manipulation_detected": False,
            "transparency_score": 0.90,
            "fairness_score": 0.85,
            "compliance_score": 0.95,
        }
        controller.evaluate_market_ethics(good_market)

        # Phase 9: Full recovery (if reserves sufficient)
        controller._state.reserves_days = 35.0
        assert controller.should_recover()

        result = await controller.evaluate_and_transition(good_market)
        assert result is not None
        assert controller.mode == TreasuryMode.ETHICAL


# =============================================================================
# RUN TESTS
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
