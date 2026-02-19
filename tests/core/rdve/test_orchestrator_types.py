"""
Tests for RDVE orchestrator data classes and enums (core.rdve.orchestrator).

Only tests the lightweight type layer — RDVEStage, RDVEStatus, CycleOutcome,
RDVEConfig, StageResult, and RDVECycleResult.  The RDVEOrchestrator class
itself is NOT tested here because it requires heavy external dependencies
(HypothesisGenerator, GoTHypothesisExplorer, AutopoieticLoop, etc.).

Covers:
    - Enum member values and counts
    - RDVEConfig defaults match core.integration.constants
    - StageResult construction
    - RDVECycleResult.to_dict() shape and content
    - RDVECycleResult.duration_ms property
    - Version/codename constants
"""

from datetime import datetime, timedelta, timezone

import pytest

from core.integration.constants import (
    SNR_THRESHOLD_T1_HIGH,
    STRICT_IHSAN_THRESHOLD,
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)
from core.rdve.orchestrator import (
    RDVE_CODENAME,
    RDVE_VERSION,
    STANDING_ON_GIANTS,
    CycleOutcome,
    RDVEConfig,
    RDVECycleResult,
    RDVEStage,
    RDVEStatus,
    StageResult,
)


# ============================================================================
# RDVEStage Enum
# ============================================================================


class TestRDVEStage:
    def test_has_eight_stages(self):
        assert len(RDVEStage) == 8

    def test_values_match_pipeline_order(self):
        expected = [
            "observe", "generate", "explore", "filter",
            "verify", "implement", "integrate", "learn",
        ]
        actual = [s.value for s in RDVEStage]
        assert actual == expected

    def test_is_str_enum(self):
        assert isinstance(RDVEStage.OBSERVE, str)
        assert RDVEStage.OBSERVE == "observe"


# ============================================================================
# RDVEStatus Enum
# ============================================================================


class TestRDVEStatus:
    def test_has_five_statuses(self):
        assert len(RDVEStatus) == 5

    def test_values(self):
        assert RDVEStatus.IDLE.value == "idle"
        assert RDVEStatus.RUNNING.value == "running"
        assert RDVEStatus.COMPLETED.value == "completed"
        assert RDVEStatus.FAILED.value == "failed"
        assert RDVEStatus.HALTED.value == "halted"


# ============================================================================
# CycleOutcome Enum
# ============================================================================


class TestCycleOutcome:
    def test_has_five_outcomes(self):
        assert len(CycleOutcome) == 5

    def test_values(self):
        assert CycleOutcome.DISCOVERY.value == "discovery"
        assert CycleOutcome.NO_SIGNAL.value == "no_signal"
        assert CycleOutcome.VERIFICATION_FAIL.value == "verification_fail"
        assert CycleOutcome.IMPLEMENTATION_FAIL.value == "implementation_fail"
        assert CycleOutcome.CONVERGED.value == "converged"


# ============================================================================
# RDVEConfig — Defaults Match Constitutional Constants
# ============================================================================


class TestRDVEConfig:
    def setup_method(self):
        self.config = RDVEConfig()

    def test_snr_floor_matches_unified_snr_threshold(self):
        assert self.config.snr_floor == pytest.approx(UNIFIED_SNR_THRESHOLD)

    def test_snr_target_matches_t1_high(self):
        assert self.config.snr_target == pytest.approx(SNR_THRESHOLD_T1_HIGH)

    def test_ihsan_floor_matches_unified_ihsan_threshold(self):
        assert self.config.ihsan_floor == pytest.approx(UNIFIED_IHSAN_THRESHOLD)

    def test_ihsan_strict_matches_strict_ihsan_threshold(self):
        assert self.config.ihsan_strict == pytest.approx(STRICT_IHSAN_THRESHOLD)

    def test_default_exploration_paths(self):
        assert self.config.num_exploration_paths == 5

    def test_default_convergence_window(self):
        assert self.config.convergence_window == 5

    def test_default_convergence_threshold(self):
        assert self.config.convergence_threshold == pytest.approx(0.01)

    def test_default_max_cycles(self):
        assert self.config.max_cycles == 100

    def test_default_safety_flags(self):
        assert self.config.require_human_approval is True
        assert self.config.max_concurrent_implementations == 1
        assert self.config.enable_recursive_self_improvement is False

    def test_custom_overrides(self):
        custom = RDVEConfig(
            snr_floor=0.90,
            ihsan_floor=0.99,
            max_cycles=50,
        )
        assert custom.snr_floor == pytest.approx(0.90)
        assert custom.ihsan_floor == pytest.approx(0.99)
        assert custom.max_cycles == 50


# ============================================================================
# StageResult
# ============================================================================


class TestStageResult:
    def test_construction_with_success(self):
        result = StageResult(
            stage=RDVEStage.OBSERVE,
            success=True,
            duration_ms=42.5,
        )
        assert result.stage == RDVEStage.OBSERVE
        assert result.success is True
        assert result.duration_ms == pytest.approx(42.5)
        assert result.error is None
        assert result.artifacts == {}

    def test_construction_with_error(self):
        result = StageResult(
            stage=RDVEStage.VERIFY,
            success=False,
            duration_ms=100.0,
            error="Constitutional gate rejected hypothesis",
        )
        assert result.success is False
        assert result.error == "Constitutional gate rejected hypothesis"

    def test_construction_with_artifacts(self):
        result = StageResult(
            stage=RDVEStage.FILTER,
            success=True,
            duration_ms=15.0,
            artifacts={"passed": 3, "rejected": 7},
        )
        assert result.artifacts["passed"] == 3
        assert result.artifacts["rejected"] == 7


# ============================================================================
# RDVECycleResult — to_dict and duration_ms
# ============================================================================


class TestRDVECycleResult:
    def test_default_construction(self):
        result = RDVECycleResult()
        assert result.cycle_number == 0
        assert result.outcome == CycleOutcome.NO_SIGNAL
        assert result.hypotheses_generated == 0
        assert result.best_snr_score == pytest.approx(0.0)
        assert result.winning_hypothesis is None

    def test_cycle_id_is_auto_generated(self):
        r1 = RDVECycleResult()
        r2 = RDVECycleResult()
        assert isinstance(r1.cycle_id, str)
        assert len(r1.cycle_id) == 8
        # Unique IDs (uuid-based, extremely unlikely to collide)
        assert r1.cycle_id != r2.cycle_id

    def test_duration_ms_with_timestamps(self):
        start = datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)
        end = start + timedelta(milliseconds=1500)
        result = RDVECycleResult(started_at=start, completed_at=end)
        assert result.duration_ms == pytest.approx(1500.0, abs=1.0)

    def test_duration_ms_without_timestamps(self):
        result = RDVECycleResult()
        assert result.duration_ms == pytest.approx(0.0)

    def test_to_dict_shape(self):
        start = datetime(2026, 2, 20, 12, 0, 0, tzinfo=timezone.utc)
        end = start + timedelta(seconds=2)

        result = RDVECycleResult(
            cycle_number=3,
            outcome=CycleOutcome.DISCOVERY,
            started_at=start,
            completed_at=end,
            hypotheses_generated=10,
            hypotheses_explored=8,
            hypotheses_passed_snr=5,
            hypotheses_verified=2,
            hypotheses_implemented=1,
            best_snr_score=0.95,
            best_ihsan_score=0.97,
            best_confidence=0.88,
            winning_hypothesis={"id": "h1", "description": "Test hypothesis"},
        )

        d = result.to_dict()

        assert d["cycle_number"] == 3
        assert d["outcome"] == "discovery"
        assert d["duration_ms"] == pytest.approx(2000.0, abs=1.0)

        # Hypotheses sub-dict
        assert d["hypotheses"]["generated"] == 10
        assert d["hypotheses"]["explored"] == 8
        assert d["hypotheses"]["passed_snr"] == 5
        assert d["hypotheses"]["verified"] == 2
        assert d["hypotheses"]["implemented"] == 1

        # Quality sub-dict
        assert d["quality"]["best_snr"] == pytest.approx(0.95)
        assert d["quality"]["best_ihsan"] == pytest.approx(0.97)
        assert d["quality"]["best_confidence"] == pytest.approx(0.88)

        assert d["winning_hypothesis"]["id"] == "h1"

    def test_to_dict_with_no_signal(self):
        result = RDVECycleResult(outcome=CycleOutcome.NO_SIGNAL)
        d = result.to_dict()
        assert d["outcome"] == "no_signal"
        assert d["winning_hypothesis"] is None
        assert d["hypotheses"]["generated"] == 0


# ============================================================================
# Module-Level Constants
# ============================================================================


class TestModuleConstants:
    def test_rdve_version_format(self):
        # Semantic version format: X.Y.Z
        parts = RDVE_VERSION.split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()

    def test_rdve_codename_is_string(self):
        assert isinstance(RDVE_CODENAME, str)
        assert len(RDVE_CODENAME) > 0

    def test_standing_on_giants_is_nonempty_list(self):
        assert isinstance(STANDING_ON_GIANTS, list)
        assert len(STANDING_ON_GIANTS) >= 6
