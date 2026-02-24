"""
Cognitive Fusion Module -- Test Suite (10 Tests)

Validates:
  A. CognitiveFusionEngine pipeline (tests 1-3)
  B. FusionResult properties (tests 4-5)
  C. Retrieval depth scaling (test 6)
  D. ComplexityAdapter mappings (tests 7-10)

Constitutional Alignment:
  All expected values reference core/integration/constants.py (SSOT).
  No hardcoded threshold literals in assertions -- always compare
  against the imported constant.

Created: 2026-02-17 | BIZRA Node0 | Cognitive Fusion Phase
"""

from __future__ import annotations

from core.cognitive_fusion.complexity_adapter import ComplexityAdapter
from core.cognitive_fusion.fusion_engine import (
    CognitiveFusionEngine,
    FusionResult,
    HRMResult,
    NorthStarResult,
    RoutingResult,
)
from core.integration.constants import (
    SNR_THRESHOLD_T0_ELITE,
    STRICT_IHSAN_THRESHOLD,
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)

# =============================================================================
# HELPERS
# =============================================================================

_DUMMY_EMBEDDING: list[float] = [0.1, 0.2, 0.3]


def _build_fusion_result(
    *,
    snr: float = UNIFIED_SNR_THRESHOLD,
    ihsan: float = UNIFIED_IHSAN_THRESHOLD,
    complexity: str = "STANDARD",
    target_level: str = "OPERATIONAL",
    passes: bool = True,
) -> FusionResult:
    """Factory for FusionResult with controllable scores."""
    return FusionResult(
        routing=RoutingResult(complexity_class=complexity, expert_tier="EDGE"),
        hrm_result=HRMResult(compound_snr=snr, level_reached=target_level),
        retrieval=[],
        northstar_report=NorthStarResult(
            unified_snr=snr, ihsan_score=ihsan, passes_all_gates=passes
        ),
        target_level=target_level,
        snr_score=snr,
        ihsan_score=ihsan,
        passes_gate=passes,
    )


# =============================================================================
# A. CognitiveFusionEngine Pipeline
# =============================================================================


class TestFusionEnginePipeline:
    """Test the engine processes queries end-to-end with default deps."""

    def test_fusion_engine_process_without_dependencies(self) -> None:
        """Engine with all None deps returns a valid FusionResult."""
        engine = CognitiveFusionEngine()
        result = engine.process("What is autopoiesis?", _DUMMY_EMBEDDING)

        assert isinstance(result, FusionResult)
        assert result.routing.complexity_class == "STANDARD"
        assert result.target_level == "OPERATIONAL"
        assert result.snr_score >= UNIFIED_SNR_THRESHOLD
        assert result.ihsan_score >= UNIFIED_IHSAN_THRESHOLD
        assert result.passes_gate is True
        assert isinstance(result.retrieval, list)

    def test_fusion_engine_maps_trivial_to_perceptual(self) -> None:
        """TRIVIAL complexity routes to PERCEPTUAL level via ComplexityAdapter."""
        adapter = ComplexityAdapter()
        level, snr = adapter.adapt("TRIVIAL")

        assert level == "PERCEPTUAL"
        assert snr == UNIFIED_SNR_THRESHOLD

    def test_fusion_engine_maps_frontier_to_meta_cognitive(self) -> None:
        """FRONTIER complexity routes to META_COGNITIVE level."""
        adapter = ComplexityAdapter()
        level, snr = adapter.adapt("FRONTIER")

        assert level == "META_COGNITIVE"
        assert snr == SNR_THRESHOLD_T0_ELITE


# =============================================================================
# B. FusionResult Properties
# =============================================================================


class TestFusionResultProperties:
    """Validate computed properties on FusionResult."""

    def test_fusion_result_is_elite(self) -> None:
        """is_elite is True when SNR >= T0_ELITE and Ihsan >= STRICT."""
        result = _build_fusion_result(
            snr=SNR_THRESHOLD_T0_ELITE,
            ihsan=STRICT_IHSAN_THRESHOLD,
        )
        assert result.is_elite is True

    def test_fusion_result_not_elite_low_snr(self) -> None:
        """is_elite is False when SNR is below T0_ELITE."""
        result = _build_fusion_result(
            snr=UNIFIED_SNR_THRESHOLD,
            ihsan=STRICT_IHSAN_THRESHOLD,
        )
        assert result.is_elite is False

    def test_fusion_result_expert_tier_and_compound_snr(self) -> None:
        """expert_tier and compound_snr delegate to child dataclasses."""
        result = _build_fusion_result(snr=0.92)
        assert result.expert_tier == "EDGE"
        assert result.compound_snr == 0.92


# =============================================================================
# C. Retrieval Depth Scaling
# =============================================================================


class TestRetrievalDepth:
    """Verify depth scales with complexity class."""

    def test_retrieval_depth_scaling(self) -> None:
        """TRIVIAL=3, STANDARD=5, COMPLEX=10, EXPERT=20, FRONTIER=50."""
        assert CognitiveFusionEngine._retrieval_depth("TRIVIAL") == 3
        assert CognitiveFusionEngine._retrieval_depth("STANDARD") == 5
        assert CognitiveFusionEngine._retrieval_depth("COMPLEX") == 10
        assert CognitiveFusionEngine._retrieval_depth("EXPERT") == 20
        assert CognitiveFusionEngine._retrieval_depth("FRONTIER") == 50
        # Unknown defaults to 5
        assert CognitiveFusionEngine._retrieval_depth("UNKNOWN") == 5


# =============================================================================
# D. ComplexityAdapter Mappings
# =============================================================================


class TestComplexityAdapter:
    """Validate all adapter mapping paths."""

    def test_complexity_adapter_adapt_all_levels(self) -> None:
        """All 5 complexity classes map to the correct HRM level."""
        adapter = ComplexityAdapter()
        expected = {
            "TRIVIAL": "PERCEPTUAL",
            "STANDARD": "OPERATIONAL",
            "COMPLEX": "TACTICAL",
            "EXPERT": "STRATEGIC",
            "FRONTIER": "META_COGNITIVE",
        }
        for complexity, level in expected.items():
            result_level, _ = adapter.adapt(complexity)
            assert result_level == level, f"{complexity} should map to {level}"

    def test_complexity_adapter_snr_gradient(self) -> None:
        """SNR requirement for TRIVIAL < FRONTIER (monotonic gradient)."""
        adapter = ComplexityAdapter()
        _, snr_trivial = adapter.adapt("TRIVIAL")
        _, snr_standard = adapter.adapt("STANDARD")
        _, snr_complex = adapter.adapt("COMPLEX")
        _, snr_expert = adapter.adapt("EXPERT")
        _, snr_frontier = adapter.adapt("FRONTIER")

        assert snr_trivial <= snr_standard
        assert snr_standard <= snr_complex
        assert snr_complex <= snr_expert
        assert snr_expert <= snr_frontier
        assert snr_trivial < snr_frontier  # strict inequality end-to-end

    def test_complexity_adapter_level_to_tier(self) -> None:
        """Each HRM level maps to the correct expert tier."""
        expected = {
            "PERCEPTUAL": "NANO",
            "OPERATIONAL": "EDGE",
            "TACTICAL": "LOCAL",
            "STRATEGIC": "POOL",
            "META_COGNITIVE": "FRONTIER",
        }
        for level, tier in expected.items():
            assert ComplexityAdapter.level_to_tier(level) == tier

    def test_complexity_adapter_unknown_complexity(self) -> None:
        """Unknown complexity defaults to OPERATIONAL / base SNR."""
        adapter = ComplexityAdapter()
        level, snr = adapter.adapt("QUANTUM_WARP")
        assert level == "OPERATIONAL"
        assert snr == UNIFIED_SNR_THRESHOLD
