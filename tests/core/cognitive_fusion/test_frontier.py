"""Tests for P4: FRONTIER tier activation in CognitiveFusionEngine.

Covers:
- frontier_mode=False (default): FRONTIER queries treated as EXPERT-equivalent
- frontier_mode=True: Dedicated FRONTIER pipeline with:
  - SNR gate raised to T0_ELITE (0.98)
  - GoT max_depth doubled (GOT_MAX_DEPTH * 2)
  - Cross-domain RAG retrieval (depth >= 50)
  - SYNTHESIS consciousness event with domains_crossed metric
- Backward compatibility: non-FRONTIER queries unaffected by frontier_mode

Blueprint Reference: Elite Implementation Blueprint v1.0 — P4 FRONTIER Tier
"""

from unittest.mock import MagicMock


from core.cognitive_fusion.fusion_engine import (
    CognitiveFusionEngine,
    FusionResult,
    HRMResult,
    NorthStarResult,
    RoutingResult,
)
from core.integration.constants import (
    GOT_MAX_DEPTH,
    SNR_THRESHOLD_T0_ELITE,
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)


def _mock_moe_router(complexity: str = "FRONTIER"):
    """Create a mock MoE router that returns the given complexity."""
    router = MagicMock()
    router.route.return_value = RoutingResult(
        complexity_class=complexity,
        expert_tier="FRONTIER" if complexity == "FRONTIER" else "POOL",
        confidence=0.95,
    )
    return router


def _mock_hrm_engine(snr: float = 0.99):
    """Create a mock HRM engine."""
    engine = MagicMock()
    engine.run_cycle.return_value = HRMResult(
        compound_snr=snr,
        level_reached="META_COGNITIVE",
        observations=["frontier reasoning"],
    )
    return engine


def _mock_rag(results=None):
    """Create a mock RAG retriever."""
    rag = MagicMock()
    if results is None:
        results = [
            {"domain": "security", "text": "finding 1"},
            {"domain": "economics", "text": "finding 2"},
            {"domain": "governance", "text": "finding 3"},
        ]
    rag.retrieve.return_value = results
    return rag


def _mock_northstar(snr: float = 0.99, ihsan: float = 0.99, passes: bool = True):
    """Create a mock NorthStar gate."""
    ns = MagicMock()
    ns.run_cycle.return_value = NorthStarResult(
        unified_snr=snr,
        ihsan_score=ihsan,
        passes_all_gates=passes,
    )
    return ns


# ═══════════════════════════════════════════════════════════════════════════
# FRONTIER mode disabled (backward compatibility)
# ═══════════════════════════════════════════════════════════════════════════


class TestFrontierModeDisabled:
    """When frontier_mode=False (default), FRONTIER queries use standard pipeline."""

    def test_default_frontier_mode_is_false(self):
        engine = CognitiveFusionEngine()
        assert engine._frontier_mode is False

    def test_frontier_query_uses_standard_pipeline(self):
        """FRONTIER query without frontier_mode → no special handling."""
        router = _mock_moe_router("FRONTIER")
        hrm = _mock_hrm_engine(snr=0.90)
        rag = _mock_rag()
        ns = _mock_northstar(snr=0.90, ihsan=0.96)

        engine = CognitiveFusionEngine(
            moe_router=router,
            hrm_engine=hrm,
            hypergraph_rag=rag,
            northstar_engine=ns,
            frontier_mode=False,
        )
        result = engine.process("multi-domain question", [0.1, 0.2])

        assert isinstance(result, FusionResult)
        assert result.routing.complexity_class == "FRONTIER"
        # Standard pipeline — RAG uses default depth, not forced to 50
        rag.retrieve.assert_called_once()
        _, kwargs = rag.retrieve.call_args
        assert kwargs.get("top_k", 50) == 50  # FRONTIER default is 50

    def test_non_frontier_query_unaffected(self):
        """Non-FRONTIER queries work normally regardless of frontier_mode."""
        router = _mock_moe_router("STANDARD")
        hrm = _mock_hrm_engine(snr=0.90)
        ns = _mock_northstar(snr=0.90, ihsan=0.96)

        engine = CognitiveFusionEngine(
            moe_router=router,
            hrm_engine=hrm,
            northstar_engine=ns,
            frontier_mode=True,
        )
        result = engine.process("simple question", [0.1])

        assert result.routing.complexity_class == "STANDARD"
        assert result.target_level == "OPERATIONAL"


# ═══════════════════════════════════════════════════════════════════════════
# FRONTIER mode enabled
# ═══════════════════════════════════════════════════════════════════════════


class TestFrontierModeEnabled:
    """When frontier_mode=True and complexity=FRONTIER, dedicated pipeline activates."""

    def test_frontier_flag_stored(self):
        engine = CognitiveFusionEngine(frontier_mode=True)
        assert engine._frontier_mode is True

    def test_snr_gate_raised_to_t0_elite(self):
        """FRONTIER queries require SNR >= T0_ELITE (0.98)."""
        router = _mock_moe_router("FRONTIER")
        # HRM and NS both return high SNR
        hrm = _mock_hrm_engine(snr=0.99)
        ns = _mock_northstar(snr=0.99, ihsan=0.99)

        engine = CognitiveFusionEngine(
            moe_router=router,
            hrm_engine=hrm,
            northstar_engine=ns,
            frontier_mode=True,
        )
        result = engine.process("frontier question", [0.1])

        # Should pass — aggregate SNR = sqrt(0.99 * 0.99) = 0.99 >= 0.98
        assert result.passes_gate is True
        assert result.snr_score >= SNR_THRESHOLD_T0_ELITE

    def test_snr_below_t0_elite_fails_gate(self):
        """FRONTIER with SNR below T0_ELITE fails the gate."""
        router = _mock_moe_router("FRONTIER")
        # Marginal SNR — below T0_ELITE after aggregation
        hrm = _mock_hrm_engine(snr=0.90)
        ns = _mock_northstar(snr=0.90, ihsan=0.99)

        engine = CognitiveFusionEngine(
            moe_router=router,
            hrm_engine=hrm,
            northstar_engine=ns,
            frontier_mode=True,
        )
        result = engine.process("frontier question", [0.1])

        # Aggregate = sqrt(0.90 * 0.90) = 0.90 < 0.98 → fails
        assert result.passes_gate is False

    def test_got_max_depth_doubled(self):
        """HRM engine receives got_max_depth = GOT_MAX_DEPTH * 2."""
        router = _mock_moe_router("FRONTIER")
        hrm = _mock_hrm_engine()
        ns = _mock_northstar()

        engine = CognitiveFusionEngine(
            moe_router=router,
            hrm_engine=hrm,
            northstar_engine=ns,
            frontier_mode=True,
        )
        engine.process("deep reasoning query", [0.1])

        # HRM should have been called with context containing doubled depth
        hrm.run_cycle.assert_called_once()
        observation = hrm.run_cycle.call_args[0][0]
        assert observation["got_max_depth"] == GOT_MAX_DEPTH * 2
        assert observation["frontier_mode"] is True

    def test_cross_domain_rag_depth(self):
        """FRONTIER forces RAG retrieval depth >= 50."""
        router = _mock_moe_router("FRONTIER")
        rag = _mock_rag()
        ns = _mock_northstar()

        engine = CognitiveFusionEngine(
            moe_router=router,
            hypergraph_rag=rag,
            northstar_engine=ns,
            frontier_mode=True,
        )
        engine.process("cross-domain question", [0.1])

        rag.retrieve.assert_called_once()
        _, kwargs = rag.retrieve.call_args
        assert kwargs["top_k"] >= 50

    def test_northstar_receives_frontier_flag(self):
        """NorthStar observation includes frontier_mode=True."""
        router = _mock_moe_router("FRONTIER")
        ns = _mock_northstar()

        engine = CognitiveFusionEngine(
            moe_router=router,
            northstar_engine=ns,
            frontier_mode=True,
        )
        engine.process("frontier analysis", [0.1])

        ns.run_cycle.assert_called_once()
        observation = ns.run_cycle.call_args[0][0]
        assert observation["frontier_mode"] is True

    def test_non_frontier_query_not_affected(self):
        """STANDARD queries use normal pipeline even with frontier_mode=True."""
        router = _mock_moe_router("STANDARD")
        hrm = _mock_hrm_engine(snr=0.88)
        ns = _mock_northstar(snr=0.88, ihsan=0.96)

        engine = CognitiveFusionEngine(
            moe_router=router,
            hrm_engine=hrm,
            northstar_engine=ns,
            frontier_mode=True,
        )
        result = engine.process("normal question", [0.1])

        # HRM should NOT receive frontier context
        observation = hrm.run_cycle.call_args[0][0]
        assert "got_max_depth" not in observation
        assert "frontier_mode" not in observation

        # Standard SNR requirement applies (0.85), not T0_ELITE
        assert result.passes_gate is True

    def test_process_returns_fusion_result(self):
        """Full pipeline returns complete FusionResult."""
        router = _mock_moe_router("FRONTIER")
        hrm = _mock_hrm_engine(snr=0.99)
        rag = _mock_rag()
        ns = _mock_northstar(snr=0.99, ihsan=0.99)

        engine = CognitiveFusionEngine(
            moe_router=router,
            hrm_engine=hrm,
            hypergraph_rag=rag,
            northstar_engine=ns,
            frontier_mode=True,
        )
        result = engine.process("frontier synthesis", [0.1, 0.2])

        assert isinstance(result, FusionResult)
        assert result.routing.complexity_class == "FRONTIER"
        assert result.target_level == "META_COGNITIVE"
        assert result.snr_score >= SNR_THRESHOLD_T0_ELITE
        assert result.ihsan_score >= UNIFIED_IHSAN_THRESHOLD
        assert result.passes_gate is True
        assert result.is_elite is True
        assert len(result.retrieval) == 3


# ═══════════════════════════════════════════════════════════════════════════
# Edge cases
# ═══════════════════════════════════════════════════════════════════════════


class TestFrontierEdgeCases:

    def test_frontier_no_subsystems(self):
        """FRONTIER with no subsystems still returns valid result."""
        engine = CognitiveFusionEngine(frontier_mode=True)
        result = engine.process("frontier without engines", [0.1])

        # Default routing is STANDARD (no MoE router) → not FRONTIER path
        assert result.routing.complexity_class == "STANDARD"
        assert isinstance(result, FusionResult)

    def test_frontier_empty_rag_results(self):
        """FRONTIER with empty RAG results still passes."""
        router = _mock_moe_router("FRONTIER")
        rag = _mock_rag(results=[])
        ns = _mock_northstar(snr=0.99, ihsan=0.99)

        engine = CognitiveFusionEngine(
            moe_router=router,
            hypergraph_rag=rag,
            northstar_engine=ns,
            frontier_mode=True,
        )
        result = engine.process("frontier no context", [0.1])

        assert result.retrieval == []
        # Still passes if SNR meets threshold
        assert result.snr_score >= UNIFIED_SNR_THRESHOLD

    def test_frontier_moe_failure_falls_back(self):
        """If MoE router fails, fallback is STANDARD → no frontier activation."""
        router = MagicMock()
        router.route.side_effect = RuntimeError("router down")
        ns = _mock_northstar()

        engine = CognitiveFusionEngine(
            moe_router=router,
            northstar_engine=ns,
            frontier_mode=True,
        )
        result = engine.process("test", [0.1])

        # Fallback to STANDARD — not FRONTIER path
        assert result.routing.complexity_class == "STANDARD"
