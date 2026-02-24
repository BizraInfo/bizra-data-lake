"""
Cognitive Fusion Pipeline Integration Tests — Verifies the wiring of
CognitiveFusionEngine into the Sovereign query pipeline (Stage 1.5)
and the new /v1/cognitive/* API endpoints.

Standing on: Vaswani (MoE) + Simon (hierarchy) + Shannon (SNR) + Besta (GoT)

Created: 2026-02-17 | Phase 31.1 — Pipeline Integration
"""

from __future__ import annotations

import pytest

from core.integration.constants import (
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)

# ─── CognitiveFusionEngine standalone pipeline ─────────────────────────────


class TestCognitiveFusionProcess:
    """Verify the 4-stage pipeline returns valid FusionResult."""

    def test_process_returns_fusion_result(self):
        from core.cognitive_fusion import CognitiveFusionEngine, FusionResult

        engine = CognitiveFusionEngine()
        result = engine.process("What is entropy?", [0.1] * 768)

        assert isinstance(result, FusionResult)
        assert result.routing.complexity_class in (
            "TRIVIAL",
            "STANDARD",
            "COMPLEX",
            "EXPERT",
            "FRONTIER",
        )

    def test_process_passes_gate_with_defaults(self):
        from core.cognitive_fusion import CognitiveFusionEngine

        engine = CognitiveFusionEngine()
        result = engine.process("Simple question", [0.0] * 768)

        assert result.passes_gate is True
        assert result.snr_score >= UNIFIED_SNR_THRESHOLD
        assert result.ihsan_score >= UNIFIED_IHSAN_THRESHOLD

    def test_process_default_complexity_is_standard(self):
        from core.cognitive_fusion import CognitiveFusionEngine

        engine = CognitiveFusionEngine()
        result = engine.process("Test", [0.0] * 10)

        # Without MoE router, defaults to STANDARD
        assert result.routing.complexity_class == "STANDARD"

    def test_process_expert_tier_from_adapter(self):
        from core.cognitive_fusion import CognitiveFusionEngine

        engine = CognitiveFusionEngine()
        result = engine.process("Query", [0.0] * 10)

        assert result.expert_tier in ("NANO", "EDGE", "LOCAL", "POOL", "FRONTIER")

    def test_process_target_level_assigned(self):
        from core.cognitive_fusion import CognitiveFusionEngine

        engine = CognitiveFusionEngine()
        result = engine.process("Query", [0.0] * 10)

        assert result.target_level in (
            "PERCEPTUAL",
            "OPERATIONAL",
            "TACTICAL",
            "STRATEGIC",
            "META_COGNITIVE",
        )

    def test_process_empty_retrieval_without_rag(self):
        from core.cognitive_fusion import CognitiveFusionEngine

        engine = CognitiveFusionEngine()
        result = engine.process("No RAG", [0.0] * 10)

        assert result.retrieval == []

    def test_process_with_context(self):
        from core.cognitive_fusion import CognitiveFusionEngine

        engine = CognitiveFusionEngine()
        result = engine.process(
            "Query with context",
            [0.0] * 10,
            context={"domain": "science", "priority": "high"},
        )

        assert isinstance(result.routing.metadata, dict)


# ─── Complexity Adapter mapping ────────────────────────────────────────────


class TestComplexityAdapterMapping:
    """Verify complexity → level → tier mapping chain."""

    def test_all_complexity_classes_map_to_level(self):
        from core.cognitive_fusion import ComplexityAdapter

        adapter = ComplexityAdapter()
        for cls in ("TRIVIAL", "STANDARD", "COMPLEX", "EXPERT", "FRONTIER"):
            level, snr = adapter.adapt(cls)
            assert level in (
                "PERCEPTUAL",
                "OPERATIONAL",
                "TACTICAL",
                "STRATEGIC",
                "META_COGNITIVE",
            )
            assert snr > 0.0

    def test_snr_increases_with_complexity(self):
        from core.cognitive_fusion import ComplexityAdapter

        adapter = ComplexityAdapter()
        _, snr_trivial = adapter.adapt("TRIVIAL")
        _, snr_expert = adapter.adapt("EXPERT")
        _, snr_frontier = adapter.adapt("FRONTIER")

        assert snr_trivial <= snr_expert <= snr_frontier

    def test_level_to_tier_maps_all_levels(self):
        from core.cognitive_fusion import ComplexityAdapter

        adapter = ComplexityAdapter()
        for level in (
            "PERCEPTUAL",
            "OPERATIONAL",
            "TACTICAL",
            "STRATEGIC",
            "META_COGNITIVE",
        ):
            tier = adapter.level_to_tier(level)
            assert tier in ("NANO", "EDGE", "LOCAL", "POOL", "FRONTIER")


# ─── Runtime integration helpers (mock-free) ───────────────────────────────


class TestRuntimeFusionHelpers:
    """Verify _run_cognitive_fusion and _enrich_prompt_with_fusion."""

    def test_enrich_prompt_with_empty_retrieval(self):
        from core.sovereign.runtime_core import SovereignRuntime

        prompt = "What is X?"

        # Create a mock fusion result with no retrieval
        class FakeFusion:
            retrieval = []

        result = SovereignRuntime._enrich_prompt_with_fusion(prompt, FakeFusion())
        assert result == prompt  # No enrichment when no retrieval

    def test_enrich_prompt_with_retrieval(self):
        from core.sovereign.runtime_core import SovereignRuntime

        prompt = "What is X?"

        class FakeItem:
            content = "Context about X from knowledge base"

        class FakeFusion:
            retrieval = [FakeItem(), FakeItem()]

        enriched = SovereignRuntime._enrich_prompt_with_fusion(prompt, FakeFusion())
        assert "[Retrieved Context" in enriched
        assert "Context about X" in enriched
        assert prompt in enriched

    def test_enrich_prompt_with_dict_retrieval(self):
        from core.sovereign.runtime_core import SovereignRuntime

        prompt = "Query"

        class FakeFusion:
            retrieval = [
                {"content": "Dict item 1"},
                {"content": "Dict item 2"},
            ]

        enriched = SovereignRuntime._enrich_prompt_with_fusion(prompt, FakeFusion())
        assert "Dict item 1" in enriched
        assert "Dict item 2" in enriched

    def test_enrich_prompt_with_string_retrieval(self):
        from core.sovereign.runtime_core import SovereignRuntime

        prompt = "Query"

        class FakeFusion:
            retrieval = ["Raw string chunk 1", "Raw string chunk 2"]

        enriched = SovereignRuntime._enrich_prompt_with_fusion(prompt, FakeFusion())
        assert "Raw string chunk 1" in enriched

    def test_enrich_prompt_limits_to_5_chunks(self):
        from core.sovereign.runtime_core import SovereignRuntime

        prompt = "Q"

        class FakeFusion:
            retrieval = [f"Chunk {i}" for i in range(20)]

        enriched = SovereignRuntime._enrich_prompt_with_fusion(prompt, FakeFusion())
        assert "(5 sources)" in enriched
        assert "Chunk 4" in enriched
        assert "Chunk 5" not in enriched  # 0-indexed, limit is [:5]

    def test_enrich_prompt_truncates_long_chunks(self):
        from core.sovereign.runtime_core import SovereignRuntime

        prompt = "Q"

        class FakeFusion:
            retrieval = ["A" * 1000]  # Over 500 char limit

        enriched = SovereignRuntime._enrich_prompt_with_fusion(prompt, FakeFusion())
        # Original prompt is preserved, chunk is truncated
        assert "Q" in enriched
        assert len(enriched) < 1200  # Should be truncated


# ─── Retrieval depth scaling ───────────────────────────────────────────────


class TestRetrievalDepthScaling:
    """Verify complexity → retrieval depth mapping."""

    def test_trivial_gets_shallow_retrieval(self):
        from core.cognitive_fusion import CognitiveFusionEngine

        assert CognitiveFusionEngine._retrieval_depth("TRIVIAL") == 3

    def test_standard_gets_moderate_retrieval(self):
        from core.cognitive_fusion import CognitiveFusionEngine

        assert CognitiveFusionEngine._retrieval_depth("STANDARD") == 5

    def test_complex_gets_deeper_retrieval(self):
        from core.cognitive_fusion import CognitiveFusionEngine

        assert CognitiveFusionEngine._retrieval_depth("COMPLEX") == 10

    def test_expert_gets_extensive_retrieval(self):
        from core.cognitive_fusion import CognitiveFusionEngine

        assert CognitiveFusionEngine._retrieval_depth("EXPERT") == 20

    def test_frontier_gets_maximum_retrieval(self):
        from core.cognitive_fusion import CognitiveFusionEngine

        assert CognitiveFusionEngine._retrieval_depth("FRONTIER") == 50

    def test_unknown_complexity_defaults(self):
        from core.cognitive_fusion import CognitiveFusionEngine

        assert CognitiveFusionEngine._retrieval_depth("UNKNOWN") == 5

    def test_depth_monotonically_increases(self):
        from core.cognitive_fusion import CognitiveFusionEngine

        classes = ["TRIVIAL", "STANDARD", "COMPLEX", "EXPERT", "FRONTIER"]
        depths = [CognitiveFusionEngine._retrieval_depth(c) for c in classes]
        for i in range(len(depths) - 1):
            assert depths[i] <= depths[i + 1]


# ─── SNR aggregation ──────────────────────────────────────────────────────


class TestSNRAggregation:
    """Verify geometric mean SNR aggregation."""

    def test_aggregate_snr_geometric_mean(self):
        from core.cognitive_fusion.fusion_engine import (
            CognitiveFusionEngine,
            HRMResult,
            NorthStarResult,
        )

        hrm = HRMResult(compound_snr=0.90)
        ns = NorthStarResult(unified_snr=0.95)
        score = CognitiveFusionEngine._aggregate_snr(hrm, ns)

        # Geometric mean of 0.90 * 0.95 = sqrt(0.855)
        expected = (0.90 * 0.95) ** 0.5
        assert abs(score - expected) < 1e-6

    def test_aggregate_snr_penalizes_low_scores(self):
        from core.cognitive_fusion.fusion_engine import (
            CognitiveFusionEngine,
            HRMResult,
            NorthStarResult,
        )

        # One high, one low — geometric mean pulls down
        hrm = HRMResult(compound_snr=0.99)
        ns = NorthStarResult(unified_snr=0.50)
        score = CognitiveFusionEngine._aggregate_snr(hrm, ns)

        assert score < 0.80  # Penalized by the low NorthStar score

    def test_aggregate_snr_rewards_consistency(self):
        from core.cognitive_fusion.fusion_engine import (
            CognitiveFusionEngine,
            HRMResult,
            NorthStarResult,
        )

        # Both high — geometric mean stays high
        hrm = HRMResult(compound_snr=0.95)
        ns = NorthStarResult(unified_snr=0.95)
        score = CognitiveFusionEngine._aggregate_snr(hrm, ns)

        assert score >= 0.94


# ─── FusionResult properties ──────────────────────────────────────────────


class TestFusionResultProperties:
    """Verify FusionResult computed properties."""

    def test_is_elite_requires_both_thresholds(self):
        from core.cognitive_fusion.fusion_engine import (
            FusionResult,
            HRMResult,
            NorthStarResult,
            RoutingResult,
        )
        from core.integration.constants import (
            SNR_THRESHOLD_T0_ELITE,
            STRICT_IHSAN_THRESHOLD,
        )

        result = FusionResult(
            routing=RoutingResult(),
            hrm_result=HRMResult(),
            retrieval=[],
            northstar_report=NorthStarResult(),
            target_level="SOVEREIGN",
            snr_score=SNR_THRESHOLD_T0_ELITE,
            ihsan_score=STRICT_IHSAN_THRESHOLD,
            passes_gate=True,
        )

        assert result.is_elite is True

    def test_is_not_elite_with_low_snr(self):
        from core.cognitive_fusion.fusion_engine import (
            FusionResult,
            HRMResult,
            NorthStarResult,
            RoutingResult,
        )

        result = FusionResult(
            routing=RoutingResult(),
            hrm_result=HRMResult(),
            retrieval=[],
            northstar_report=NorthStarResult(),
            target_level="OPERATIONAL",
            snr_score=0.85,  # Below T0 elite
            ihsan_score=0.99,
            passes_gate=True,
        )

        assert result.is_elite is False


# ─── API endpoint model validation ────────────────────────────────────────


class TestCognitiveFuseAPIModel:
    """Verify the CognitiveFuseModel Pydantic schema."""

    def test_model_accepts_query_only(self):
        from core.sovereign.api import CognitiveFuseModel

        if CognitiveFuseModel is None:
            pytest.skip("Pydantic not available")

        model = CognitiveFuseModel(query="Test query")
        assert model.query == "Test query"
        assert model.context == {}

    def test_model_accepts_query_with_context(self):
        from core.sovereign.api import CognitiveFuseModel

        if CognitiveFuseModel is None:
            pytest.skip("Pydantic not available")

        model = CognitiveFuseModel(
            query="Complex query",
            context={"domain": "quantum", "priority": 1},
        )
        assert model.query == "Complex query"
        assert model.context["domain"] == "quantum"

    def test_model_rejects_missing_query(self):
        from core.sovereign.api import CognitiveFuseModel

        if CognitiveFuseModel is None:
            pytest.skip("Pydantic not available")

        with pytest.raises(Exception):
            CognitiveFuseModel()
