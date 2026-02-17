"""
Phase 31 Integration Tests — Verifies wiring between HyperGraph, CognitiveFusion,
MemoryCoder, AgentDB, and SovereignRuntime.

Standing on: Berge (hypergraph) + Simon (hierarchy) + Shannon (SNR) + Deming (PDCA)
"""
from __future__ import annotations

import pytest

from core.integration.constants import (
    SNR_THRESHOLD_T0_ELITE,
    SNR_THRESHOLD_T1_HIGH,
    SNR_THRESHOLD_T2_STANDARD,
    UNIFIED_IHSAN_THRESHOLD,
    UNIFIED_SNR_THRESHOLD,
)


# ─── HyperGraph ↔ CognitiveFusion ───────────────────────────────────────────


class TestHyperGraphRAGIntegration:
    """Verify HyperGraph and RAG Fusion wire together correctly."""

    def test_rag_fusion_uses_hypergraph_store(self):
        from core.hypergraph import HyperEdgeType, HyperGraphNode, HyperGraphStore
        from core.hypergraph import HyperGraphRAGFusion

        store = HyperGraphStore()
        store.add_node(HyperGraphNode("a", "Node A", "science", embedding=[1.0, 0.0, 0.0]))
        store.add_node(HyperGraphNode("b", "Node B", "science", embedding=[0.9, 0.1, 0.0]))
        store.add_node(HyperGraphNode("c", "Node C", "ethics", embedding=[0.0, 0.0, 1.0]))
        store.add_hyperedge({"a", "b"}, HyperEdgeType.CONCEPT_CLUSTER, weight=0.95)
        store.add_hyperedge({"a", "c"}, HyperEdgeType.CROSS_DOMAIN_BRIDGE, weight=0.80)

        fusion = HyperGraphRAGFusion(store=store, agent_db=None)
        # Without agent_db, vector/keyword sources are empty, but graph traversal works
        results = fusion.retrieve("test query", [1.0, 0.0, 0.0], top_k=5)
        # Results may be empty without agent_db seeds, but call succeeds
        assert isinstance(results, list)

    def test_hypergraph_store_cross_domain_bridges(self):
        from core.hypergraph import HyperEdgeType, HyperGraphNode, HyperGraphStore

        store = HyperGraphStore()
        store.add_node(HyperGraphNode("a", "A", "physics"))
        store.add_node(HyperGraphNode("b", "B", "biology"))
        store.add_node(HyperGraphNode("c", "C", "physics"))
        store.add_hyperedge({"a", "b"}, HyperEdgeType.CROSS_DOMAIN_BRIDGE, weight=0.9)
        store.add_hyperedge({"a", "c"}, HyperEdgeType.CONCEPT_CLUSTER, weight=0.8)

        bridges = store.get_cross_domain_bridges()
        assert len(bridges) == 1
        assert bridges[0].edge_type == HyperEdgeType.CROSS_DOMAIN_BRIDGE


# ─── CognitiveFusion ↔ ComplexityAdapter ─────────────────────────────────────


class TestCognitiveFusionIntegration:
    """Verify CognitiveFusionEngine wires with ComplexityAdapter correctly."""

    def test_engine_processes_query_standalone(self):
        from core.cognitive_fusion import CognitiveFusionEngine, FusionResult

        engine = CognitiveFusionEngine()
        result = engine.process("What is autopoiesis?", [0.5] * 10)

        assert isinstance(result, FusionResult)
        assert result.snr_score >= UNIFIED_SNR_THRESHOLD
        assert result.ihsan_score >= UNIFIED_IHSAN_THRESHOLD

    def test_complexity_adapter_snr_gradient_matches_constants(self):
        from core.cognitive_fusion import ComplexityAdapter

        adapter = ComplexityAdapter()
        _, snr_trivial = adapter.adapt("TRIVIAL")
        _, snr_standard = adapter.adapt("STANDARD")
        _, snr_complex = adapter.adapt("COMPLEX")
        _, snr_expert = adapter.adapt("EXPERT")
        _, snr_frontier = adapter.adapt("FRONTIER")

        assert snr_trivial == UNIFIED_SNR_THRESHOLD
        assert snr_complex == SNR_THRESHOLD_T2_STANDARD
        assert snr_expert == SNR_THRESHOLD_T1_HIGH
        assert snr_frontier == SNR_THRESHOLD_T0_ELITE
        # Monotonically increasing
        assert snr_trivial <= snr_standard <= snr_complex <= snr_expert <= snr_frontier


# ─── MemoryCoder ↔ PatternCodebook ───────────────────────────────────────────


class TestMemoryCoderIntegration:
    """Verify MemorySynthesizer wires with PatternCodebook correctly."""

    def test_synthesizer_codebook_roundtrip(self):
        from core.memory_coder import MemorySynthesizer, PatternCodebook, SynthesizedPattern

        codebook = PatternCodebook()
        synthesizer = MemorySynthesizer(agent_db=None, codebook=codebook)

        # Manually add a pattern
        pattern = SynthesizedPattern(
            pattern_id="test-001",
            embedding=[0.5] * 10,
            keywords=["autopoiesis", "self-repair"],
            snr=0.96,
            source_count=12,
            access_count=100,
        )
        codebook.add(pattern)

        # Verify it's findable
        assert codebook.size == 1
        assert len(codebook.strong_patterns) == 1
        assert codebook.contains_similar(pattern, threshold=0.90)

        # Synthesize cycle with no data returns empty
        result = synthesizer.synthesize_cycle()
        assert isinstance(result, list)

    def test_pattern_is_strong_uses_ssot_threshold(self):
        from core.memory_coder import SynthesizedPattern

        # is_strong checks SNR >= SNR_THRESHOLD_T1_HIGH (0.95)
        strong = SynthesizedPattern("s", [0.1], ["a"], snr=0.96, source_count=15, access_count=50)
        weak = SynthesizedPattern("w", [0.1], ["a"], snr=0.94, source_count=15, access_count=50)
        assert strong.is_strong is True
        assert weak.is_strong is False


# ─── Cross-Package Integration ───────────────────────────────────────────────


class TestCrossPackageIntegration:
    """Verify all 3 packages compose correctly."""

    def test_all_packages_importable_via_core(self):
        import core

        assert hasattr(core, "hypergraph")
        assert hasattr(core, "cognitive_fusion")
        assert hasattr(core, "memory_coder")
        assert core.__version__ == "2.5.0"

    def test_runtime_config_has_fusion_flags(self):
        from core.sovereign.runtime_types import RuntimeConfig

        cfg = RuntimeConfig()
        assert cfg.enable_cognitive_fusion is True
        assert cfg.enable_memory_synthesizer is True
        assert cfg.memory_synthesizer_window_hours == 24

    def test_runtime_state_includes_phase31_components(self):
        """Verify _get_runtime_state includes Phase 31 component status."""
        from unittest.mock import AsyncMock, patch
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        rt = SovereignRuntime(RuntimeConfig.minimal())
        state = rt._get_runtime_state()

        assert "components" in state
        assert "hypergraph_store" in state["components"]
        assert "cognitive_fusion" in state["components"]
        assert "memory_synthesizer" in state["components"]
        # Not initialized yet, so should be False
        assert state["components"]["hypergraph_store"] is False
        assert state["components"]["cognitive_fusion"] is False
        assert state["components"]["memory_synthesizer"] is False

    def test_cognitive_fusion_init_standalone(self):
        """Verify _init_cognitive_fusion can run without crashing."""
        from core.sovereign.runtime_core import SovereignRuntime
        from core.sovereign.runtime_types import RuntimeConfig

        rt = SovereignRuntime(RuntimeConfig.minimal())
        rt._init_cognitive_fusion()

        # After init, components should be initialized
        assert rt._hypergraph_store is not None
        assert rt._cognitive_fusion is not None
        assert rt._memory_synthesizer is not None
        assert rt._pattern_codebook is not None
